"""L2DM trainer (Sun et al. 2023) -- SVDiff backbone + TAME + ECD + rainbow memory.

Total objective (Eq. 7), per task k:
    L = L_diff(current task)
      + alpha * E||eps - eps_theta(z_t^l|c^l)||^2      (TAME replay, l < k)
      + beta  * L_PR                                    (DreamBooth prior preservation)
      + gamma * E||eps_{theta^{k-1}} - eps_theta||^2    (ECD distillation, same memory samples)

Paper hyperparameters: SVDiff spectral shifts (UNet + text encoder), lr 1.5e-3, 500 steps/task,
batch 2, 200 prior images per class, rare identifier token (not learned). alpha/beta/gamma are
not stated in the paper and are swept.

Run:  python -u -m src.train_l2dm --config configs/baseline_l2dm.yaml --alpha 1 --beta 1 --gamma 1
"""

from __future__ import annotations

import argparse
import itertools
import os
from contextlib import contextmanager

import torch

from .common import load_config, set_seed
from .data import ConceptDataset, collate_fn, specs_from_config
from .l2dm import MemoryBanks, load_prior_batch, rainbow_scores, tame_replay_loss
from .losses import reconstruction_loss
from .sampling import ddim_sample
from .sd_loader import load_sd
from .svdiff import inject_svdiff, load_shift_state, shift_state, spectral_shifts


class _Plain(torch.nn.Module):
    """Manager stub: SVDiff changes the weights themselves, so no adapter cache is needed.

    Subclasses nn.Module on purpose -- gen_cifc drives every method through the same manager
    interface, so eval()/load_state_dict() have to exist. The attributes below are the rest of
    that interface; keep them in sync with what sampling.py / gen_cifc.py / injection.py read
    (grep -hoE "manager\\.[a-zA-Z_]+|hyper\\.[a-zA-Z_]+" over those three files lists them all).
    """
    lora_enabled = False
    lora_scale = 1.0
    lora_scale_map = None
    solo_task = None
    compose_tasks = None
    _active = 0

    def get_cached_lora(self, name):
        return None

    def get_token_mask(self):
        return None

    def enable_lora(self):
        pass

    def disable_lora(self):
        pass

    @contextmanager
    def no_lora(self):
        yield

    def set_context(self, *a, **kw):
        pass

    def eval_only_grads(self):
        return []

    def compute_and_cache_loras(self, *a, **kw):
        pass


def parse_args():
    p = argparse.ArgumentParser(description="L2DM (SVDiff + TAME + ECD + rainbow memory)")
    p.add_argument("--config", required=True)
    p.add_argument("--alpha", type=float, default=None, help="TAME replay weight")
    p.add_argument("--beta", type=float, default=None, help="prior-preservation weight")
    p.add_argument("--gamma", type=float, default=None, help="ECD distillation weight")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    bl = cfg.get("baseline", {})
    alpha = float(args.alpha if args.alpha is not None else bl.get("alpha", 1.0))
    beta = float(args.beta if args.beta is not None else bl.get("beta", 1.0))
    gamma = float(args.gamma if args.gamma is not None else bl.get("gamma", 1.0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 2024))
    set_seed(seed)

    train = cfg.get("training", {})
    steps_per_task = int(train.get("steps_per_task", 500))
    batch_size = int(train.get("batch_size", 2))
    lr = float(cfg.get("svdiff", {}).get("lr", 1.5e-3))
    resolution = int(cfg.get("resolution", 512))
    log_every = int(train.get("log_every", 100))
    eta = int(bl.get("eta", 20))               # generated candidates per past concept
    short_mem = int(bl.get("short_mem", 5))    # kept per past concept
    prior_dir = bl.get("prior_dir", "data/prior")
    beta_s = float(bl.get("beta_s", 1.0))
    output_dir = args.output_dir or cfg.get("output_dir", "./outputs/baseline/l2dm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _mid = cfg.get("sd_model_id", "")
    _wd = cfg.get("weight_dtype", "fp32")
    if "xl" in str(_mid).lower():
        from .sd_loader import load_sdxl
        bundle = load_sdxl(model_id=_mid, device=device, dtype=_wd)
    else:
        bundle = load_sd(model_id=_mid, device=device, dtype=_wd)
    specs = specs_from_config(cfg["concepts"])
    mgr = _Plain()
    bundle.unet.hyper = mgr

    swapped = inject_svdiff(bundle.unet)
    if cfg.get("svdiff", {}).get("text_encoder", True):
        swapped += inject_svdiff(bundle.text_encoder)
    deltas = spectral_shifts(swapped)
    print(f"[L2DM] SVDiff: {len(swapped)} warstw, {sum(d.numel() for d in deltas)/1e6:.3f}M "
          f"spectral shifts | alpha={alpha} beta={beta} gamma={gamma} seed={seed}", flush=True)

    from .cifc_metrics import _Clip                      # CLIP features for the rainbow bank
    clip = _Clip(device)
    banks = MemoryBanks()
    teacher_state = None
    unet = bundle.unet

    wandb = None
    if cfg.get("wandb", {}).get("enabled"):
        try:
            import wandb as _w
            _w.init(project=cfg["wandb"].get("project", "ContinualHyper"),
                    name=f"{cfg['wandb'].get('name', 'l2dm')}_a{alpha}_b{beta}_g{gamma}_s{seed}",
                    config={**cfg, "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed})
            wandb = _w
        except Exception as e:
            print(f"[L2DM] wandb off ({e})", flush=True)

    @contextmanager
    def teacher_ctx():
        """Temporarily restore the previous task's spectral shifts (ECD teacher)."""
        cur = shift_state(swapped)
        load_shift_state(swapped, teacher_state)
        try:
            yield
        finally:
            load_shift_state(swapped, cur)

    gstep = 0
    for k, spec in enumerate(specs):
        opt = torch.optim.AdamW(deltas, lr=lr, weight_decay=0.0)
        loader = DataLoader = None
        from torch.utils.data import DataLoader as _DL
        loader = _DL(ConceptDataset(spec, resolution), batch_size=batch_size, shuffle=True,
                     drop_last=True, collate_fn=collate_fn, num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)

        # ---- rainbow memory: refresh the short-term bank with the CURRENT model (Alg. 1)
        if k > 0:
            gen_feats, gen_imgs = {}, {}
            with torch.no_grad():
                for t in range(k):
                    prompts = [specs[t].diag_prompt] * eta
                    ctx, pooled, _ = bundle.encode_text([specs[t].diag_prompt])
                    unc, _, _ = bundle.encode_text([""])
                    ims = []
                    for i0 in range(0, eta, 10):
                        bs = min(10, eta - i0)
                        lat = torch.stack([torch.randn((bundle.latent_channels, 64, 64),
                                           generator=torch.Generator(device=device).manual_seed(9000 + 97 * t + i0 + j),
                                           device=device, dtype=bundle.dtype) for j in range(bs)])
                        ims.append(ddim_sample(bundle, mgr, ctx, unc, pooled, num_inference_steps=25,
                                               guidance_scale=7.5, batch_size=bs,
                                               scheduler=bundle.dpm_scheduler, latents=lat).cpu())
                    gen_imgs[t] = torch.cat(ims)                       # [eta,3,H,W] in [0,1]
                    gen_feats[t] = clip.img_feats_from_tensor(gen_imgs[t].to(device))
            scores = rainbow_scores(gen_feats, banks.long_feats, beta_s=beta_s)
            for t, s in scores.items():
                keep = torch.topk(s, min(short_mem, s.numel())).indices.tolist()
                banks.short_images[t] = (gen_imgs[t][keep] * 2 - 1)     # to [-1,1] like the dataset
                banks.short_prompts[t] = [specs[t].diag_prompt] * len(keep)
            print(f"[L2DM] rainbow memory: {sum(len(v) for v in banks.short_prompts.values())} obrazow "
                  f"z {k} poprzednich konceptow", flush=True)

        for step in range(steps_per_task):
            batch = next(data_iter)
            cond, pooled, _ = bundle.encode_text(batch["captions"], train_tokens=True)
            z0 = bundle.encode_images(batch["pixel_values"].to(device))
            noise = torch.randn_like(z0)
            t_ = torch.randint(0, bundle.num_train_timesteps, (z0.shape[0],), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t_)
            ac = (bundle.added_cond(z_t.shape[0], resolution, resolution, pooled=pooled)
                  if getattr(bundle, "is_sdxl", False) else None)
            eps = unet(z_t, t_, encoder_hidden_states=cond, added_cond_kwargs=ac).sample
            loss = reconstruction_loss(eps.float(), noise.float())
            l_tame = l_pr = l_ecd = 0.0

            mem = banks.sample(k, batch_size, device) if k > 0 else None
            if mem is not None and (alpha > 0 or gamma > 0):
                imgs_m, prompts_m = mem
                rep_loss, (z_m, t_m, cond_m, ac_m) = tame_replay_loss(bundle, unet, imgs_m, prompts_m)
                if alpha > 0:
                    loss = loss + alpha * rep_loss
                    l_tame = float(rep_loss.item())
                if gamma > 0 and teacher_state is not None:
                    eps_s = unet(z_m, t_m, encoder_hidden_states=cond_m, added_cond_kwargs=ac_m).sample
                    with torch.no_grad(), teacher_ctx():
                        eps_t = unet(z_m, t_m, encoder_hidden_states=cond_m, added_cond_kwargs=ac_m).sample
                    d_loss = torch.nn.functional.mse_loss(eps_s.float(), eps_t.detach().float())
                    loss = loss + gamma * d_loss
                    l_ecd = float(d_loss.item())

            if beta > 0:
                pr = load_prior_batch(prior_dir, spec.class_word, batch_size, resolution, device)
                if pr is not None:
                    pr_loss, _ = tame_replay_loss(bundle, unet, pr[0], pr[1])
                    loss = loss + beta * pr_loss
                    l_pr = float(pr_loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deltas, float(train.get("grad_clip", 1.0)))
            opt.step()

            gstep += 1
            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[L2DM] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}"
                      f" | tame {l_tame:.4f} | pr {l_pr:.4f} | ecd {l_ecd:.4f}", flush=True)
            if wandb is not None:
                wandb.log({"loss": float(loss.item()), "tame": l_tame, "pr": l_pr, "ecd": l_ecd,
                           "task": k}, step=gstep)

        # ---- long-term bank: CLIP features + prompts of this task's REAL images (privacy-safe)
        with torch.no_grad():
            ds = ConceptDataset(spec, resolution)
            reals = torch.stack([ds[i]["pixel_values"] for i in range(len(ds))]).to(device)
            banks.long_feats[k] = clip.img_feats_from_tensor((reals + 1) / 2)
            banks.long_prompts[k] = [spec.diag_prompt] * len(ds)

        teacher_state = shift_state(swapped)
        ck = os.path.join(output_dir, "ckpts", f"shifts_after_task{k:02d}.pt")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"shifts": teacher_state}, ck)
        print(f"[L2DM] done task {k}:{spec.concept_id} -> {ck}", flush=True)

    torch.save({"shifts": shift_state(swapped)}, os.path.join(output_dir, "shifts.pt"))
    if wandb is not None:
        wandb.finish()
    print(f"[L2DM] DONE -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
