"""Sequential continual-learning baselines (Finetuning / EWC / LwF / C-LoRA).

One shared LoRA (or one pair per task for C-LoRA) trained over the CIFC task sequence, with a
learned identifier token per concept -- the CIDM recipe, so the only difference between the runs
in the comparison table is the CL mechanism. Per-task checkpoints are written so the same
forgetting-matrix evaluation as for our method can be run.

Run:  python -u -m src.train_baselines --config configs/baseline_ewc.yaml --lam 1000
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
from torch.utils.data import DataLoader

from .baselines import (StaticLoRABank, clone_bank, clora_penalty, diagonal_fisher, ewc_penalty,
                        lwf_distill, snapshot)
from .common import load_config, set_seed
from .data import ConceptDataset, collate_fn, specs_from_config
from .injection import DEFAULT_TARGETS, inject_lora
from .losses import reconstruction_loss
from .sd_loader import load_sd
from .tokens import add_learned_tokens


def parse_args():
    p = argparse.ArgumentParser(description="CL baselines: finetune / ewc / lwf / clora")
    p.add_argument("--config", required=True)
    p.add_argument("--method", default=None,
                   choices=["finetune", "ewc", "lwf", "clora", "lora_m", "lora_c"])
    p.add_argument("--lam", type=float, default=None, help="regularization weight")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    bl = cfg.get("baseline", {})
    method = args.method or bl.get("method", "finetune")
    lam = float(args.lam if args.lam is not None else bl.get("lam", 0.0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 2024))
    set_seed(seed)

    train = cfg.get("training", {})
    steps_per_task = int(train.get("steps_per_task", 800))
    batch_size = int(train.get("batch_size", 2))
    lr = float(train.get("lr", 1e-4))
    tok_lr = float(cfg.get("learned_tokens", {}).get("lr", 1e-3))
    grad_clip = float(train.get("grad_clip", 1.0))
    log_every = int(train.get("log_every", 100))
    resolution = int(cfg.get("resolution", 512))
    output_dir = args.output_dir or cfg.get("output_dir", f"./outputs/baseline_{method}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _mid = cfg.get("sd_model_id", "")
    _wd = cfg.get("weight_dtype", "fp32")
    if "xl" in str(_mid).lower():
        from .sd_loader import load_sdxl
        bundle = load_sdxl(model_id=_mid, device=device, dtype=_wd)
    else:
        bundle = load_sd(model_id=_mid, device=device, dtype=_wd)
    specs = specs_from_config(cfg["concepts"])

    token_ids = add_learned_tokens(bundle, [(s.identifier, s.class_word) for s in specs],
                                   init_from_class=True)
    emb = bundle.text_encoder.get_input_embeddings().weight
    emb.requires_grad_(True)

    wrappers = inject_lora(bundle.unet, tuple(cfg.get("target_modules", DEFAULT_TARGETS)))
    # per-task adapters: C-LoRA (with its penalty) and LoRA-M / LoRA-C (independent, no penalty)
    per_task = method in ("clora", "lora_m", "lora_c", "lora_solo")
    bank = StaticLoRABank(wrappers, rank=int(cfg.get("hyper", {}).get("rank", 4)),
                          per_task=per_task, n_tasks=len(specs)).to(device)
    for _, w in wrappers:
        w.set_parent(bundle.unet)
    bundle.unet.hyper = bank

    print(f"[BL] method={method} lam={lam} seed={seed} | {len(wrappers)} LoRA layers | "
          f"{steps_per_task} steps/task | lr={lr} tok_lr={tok_lr}", flush=True)

    wandb = None
    if cfg.get("wandb", {}).get("enabled"):
        try:
            import wandb as _w
            _w.init(project=cfg["wandb"].get("project", "ContinualHyper"),
                    name=f"{cfg['wandb'].get('name', method)}_lam{lam}_s{seed}",
                    config={**cfg, "method": method, "lam": lam, "seed": seed})
            wandb = _w
        except Exception as e:
            print(f"[BL] wandb off ({e})", flush=True)

    ewc_anchors, teacher = [], None
    unet, gstep = bundle.unet, 0

    for k, spec in enumerate(specs):
        lora_params = bank.start_task(k)
        if method in ("lora_m", "lora_c"):
            bank.solo_task = k        # train each adapter on the frozen base, independently
        cur_tid = token_ids[spec.identifier]
        opt = torch.optim.AdamW([{"params": lora_params, "lr": lr},
                                 {"params": [emb], "lr": tok_lr}], weight_decay=0.0)
        loader = DataLoader(ConceptDataset(spec, resolution), batch_size=batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_fn,
                            num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)

        def task_loss():
            batch = next(data_iter)
            images = batch["pixel_values"].to(device)
            cond, pooled, _ = bundle.encode_text(batch["captions"], train_tokens=True)
            z0 = bundle.encode_images(images)
            noise = torch.randn_like(z0)
            t = torch.randint(0, bundle.num_train_timesteps, (z0.shape[0],), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t)
            ac = (bundle.added_cond(z_t.shape[0], resolution, resolution, pooled=pooled)
                  if getattr(bundle, "is_sdxl", False) else None)
            eps = unet(z_t, t, encoder_hidden_states=cond, added_cond_kwargs=ac).sample
            return reconstruction_loss(eps.float(), noise.float()), (z_t, t, noise, ac)

        for step in range(steps_per_task):
            loss, (z_t, t, _, ac) = task_loss()
            pen_val = 0.0
            if method == "ewc" and ewc_anchors:
                pen = lam * ewc_penalty(lora_params, ewc_anchors)
                loss = loss + pen; pen_val = float(pen.item())
            elif method == "lwf" and teacher is not None and k > 0:
                j = int(torch.randint(0, k, (1,)).item())          # random old concept
                with torch.no_grad():
                    cond_old, pooled_old, _ = bundle.encode_text([specs[j].diag_prompt])
                    cond_old = cond_old.expand(z_t.shape[0], -1, -1)
                    ac_old = (bundle.added_cond(z_t.shape[0], resolution, resolution,
                                                pooled=pooled_old.expand(z_t.shape[0], -1))
                              if getattr(bundle, "is_sdxl", False) else None)
                    bundle.unet.hyper = teacher
                    eps_prev = unet(z_t, t, encoder_hidden_states=cond_old, added_cond_kwargs=ac_old).sample
                    bundle.unet.hyper = bank
                eps_now = unet(z_t, t, encoder_hidden_states=cond_old, added_cond_kwargs=ac_old).sample
                pen = lam * lwf_distill(eps_now, eps_prev)
                loss = loss + pen; pen_val = float(pen.item())
            elif method == "clora" and k > 0:
                pen = lam * clora_penalty(bank, k)
                loss = loss + pen; pen_val = float(pen.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if emb.grad is not None:                    # only the current identifier row trains
                keep = emb.grad[cur_tid].clone()
                emb.grad.zero_(); emb.grad[cur_tid] = keep
            if per_task:                                # freeze previous tasks' pairs
                for pdict in (bank.A, bank.B):
                    for key in pdict:
                        if pdict[key].grad is not None:
                            pdict[key].grad[:k] = 0
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lora_params + [emb], grad_clip)
            opt.step()

            gstep += 1
            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[BL] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}"
                      + (f" | pen {pen_val:.4f}" if pen_val else ""), flush=True)
            if wandb is not None:
                wandb.log({"loss": float(loss.item()), "penalty": pen_val, "task": k}, step=gstep)

        if method == "ewc":
            fisher = diagonal_fisher(lambda: task_loss()[0], lora_params,
                                     int(bl.get("fisher_batches", 200)))
            ewc_anchors.append((fisher, snapshot(lora_params)))
            print(f"[BL] Fisher for task {k}: mean {sum(f.mean() for f in fisher) / len(fisher):.3e}",
                  flush=True)
        if method == "lwf":
            teacher = clone_bank(bank)

        bank.solo_task = None                       # checkpoints store the full bank
        ck = os.path.join(output_dir, "ckpts", f"bank_after_task{k:02d}.pt")
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"bank": bank.state_dict(), "active": bank._active,
                    "learned_tokens": {t_: emb[i].detach().cpu().clone()
                                       for t_, i in token_ids.items()}}, ck)
        print(f"[BL] done task {k}:{spec.concept_id} -> {ck}", flush=True)

    torch.save({"bank": bank.state_dict(), "active": bank._active,
                "learned_tokens": {t_: emb[i].detach().cpu().clone() for t_, i in token_ids.items()}},
               os.path.join(output_dir, "bank.pt"))
    if wandb is not None:
        wandb.finish()
    print(f"[BL] DONE -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
