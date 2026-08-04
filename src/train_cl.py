"""Continual-learning trainer for the LoRA-hypernetwork (sequential over concepts).

Each concept is one task: prompt (CLIP pooled) -> hypernet -> LoRA -> diffusion reconstruction.
With `reg.weight == 0` this is the no-regularization baseline (catastrophic forgetting). With
`reg.weight > 0` it adds the von-Oswald hypernetwork output-regularization (arXiv:1906.00695),
two-stage (lookahead):

  Stage 1: DeltaTheta = -lookahead_lr * grad(L_recon)            (candidate step, detached)
  Stage 2: L = L_recon(Theta) + beta * mean_{t<k} || H_{Theta*}(c*_t) - H_{Theta+DeltaTheta}(c*_t) ||^2

i.e. the hypernet may move to fit the new concept, but its LoRA output for old concepts' prompts
(snapshotted at the start of the task) must stay put -> old concepts are not forgotten.

Forgetting is tracked by sampling each concept right after it is learned and re-sampling the first
concept after every task; per-task checkpoints feed the CIDM forgetting-matrix eval.

Run:  python -u -m src.train_cl --config configs/cl_unhype.yaml                 # baseline
      python -u -m src.train_cl --config configs/cl_unhype_reg.yaml             # with reg
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .common import load_config, save_hyper, set_seed
from .data import ConceptDataset, collate_fn, specs_from_config
from .injection import DEFAULT_TARGETS
from .losses import reconstruction_loss
from .manager import build_hyper
from .sampling import ddim_sample
from .sd_loader import load_sd


@torch.no_grad()
def _gen_one(bundle, manager, prompt, steps, gscale, seed, task_idx=None, mask_phrase=None):
    """Sample one image for `prompt`; returns [3,H,W] in [0,1] on cpu."""
    manager.eval()
    cond_hidden, pooled, _ = bundle.encode_text([prompt])
    uncond_hidden, _, _ = bundle.encode_text([""])
    token_mask = None
    if mask_phrase:
        from .tokens import token_span_mask
        token_mask = token_span_mask(bundle.tokenizer, [prompt], mask_phrase)
    gen = torch.Generator(device=bundle.device).manual_seed(seed)
    img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                      num_inference_steps=steps, guidance_scale=gscale, batch_size=1,
                      generator=gen, scheduler=bundle.dpm_scheduler,
                      task_idx=task_idx, token_mask=token_mask)[0].clamp(0, 1).cpu()
    manager.train()
    return img


@torch.no_grad()
def _sample(bundle, manager, prompt, out_dir, n, steps, gscale, seed, task_idx=None, mask_phrase=None):
    """Generate `n` images for `prompt` into out_dir/sample_{i}.png."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        img = _gen_one(bundle, manager, prompt, steps, gscale, seed + i, task_idx=task_idx,
                       mask_phrase=mask_phrase)
        save_image(img, os.path.join(out_dir, f"sample_{i:02d}.png"))


def _reg_mse(now, targets):
    """von-Oswald output reg: mean over layers of MSE on (x_L, x_R) between the current LoRA
    `now` and the start-of-task snapshot `targets`. F.mse_loss averages over anchors+elements."""
    terms = [F.mse_loss(now[n][0], targets[n][0]) + F.mse_loss(now[n][1], targets[n][1]) for n in targets]
    return torch.stack(terms).mean()


def parse_args():
    p = argparse.ArgumentParser(description="ContinualHyper continual trainer (optional von-Oswald reg)")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--reg_weight", type=float, default=None, help="override reg.weight (von-Oswald beta)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2024)))

    resolution = int(cfg.get("resolution", 512))
    output_dir = args.output_dir or cfg.get("output_dir", "./outputs/cl")
    train = cfg.get("training", {})
    steps_per_task = int(train.get("steps_per_task", 800))
    batch_size = int(train.get("batch_size", 2))
    lr = float(train.get("lr", 1e-4))
    grad_clip = float(train.get("grad_clip", 1.0))
    log_every = int(train.get("log_every", 50))
    diag_freq = int(train.get("diagnostic_freq", 200))   # wandb diagnostic generations every N steps
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))
    reg_cfg = cfg.get("reg", {})
    reg_weight = float(args.reg_weight if args.reg_weight is not None else reg_cfg.get("weight", 0.0))
    lookahead_lr = float(reg_cfg.get("lookahead_lr", lr))   # von-Oswald candidate-step size (default = lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fp32 backbone for the baseline (no grad-scaler headaches; hyper grads stay clean).
    bundle = (load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
              if cfg.get("sd_model_id") else load_sd(device=device, dtype=cfg.get("weight_dtype", "fp32")))

    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    manager.train()

    specs = specs_from_config(cfg["concepts"])
    n_tasks = len(specs)

    # Option C: learned identifier tokens ("<Vk>") -- TI-style rows in the CLIP input embedding.
    tok_cfg = cfg.get("learned_tokens", {}) or {}
    use_tokens = bool(tok_cfg.get("enabled", False))
    token_ids, emb_weight, tok_lr = {}, None, 0.0
    if use_tokens:
        from .tokens import add_learned_tokens
        token_ids = add_learned_tokens(bundle, [(sp.identifier, sp.class_word) for sp in specs],
                                       init_from_class=bool(tok_cfg.get("init_from_class", True)))
        emb_weight = bundle.text_encoder.get_input_embeddings().weight
        emb_weight.requires_grad_(True)      # grads row-masked to the current task's token
        tok_lr = float(tok_cfg.get("lr", 5e-3))
        print(f"[CL] learned tokens ON: {sorted(token_ids)} (tok_lr={tok_lr})", flush=True)

    tm_enabled = bool(cfg.get("token_mask_lora", False))
    if tm_enabled:
        from .tokens import token_span_mask
        print("[CL] token-masked LoRA ON (delta only at concept-token positions)", flush=True)

    def _tok_extra():
        if not use_tokens:
            return None
        with torch.no_grad():
            return {"learned_tokens": {t: emb_weight[i].detach().cpu().clone()
                                       for t, i in token_ids.items()}}
    inf = cfg.get("infer", {})
    gsteps = int(inf.get("steps", 50)); gscale = float(inf.get("guidance_scale", 7.5))
    n_eval = int(inf.get("n_images", 4))
    seed0 = int(cfg.get("seed", 2024))
    unet = bundle.unet

    regmsg = (f"von-Oswald reg beta={reg_weight} (lookahead_lr={lookahead_lr})" if reg_weight > 0 else "NO reg")
    tcmsg = ("task_cond ON (learned V_t + Gram-Schmidt ortho)" if manager.task_cond_enabled else "task_cond OFF")
    print(f"[CL] {n_tasks} tasks (sequential, {regmsg}, {tcmsg}) | {steps_per_task} steps/task"
          f" | bs={batch_size} | lr={lr}", flush=True)
    print("[CL] task order: " + ", ".join(f"{i}:{s.concept_id}('{s.diag_prompt}')" for i, s in enumerate(specs)),
          flush=True)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            run_name = wandb_cfg.get("name")
            if reg_weight > 0:
                run_name = (run_name or "cl") + f"_b{reg_weight}"
            _wandb.init(project=wandb_cfg.get("project", "ContinualHyper"), name=run_name,
                        config={**cfg, "reg_weight": reg_weight, "lookahead_lr": lookahead_lr})
            wandb = _wandb
        except Exception as e:
            print(f"[CL] wandb disabled (init failed: {e})", flush=True)

    named = list(manager.heads.named_parameters())       # (name, param), stable across tasks
    params = [p for _, p in named]
    anchor_conds = []   # hyper conditioning of each learned concept's canonical prompt (reg anchors)
    gstep = 0
    for k, spec in enumerate(specs):
        # Network weights PERSIST across tasks; fresh optimizer per task (clean per-task LR).
        # With task_cond: the CURRENT task's embedding V_k trains too (old V_i stay frozen).
        task_params = manager.task_parameters(k)
        cur_tid = token_ids.get(spec.identifier) if use_tokens else None
        opt_groups = [{"params": manager.hyper_parameters() + task_params}]
        if cur_tid is not None:
            # whole embedding matrix in the group (grads are row-masked); TI-style separate LR
            opt_groups.append({"params": [emb_weight], "lr": tok_lr, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(opt_groups, lr=lr,
                                      weight_decay=float(train.get("weight_decay", 0.0)))
        # task conditioning source (constant per task): the canonical prompt's pooled embedding,
        # or -- ablation `task_cond.key_prompt: identifier` -- just the identifier ("V1"): the key
        # only has to be fixed and GS-separable; generation prompts stay the full canonical text.
        with torch.no_grad():
            tc_cfg = cfg.get("task_cond", {}) or {}
            mode = str(tc_cfg.get("key_prompt", "canonical"))
            # "index": synthetic distinct key per task -- REQUIRED when prompts carry no
            # identifier (canonical prompts of same-class tasks are then identical text).
            key_text = {"identifier": spec.identifier or f"V{k + 1}",
                        "index": f"V{k + 1}"}.get(mode, spec.diag_prompt)
            _, pooled_canon, _ = bundle.encode_text([key_text])
            manager.set_canonical(k, pooled_canon[0])
        loader = DataLoader(ConceptDataset(spec, resolution, augment=bool(train.get("augment", False))),
                            batch_size=batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_fn,
                            num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)

        # von-Oswald: snapshot the hypernet output on OLD concepts at the START of this task (Theta*)
        targets, anchors = None, None
        if reg_weight > 0 and anchor_conds:
            anchors = torch.stack(anchor_conds, 0)           # [k, clip_size], already conditioned
            with torch.no_grad():
                targets = {n: (a.detach(), b.detach())
                           for n, (a, b) in manager.generate_lora(anchors).items()}

        for step in range(steps_per_task):
            batch = next(data_iter)
            images = batch["pixel_values"].to(device)
            captions = batch["captions"]
            bsz = images.shape[0]

            cond_hidden, pooled, _ = bundle.encode_text(captions, train_tokens=cur_tid is not None)
            z0 = bundle.encode_images(images)
            noise = torch.randn_like(z0)
            t = torch.randint(0, bundle.num_train_timesteps, (bsz,), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t)

            tok_mask = (token_span_mask(bundle.tokenizer, captions, spec.replacement).to(device)
                        if tm_enabled else None)
            manager.set_context(pooled, task_idx=k, token_mask=tok_mask)  # timestep-independent LoRA
            manager.compute_and_cache_loras()
            manager.enable_lora()
            eps_pred = unet(z_t, t, encoder_hidden_states=cond_hidden).sample
            loss = reconstruction_loss(eps_pred.float(), noise.float())   # new-concept reconstruction

            optimizer.zero_grad(set_to_none=True)
            reg_val = 0.0
            emb_params = [emb_weight] if cur_tid is not None else []
            if targets is not None:
                # Stage 1: candidate step that minimizes ONLY the new-task loss (detached).
                g_all = torch.autograd.grad(loss, params + task_params + emb_params, retain_graph=False)
                g = g_all[:len(params)]
                g_task = g_all[len(params):len(params) + len(task_params)]
                delta = {nm: (-lookahead_lr * gi).detach() for (nm, _), gi in zip(named, g)}
                # Stage 2: anchor the hypernet output at the lookahead params Theta + DeltaTheta.
                perturbed = {nm: p + delta[nm] for nm, p in named}
                reg = reg_weight * _reg_mse(manager.lora_from_params(anchors, perturbed), targets)
                g_reg = torch.autograd.grad(reg, params)
                for p, gi, gr in zip(params, g, g_reg):    # heads: task grad + reg grad
                    p.grad = gi + gr
                for p, gi in zip(task_params, g_task):     # V_k: task grad only (anchors are frozen)
                    p.grad = gi
                if emb_params:
                    g_emb = g_all[-1]
                    emb_weight.grad = torch.zeros_like(g_emb)
                    emb_weight.grad[cur_tid] = g_emb[cur_tid]   # only the current identifier row trains
                reg_val = float(reg.item())
            else:
                loss.backward()
                if emb_params and emb_weight.grad is not None:
                    keep = emb_weight.grad[cur_tid].clone()     # only the current identifier row trains
                    emb_weight.grad = torch.zeros_like(emb_weight.grad)
                    emb_weight.grad[cur_tid] = keep
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params + task_params + emb_params, grad_clip)
            optimizer.step()

            gstep += 1
            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[CL] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}"
                      + (f" | reg {reg_val:.4f}" if targets is not None else ""), flush=True)
            if wandb is not None:
                wandb.log({"loss": float(loss.item()), "reg": reg_val, "task": k, "task_step": step,
                           "lora_magnitude": float(manager.current_lora_magnitude().item())}, step=gstep)
                if gstep % diag_freq == 0:
                    # diagnostic generations for ALL concepts seen so far -> forgetting visible live
                    logs = {f"diag/{specs[j].concept_id}":
                            wandb.Image(_gen_one(bundle, manager, specs[j].diag_prompt, gsteps, gscale,
                                                 seed0, task_idx=j,
                                                 mask_phrase=specs[j].replacement if tm_enabled else None),
                                        caption=specs[j].diag_prompt)
                            for j in range(k + 1)}
                    wandb.log(logs, step=gstep)

        # just-learned concept (fresh)
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "fresh", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale,
                seed0, task_idx=k, mask_phrase=spec.replacement if tm_enabled else None)
        # forgetting curve: re-sample the FIRST concept after every task
        _sample(bundle, manager, specs[0].diag_prompt,
                os.path.join(output_dir, "forgetting", f"after_task{k:02d}"), n_eval, gsteps, gscale,
                seed0, task_idx=0, mask_phrase=specs[0].replacement if tm_enabled else None)
        # freeze this task's ortho-basis vector z_k BEFORE the checkpoint (ckpt carries the basis),
        # then record the anchor (the task's constant conditioning) for future reg.
        with torch.no_grad():
            manager.freeze_task_basis(k)
            if reg_weight > 0:
                _, pooled_k, _ = bundle.encode_text([spec.diag_prompt])
                anchor_conds.append(manager.condition(pooled_k, k)[0].detach())
        # per-task checkpoint (for the CIDM/CIFC forgetting-matrix eval: each task's model state)
        save_hyper(manager, os.path.join(output_dir, "ckpts", f"hyper_after_task{k:02d}.pt"),
                   extra=_tok_extra())
        print(f"[CL] done task {k}:{spec.concept_id}", flush=True)

    # final sweep: how does EACH concept look after the whole sequence?
    for k, spec in enumerate(specs):
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "final", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale,
                seed0, task_idx=k, mask_phrase=spec.replacement if tm_enabled else None)

    save_hyper(manager, os.path.join(output_dir, "hyper.pt"), extra=_tok_extra())
    if wandb is not None:
        wandb.finish()
    print(f"[CL] DONE -> {output_dir} (fresh/, forgetting/, final/, hyper.pt)", flush=True)


if __name__ == "__main__":
    main()
