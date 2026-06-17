"""Continual-learning baseline: train the UnHype LoRA-hypernetwork SEQUENTIALLY over concepts,
with NO regularization on the hypernetwork weights.

Each concept is one task: prompt (CLIP pooled) -> hypernet -> LoRA -> diffusion reconstruction.
Tasks are learned one after another; the hypernet weights persist across tasks but are NOT
protected, so training task k drifts the mapping for tasks < k -> catastrophic forgetting. This
is the intended "it should fall apart" baseline (Step 1).

Next steps (NOT here): (i) von-Oswald-style output regularization on the hypernet
(arxiv 1906.00695) to keep old-task LoRAs stable; (ii) the LoRA scheme from arxiv 2508.08812.

Forgetting is tracked by (a) sampling each task's concept right after it is learned, and
(b) re-sampling the FIRST concept after every task (a forgetting curve), plus a final sweep
over all concepts.

Run:  python -u -m src.train_cl --config configs/cl_unhype.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
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
def _gen_one(bundle, manager, prompt, steps, gscale, seed):
    """Sample one image for `prompt`; returns [3,H,W] in [0,1] on cpu."""
    manager.eval()
    cond_hidden, pooled, _ = bundle.encode_text([prompt])
    uncond_hidden, _, _ = bundle.encode_text([""])
    gen = torch.Generator(device=bundle.device).manual_seed(seed)
    img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                      num_inference_steps=steps, guidance_scale=gscale,
                      batch_size=1, generator=gen, scheduler=bundle.dpm_scheduler)[0].clamp(0, 1).cpu()
    manager.train()
    return img


@torch.no_grad()
def _sample(bundle, manager, prompt, out_dir, n, steps, gscale, seed):
    """Generate `n` images for `prompt` into out_dir/sample_{i}.png."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        img = _gen_one(bundle, manager, prompt, steps, gscale, seed + i)
        save_image(img, os.path.join(out_dir, f"sample_{i:02d}.png"))


def parse_args():
    p = argparse.ArgumentParser(description="ContinualHyper UnHype-style CL baseline (no reg)")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fp32 backbone for the baseline (no grad-scaler headaches; hyper grads stay clean).
    bundle = (load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
              if cfg.get("sd_model_id") else load_sd(device=device, dtype=cfg.get("weight_dtype", "fp32")))

    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          **cfg.get("hyper", {}))
    manager.train()

    specs = specs_from_config(cfg["concepts"])
    n_tasks = len(specs)
    inf = cfg.get("infer", {})
    gsteps = int(inf.get("steps", 50)); gscale = float(inf.get("guidance_scale", 7.5))
    n_eval = int(inf.get("n_images", 4))
    seed0 = int(cfg.get("seed", 2024))
    unet = bundle.unet

    print(f"[CL] {n_tasks} tasks (sequential, NO reg) | {steps_per_task} steps/task | bs={batch_size} | lr={lr}",
          flush=True)
    print("[CL] task order: " + ", ".join(f"{i}:{s.concept_id}('{s.diag_prompt}')" for i, s in enumerate(specs)),
          flush=True)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.init(project=wandb_cfg.get("project", "ContinualHyper"),
                        name=wandb_cfg.get("name"), config=cfg)
            wandb = _wandb
        except Exception as e:
            print(f"[CL] wandb disabled (init failed: {e})", flush=True)

    gstep = 0
    for k, spec in enumerate(specs):
        # Network weights PERSIST across tasks; fresh optimizer per task (clean per-task LR).
        optimizer = torch.optim.AdamW(manager.hyper_parameters(), lr=lr,
                                      weight_decay=float(train.get("weight_decay", 0.0)))
        loader = DataLoader(ConceptDataset(spec, resolution), batch_size=batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_fn,
                            num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)

        for step in range(steps_per_task):
            batch = next(data_iter)
            images = batch["pixel_values"].to(device)
            captions = batch["captions"]
            bsz = images.shape[0]

            cond_hidden, pooled, _ = bundle.encode_text(captions)
            z0 = bundle.encode_images(images)
            noise = torch.randn_like(z0)
            t = torch.randint(0, bundle.num_train_timesteps, (bsz,), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t)

            manager.set_context(pooled)            # timestep-independent LoRA (one per prompt)
            manager.compute_and_cache_loras()
            manager.enable_lora()
            eps_pred = unet(z_t, t, encoder_hidden_states=cond_hidden).sample
            loss = reconstruction_loss(eps_pred.float(), noise.float())   # PURE recon, no reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(manager.hyper_parameters(), grad_clip)
            optimizer.step()

            gstep += 1
            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[CL] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}", flush=True)
            if wandb is not None:
                wandb.log({"loss": float(loss.item()), "task": k, "task_step": step,
                           "lora_magnitude": float(manager.current_lora_magnitude().item())}, step=gstep)
                if gstep % diag_freq == 0:
                    # diagnostic generations for ALL concepts seen so far -> forgetting visible live
                    logs = {f"diag/{specs[j].concept_id}":
                            wandb.Image(_gen_one(bundle, manager, specs[j].diag_prompt, gsteps, gscale, seed0),
                                        caption=specs[j].diag_prompt)
                            for j in range(k + 1)}
                    wandb.log(logs, step=gstep)

        # just-learned concept (fresh)
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "fresh", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale, seed0)
        # forgetting curve: re-sample the FIRST concept after every task
        _sample(bundle, manager, specs[0].diag_prompt,
                os.path.join(output_dir, "forgetting", f"after_task{k:02d}"), n_eval, gsteps, gscale, seed0)
        # per-task checkpoint (for the CIDM/CIFC forgetting-matrix eval: each task's model state)
        save_hyper(manager, os.path.join(output_dir, "ckpts", f"hyper_after_task{k:02d}.pt"))
        print(f"[CL] done task {k}:{spec.concept_id}", flush=True)

    # final sweep: how does EACH concept look after the whole sequence?
    for k, spec in enumerate(specs):
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "final", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale, seed0)

    save_hyper(manager, os.path.join(output_dir, "hyper.pt"))
    if wandb is not None:
        wandb.finish()
    print(f"[CL] DONE -> {output_dir} (fresh/, forgetting/, final/, hyper.pt)", flush=True)


if __name__ == "__main__":
    main()
