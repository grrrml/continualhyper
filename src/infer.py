"""Inference for the LoRA-hypernetwork: a text prompt -> CLIP -> (context, pooled) ->
hypernet LoRA -> sampled image.

Run:  python -u -m src.infer --config configs/cl_unhype.yaml --ckpt outputs/cl_unhype/hyper.pt \
          --prompt "a photo of V3 dog" --out outputs/cl_unhype/infer --num_images 4
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from torchvision.utils import save_image

from .common import load_config, load_hyper
from .injection import DEFAULT_TARGETS
from .manager import build_hyper
from .sampling import ddim_sample
from .sd_loader import load_sd


def parse_args():
    p = argparse.ArgumentParser(description="UnHype-hypernet inference")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="path to hyper.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--out", default="./outputs/infer")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--sampler", choices=["dpm", "ddim"], default="dpm")
    p.add_argument("--task_idx", type=int, default=None,
                   help="task-conditioning index; default: inferred from 'V<k>' in the prompt")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = (load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
              if cfg.get("sd_model_id") else load_sd(device=device, dtype=cfg.get("weight_dtype", "fp32")))
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg.get("concepts", [])), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    blob = load_hyper(manager, args.ckpt, map_location=str(device))
    tok_cfg = cfg.get("learned_tokens", {}) or {}
    if tok_cfg.get("enabled") and blob.get("learned_tokens"):
        from .tokens import add_learned_tokens, apply_learned_tokens
        add_learned_tokens(bundle, [(c.get("identifier", f"V{i + 1}"), c["class_word"])
                                    for i, c in enumerate(cfg.get("concepts", []))],
                           init_from_class=False)
        apply_learned_tokens(bundle, blob["learned_tokens"])
    manager.eval()

    task_idx = args.task_idx
    prompt = args.prompt
    m = re.search(r"\bV(\d+)\b", prompt)
    if task_idx is None and manager.task_cond_enabled and m:
        task_idx = int(m.group(1)) - 1
        print(f"[infer] task_idx inferred from prompt: {task_idx}", flush=True)
    if cfg.get("strip_identifier") and m:
        # identifier is ROUTING syntax only: the hyper gets the task key, the diffusion
        # model gets the natural prompt without it
        prompt = re.sub(r"\s{2,}", " ", re.sub(r"\s*\bV\d+\b\s*", " ", prompt)).strip()
        print(f"[infer] routing: task {task_idx} | prompt -> {prompt!r}", flush=True)

    token_mask = None
    if cfg.get("token_mask_lora") and task_idx is not None and cfg.get("concepts"):
        from .tokens import token_span_mask
        c = cfg["concepts"][task_idx]
        phrase = " ".join(x for x in (c.get("identifier", f"V{task_idx + 1}"), c["class_word"]) if x)
        token_mask = token_span_mask(bundle.tokenizer, [prompt], phrase)

    cond_hidden, pooled, _ = bundle.encode_text([prompt])
    uncond_hidden, _, _ = bundle.encode_text([args.negative])
    scheduler = bundle.dpm_scheduler if args.sampler == "dpm" else bundle.ddim_scheduler
    os.makedirs(args.out, exist_ok=True)
    print(f"[infer] prompt: {prompt}", flush=True)
    for i in range(args.num_images):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                          num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                          batch_size=1, generator=gen, scheduler=scheduler, task_idx=task_idx,
                          token_mask=token_mask)
        save_image(img, os.path.join(args.out, f"sample_{i:03d}.png"))
        print(f"[infer] saved {args.out}/sample_{i:03d}.png", flush=True)


if __name__ == "__main__":
    main()
