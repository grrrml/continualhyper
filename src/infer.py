"""Inference for the LoRA-hypernetwork: a text prompt -> CLIP -> (context, pooled) ->
hypernet LoRA -> sampled image.

Run:  python -u -m src.infer --config configs/cl_unhype.yaml --ckpt outputs/cl_unhype/hyper.pt \
          --prompt "a photo of V3 dog" --out outputs/cl_unhype/infer --num_images 4
"""

from __future__ import annotations

import argparse
import os

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
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = (load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
              if cfg.get("sd_model_id") else load_sd(device=device, dtype=cfg.get("weight_dtype", "fp32")))
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          **cfg.get("hyper", {}))
    load_hyper(manager, args.ckpt, map_location=str(device))
    manager.eval()

    cond_hidden, pooled, _ = bundle.encode_text([args.prompt])
    uncond_hidden, _, _ = bundle.encode_text([args.negative])
    scheduler = bundle.dpm_scheduler if args.sampler == "dpm" else bundle.ddim_scheduler
    os.makedirs(args.out, exist_ok=True)
    print(f"[infer] prompt: {args.prompt}", flush=True)
    for i in range(args.num_images):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                          num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                          batch_size=1, generator=gen, scheduler=scheduler)
        save_image(img, os.path.join(args.out, f"sample_{i:03d}.png"))
        print(f"[infer] saved {args.out}/sample_{i:03d}.png", flush=True)


if __name__ == "__main__":
    main()
