"""Bake the hypernetwork's output into a per-task LoRA and report the deployment cost.

Our conditioning is a **constant key per task**, so the LoRA a task receives is deterministic and
fixed. The hypernetwork is therefore training-time machinery: at deployment one can precompute
each task's (x_L, x_R) once and drop the heads entirely. This script does that and prints the
parameter counts that belong in the paper's cost table -- the honest number to compare against
CIDM is the baked size, not the hypernetwork size.

Run: python -m scripts.export_baked_lora --config configs/cl_noid_a2full.yaml \
         --ckpt outputs/cl_noidA2F_b100/hyper.pt --out outputs/cl_noidA2F_b100/baked_lora.pt
"""

from __future__ import annotations

import argparse

import torch

from src.common import load_config, load_hyper
from src.injection import DEFAULT_TARGETS
from src.manager import build_hyper
from src.sd_loader import load_sd


def main() -> None:
    ap = argparse.ArgumentParser(description="export per-task LoRA from a trained hypernetwork")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="bake at this LoRA scale (the operating point)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    load_hyper(manager, args.ckpt, map_location=str(device))
    manager.eval()
    manager.lora_scale = float(args.scale)

    baked, n_baked = {}, 0
    with torch.no_grad():
        for k in range(len(cfg["concepts"])):
            manager.compute_and_cache_loras(manager.canon_pooled[k:k + 1], task_idx=k)
            per_task = {}
            for name in manager.layer_names:
                x_l, x_r = manager.get_cached_lora(name)
                per_task[name] = (x_l[0].detach().cpu().clone(), x_r[0].detach().cpu().clone())
                n_baked += x_l[0].numel() + x_r[0].numel()
            baked[k] = per_task

    hyper_params = sum(p.numel() for p in manager.hyper_parameters())
    print(f"hypernetwork (training-time): {hyper_params / 1e6:8.2f}M  "
          f"({hyper_params * 4 / 1e6:6.1f} MB) -- constant in the number of tasks")
    print(f"baked LoRA, {len(baked)} tasks:      {n_baked / 1e6:8.2f}M  "
          f"({n_baked * 4 / 1e6:6.1f} MB) -- what deployment actually needs")
    print(f"per task:                     {n_baked / len(baked) / 1e6:8.3f}M")

    if args.out:
        torch.save({"baked": baked, "scale": args.scale, "config": args.config}, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
