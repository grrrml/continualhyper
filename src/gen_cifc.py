"""Generate CIDM/CIFC evaluation images for the continual-learning forgetting matrix.

For each per-task checkpoint k, generate the CIFC recontextualization eval set for every concept
j<=k (lower-triangular CIL matrix). Per concept: read its category's prompt file
(datasets/evaluation_prompts/test_<category>.txt), substitute `<TOK>`:
  * generation prompt   -> "V{j+1} <class>"   (conditions our hypernet, same as training)
  * CLIP-T candidate    -> "<class>"          (natural text, written to prompts.json)
Output layout matches CIFC evaluate.py: <out>/after_task{k}/task{j}_{cid}/{samples/*.jpg, prompts.json}.

Loads SD once; swaps the hyper checkpoint per task (load_state_dict), so it is efficient.

Run:  python -u -m src.gen_cifc --config configs/cl_unhype.yaml \
          --ckpt_dir outputs/cl_unhype/ckpts --out_root outputs/cl_unhype/cifc_eval --num_samples 4
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torchvision.utils import save_image

from .common import load_config, load_hyper
from .injection import DEFAULT_TARGETS
from .manager import build_hyper
from .sampling import ddim_sample
from .sd_loader import load_sd

EVAL_DIR = "data/CIFC/datasets/evaluation_prompts"
CAT_FILE = {"pet": "test_pet.txt", "plushy": "test_plushy.txt", "style": "test_style.txt"}
NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")


def _read_prompts(category):
    with open(os.path.join(EVAL_DIR, CAT_FILE[category])) as f:
        return [ln.strip() for ln in f if ln.strip()]


@torch.no_grad()
def gen_concept(bundle, manager, ident, cls, category, out_dir, n, steps, gscale, seed):
    prompts = _read_prompts(category)
    sdir = os.path.join(out_dir, "samples")
    os.makedirs(sdir, exist_ok=True)
    uncond_hidden, _, _ = bundle.encode_text([NEG])
    info, count = [], 0
    for p in prompts:
        gen_prompt = p.replace("<TOK>", f"{ident} {cls}")
        clipt_text = p.replace("<TOK>", cls)
        cond_hidden, pooled, _ = bundle.encode_text([gen_prompt])
        for _ in range(n):
            g = torch.Generator(device=bundle.device).manual_seed(seed + count)
            img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                              num_inference_steps=steps, guidance_scale=gscale, batch_size=1,
                              generator=g, scheduler=bundle.dpm_scheduler)
            save_image(img, os.path.join(sdir, f"{count}.jpg"))
            info.append({str(count): clipt_text})
            count += 1
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump(info, f)
    return count


def parse_args():
    p = argparse.ArgumentParser(description="CIDM/CIFC eval-image generation (forgetting matrix)")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_dir", required=True, help="dir with hyper_after_task{k:02d}.pt")
    p.add_argument("--out_root", default=None)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--final_only", action="store_true",
                   help="only the final checkpoint over all concepts (no full matrix)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_root = args.out_root or os.path.join(os.path.dirname(args.ckpt_dir.rstrip("/")), "cifc_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = (load_sd(model_id=cfg["sd_model_id"], device=device, dtype=cfg.get("weight_dtype", "fp32"))
              if cfg.get("sd_model_id") else load_sd(device=device, dtype=cfg.get("weight_dtype", "fp32")))
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          **cfg.get("hyper", {}))
    manager.eval()

    concepts = cfg["concepts"]
    n_tasks = len(concepts)
    task_ks = [n_tasks - 1] if args.final_only else list(range(n_tasks))

    for k in task_ks:
        ckpt = os.path.join(args.ckpt_dir, f"hyper_after_task{k:02d}.pt")
        if not os.path.exists(ckpt):
            print(f"[gen] MISSING {ckpt} — skipping task {k}", flush=True)
            continue
        load_hyper(manager, ckpt, map_location=str(device))
        for j in range(k + 1):                              # concept j was seen by task k
            c = concepts[j]
            ident, cls, cat = f"V{j+1}", c["class_word"], c["category"]
            out_dir = os.path.join(out_root, f"after_task{k:02d}", f"task{j:02d}_{c['concept_id']}")
            nimg = gen_concept(bundle, manager, ident, cls, cat, out_dir,
                               args.num_samples, args.steps, args.guidance_scale, args.seed)
            print(f"[gen] after_task{k:02d} / {c['concept_id']} ({cat}, '{ident} {cls}'): {nimg} imgs",
                  flush=True)
    print(f"[gen] DONE -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
