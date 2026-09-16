"""Warianty configow baseline'ow, zeby zblizyc reimplementacje do liczb opublikowanych przez
benchmark (CIDM, Tab. 1 przy ich `--alpha 0.8`).

Czym ich baseline'y roznia sie od naszych: LoRA na CALEJ uwadze (`where: Attention`, czyli attn1
i attn2), augmentacje obrazu w loaderze, osobne embeddingi tekstowe per warstwa (ED-LoRA; tego
nie odwzorowujemy). Warianty:
  aug        -- wlacza `training.augment` (losowy crop 80-100% + flip, jak w naszym przepisie)
  allattn    -- LoRA takze na attn1 (self-attention)
  aug_allattn-- oba
  cifc       -- ICH pipeline danych: HumanResizeCropFinalV3 (letterbox + maska straty),
                EnhanceText na podpisach, LoRA na calej uwadze

Run:  python scripts/_mkblvar.py --base configs/baseline_lwf.yaml --variant aug_allattn \
          --out configs/blvar/lwf_aug_allattn.yaml
"""
import argparse
import os

import yaml

ATTN1 = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True, choices=["aug", "allattn", "aug_allattn", "cifc"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.base, encoding="utf-8"))
    if a.variant == "cifc":
        tr = cfg.setdefault("training", {})
        tr["augment"] = "cifc"
        tr["enhance_text"] = True
    elif "aug" in a.variant:
        cfg.setdefault("training", {})["augment"] = True
    if "allattn" in a.variant or a.variant == "cifc":
        tm = list(cfg.get("target_modules", []))
        cfg["target_modules"] = tm + [m for m in ATTN1 if m not in tm]
    name = os.path.splitext(os.path.basename(a.out))[0]
    cfg["output_dir"] = f"./outputs/blvar/{name}"
    if "wandb" in cfg:
        cfg["wandb"]["name"] = name
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"{a.out}: augment={cfg.get('training', {}).get('augment', False)} "
          f"enhance={cfg.get('training', {}).get('enhance_text', False)} "
          f"target_modules={len(cfg['target_modules'])}")


if __name__ == "__main__":
    main()
