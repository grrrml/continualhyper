"""Config baseline'u na strumieniu T=50: baza = configs/baseline_<m>.yaml (metoda, budzet, sekcja
`baseline`), koncepty = 50 z configs/phaseT/T50_mixed.yaml. Baseline'y ucza token <Vk> per koncept
(tok_lr w train_baselines), wiec kazdy koncept dostaje identifier <Vk> i prompt "a photo of <Vk> <klasa>",
tak jak dziesiec konceptow w bazie. Sila regularyzacji idzie z CLI (--lam), nie z configu, zeby jeden
config obsluzyl caly maly sweep.

Run:  python scripts/_mkbl50.py --base configs/baseline_ewc.yaml --stream configs/phaseT/T50_mixed.yaml \
          --out configs/bl50/bl50_ewc.yaml
"""
import argparse
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.base, encoding="utf-8"))
    stream = yaml.safe_load(open(a.stream, encoding="utf-8"))
    concepts = []
    for k, c in enumerate(stream["concepts"], start=1):
        c = dict(c)
        c["identifier"] = f"<V{k}>"
        c["prompt"] = f"a photo of <V{k}> {c['class_word']}"
        concepts.append(c)
    cfg["concepts"] = concepts
    name = os.path.splitext(os.path.basename(a.out))[0]
    cfg["output_dir"] = f"./outputs/baseline50/{name}"
    if "wandb" in cfg:
        cfg["wandb"]["name"] = name
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"{a.out}: {len(concepts)} konceptow, metoda {cfg.get('baseline', {}).get('method', cfg.get('l2dm', {}).get('method', '?'))}")


if __name__ == "__main__":
    main()
