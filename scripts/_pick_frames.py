"""Ranking POJEDYNCZYCH kadrow wzgledem referencji -- do wyboru ujec na figury.

Po co osobny skrypt, skoro jest `_fig_score.py`. Tamten liczy JEDNA liczbe na katalog
(usrednia po wszystkich obrazach w nim), a tu potrzebny jest ranking obraz po obrazie, zeby
wybrac najlepsze ujecie na koncept.

Po co w ogole mierzyc zamiast wybrac okiem: w tej sesji ocena wzrokowa pojedynczych obrazow
zawiodla czterokrotnie i za kazdym razem pomiar to obalil. Kryterium jest przy tym wlasciwe --
teaser ma pokazac, ze model po piecdziesieciu zadaniach nadal oddaje TOZSAMOSC, a DINO mierzy
doklanie to.

Zastrzezenie: dla konceptow STYLU (np. cifc_painting) DINO do zdjec referencyjnych znaczy
mniej, bo styl nie determinuje tresci sceny. Takie koncepty skrypt oznacza w wydruku.

Run:  python scripts/_pick_frames.py --config configs/phaseT/T50_v2.yaml \\
          --root outputs/figrender/teaser_v2 --top 2
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifc_metrics import _Dino, _cross_cos   # noqa: E402
from src.common import load_config               # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def refs_for(images_dir, dino, cache):
    if images_dir not in cache:
        paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                       if p.lower().endswith(EXTS))
        if not paths:
            raise SystemExit(f"BLAD: brak zdjec referencyjnych w {images_dir}")
        cache[images_dir] = dino.img_feats(paths)
    return cache[images_dir]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--top", type=int, default=2, help="ile najlepszych pokazac na koncept")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    concepts = cfg["concepts"]
    dino = _Dino("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "ck*", "task*", "s*"))):
        m = re.search(r"task(\d+)[\\/]+s(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        task = int(m.group(1))
        stag = m.group(2)
        scale = float(stag[0] + "." + stag[1:])
        c = concepts[task]
        refs = refs_for(c["images_dir"], dino, cache)
        for img in sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(EXTS)):
            v = _cross_cos(dino.img_feats([img]), refs)
            rows.append({"task": task, "concept": c["concept_id"], "scale": scale,
                         "category": c.get("category", ""), "file": img, "dino": v})

    for t in sorted({r["task"] for r in rows}):
        sel = sorted((r for r in rows if r["task"] == t), key=lambda r: -r["dino"])
        cid, cat = sel[0]["concept"], sel[0]["category"]
        flag = "  <-- STYL, DINO mniej wiarygodne" if cat == "style" else ""
        print(f"\nzadanie {t:2d}  {cid}{flag}")
        for r in sel[:a.top]:
            print(f"    {r['dino']:.4f}  skala {r['scale']:.2f}  ziarno "
                  f"{os.path.splitext(os.path.basename(r['file']))[0]}")
        if len(sel) > a.top:
            print(f"    (najgorszy {sel[-1]['dino']:.4f} przy skali {sel[-1]['scale']:.2f}, "
                  f"rozstrzal {sel[0]['dino'] - sel[-1]['dino']:.4f}, kandydatow {len(sel)})")
        # ile daje najlepsza POJEDYNCZA skala dla tego konceptu -- do wyboru jednej na caly pasek
        bys = {}
        for r in sel:
            bys.setdefault(r["scale"], []).append(r["dino"])
        best = {k: max(v) for k, v in bys.items()}
        print("    najlepszy kadr wg skali: "
              + "  ".join(f"{k:.2f}:{best[k]:.3f}" for k in sorted(best)))

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("\n[pick] zapisane do", a.out)


if __name__ == "__main__":
    main()
