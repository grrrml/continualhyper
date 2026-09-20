"""DINO dla obrazow z `_fig_rerender.py`: koncept x checkpoint x skala adaptera.

Po co. Twierdzenie "uzyteczny zakres skali zwęża sie wraz z dlugoscia strumienia" postawilem
patrzac na JEDEN koncept, jeden prompt i dwa ziarna -- czyli dokladnie tak, jak cztery razy
tej nocy postawilem wnioski, ktore pomiar potem obalil. Ten skrypt zamienia je w liczbe.

Miara jest ta sama, co w calym benchmarku (`cifc_metrics._Dino`, `vit_small_patch16_224.dino`
z ich preprocessingiem), zeby wynik dalo sie zestawic z kazda inna liczba DINO w pracy.

Czyta uklad, ktory pisze `_fig_rerender.py`:  <root>/ck<NN>/task<MM>/s<XXX>/<i>.jpg

Run:  python scripts/_fig_score.py --config configs/phaseT/T50_v2.yaml \\
          --root outputs/figrender/scales4 --out outputs/figrender/scales4.json
"""
import argparse
import glob
import json
import os
import re
import sys

import torch
from PIL import Image

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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    concepts = cfg["concepts"]
    dino = _Dino("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "ck*", "task*", "s*"))):
        m = re.search(r"ck(\d+)[\\/]+task(\d+)[\\/]+s(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        ck, task, stag = int(m.group(1)), int(m.group(2)), m.group(3)
        scale = float(stag[0] + "." + stag[1:])
        imgs = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(EXTS))
        if not imgs:
            continue
        feats = dino.img_feats(imgs)
        v = _cross_cos(feats, refs_for(concepts[task]["images_dir"], dino, cache))
        rows.append({"ck": ck, "task": task, "scale": scale,
                     "concept": concepts[task]["concept_id"], "n": len(imgs), "dino": v})

    tasks = sorted({r["task"] for r in rows})
    scales = sorted({r["scale"] for r in rows})
    cks = sorted({r["ck"] for r in rows})
    idx = {(r["task"], r["ck"], r["scale"]): r["dino"] for r in rows}

    for t in tasks:
        cid = next(r["concept"] for r in rows if r["task"] == t)
        print(f"\n=== zadanie {t} ({cid})", flush=True)
        print("  skala |" + "".join(f"  po {c+1:2d} zad." for c in cks))
        for sc in scales:
            cells = "".join(f"  {idx.get((t, c, sc), float('nan')):9.4f}" for c in cks)
            print(f"   {sc:.2f} |{cells}", flush=True)
        # to jest pytanie, dla ktorego skrypt powstal: czy kara za przesterowanie rosnie
        base = min(scales)
        print("  strata wzgledem skali " + f"{base:.2f}:")
        for sc in scales:
            if sc == base:
                continue
            cells = "".join(
                f"  {idx.get((t, c, sc), float('nan')) - idx.get((t, c, base), float('nan')):+9.4f}"
                for c in cks)
            print(f"   {sc:.2f} |{cells}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print("\n[fig-score] zapisane do", a.out)


if __name__ == "__main__":
    main()
