"""Ile konceptu jest w jego wlasnej ramce -- kompozycja mierzona, a nie ogladana.

Po co. Przez dwa dni wybieralismy konfiguracje kompozycji patrzac na obrazy, i trzy razy
z rzedu wniosek wyciagniety z jednej sceny okazal sie falszywy po obejrzeniu wszystkich
jedenastu. Teraz doszlismy do miejsca, gdzie roznice miedzy wariantami sa mniejsze niz
rozrzut miedzy ziarnami, wiec oko przestaje wystarczac: kazda scena ma dwa ziarna, a ocena
z jednego ziarna na scene to ten sam blad co ocena z jednej sceny.

Miara. Dla kazdego regionu wycinamy z obrazu jego wlasne pudelko i liczymy DINO do zdjec
referencyjnych TEGO konceptu. Trzy tryby awarii, ktore widzielismy, obnizaja te liczbe
kazdy z osobna: brak podmiotu (w ramce jest tlo), duplikat sasiada (w ramce jest nie ten
koncept), deformacja w "plat futra" (tekstura zamiast obiektu).

Kontrola wbudowana: dla kazdego wycinka liczymy TEZ DINO do konceptow pozostalych regionow
tej sceny. Jesli wlasny nie jest wyzszy, to nie jest brak jakosci, tylko podmiot wyladowal
w cudzej ramce -- a to inna usterka i inna poprawka. Bez tej kolumny obie wygladaja tak samo.

Run:  python scripts/_compose_score.py --config configs/phaseX/X_sdxl_b3000.yaml \
          --dirs outputs/compose_scenes/xl_b2k4,outputs/compose_scenes/xl_b1k4
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifc_metrics import _Dino, _cross_cos   # noqa: E402
from src.common import load_config               # noqa: E402

# UWAGA na wybor modelu. `src.eval.DinoScorer` bierze dinov2 (vit_small_patch14_dinov2), a caly
# nasz benchmark liczy na `vit_small_patch16_224.dino` z `cifc_metrics` -- z preprocessingiem
# skopiowanym z ich evaluate.py. Uzywamy TEGO SAMEGO, z dwoch powodow: liczby maja byc w tej
# samej skali co kazde inne DINO w pracy, a dinov2 nie ma go w cache'u na Atenie i job pada
# na HF_HUB_OFFLINE (3184841).

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@torch.no_grad()
def embed_pils(dino, imgs):
    """img_feats bierze sciezki, a wycinki mamy w pamieci -- ta sama transformacja, inne wejscie."""
    x = torch.stack([dino.tf(im) for im in imgs]).to(dino.device)
    return F.normalize(dino.m(x), dim=-1)


def refs_for(images_dir, dino, cache):
    """Cechy DINO zdjec referencyjnych konceptu, policzone raz."""
    if images_dir not in cache:
        paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                       if p.lower().endswith(EXTS))
        if not paths:
            raise SystemExit(f"BLAD: brak zdjec referencyjnych w {images_dir}")
        cache[images_dir] = dino.img_feats(paths)
    return cache[images_dir]


def crop(img, box):
    """Pudelko z manifestu jest znormalizowane [x0, y0, x1, y1]."""
    w, h = img.size
    x0, y0, x1, y1 = box
    px = (max(0, int(x0 * w)), max(0, int(y0 * h)),
          min(w, int(x1 * w)), min(h, int(y1 * h)))
    if px[2] - px[0] < 8 or px[3] - px[1] < 8:
        raise SystemExit(f"BLAD: zdegenerowane pudelko {box} przy rozmiarze {img.size}")
    return img.crop(px)


def score_one(cut_emb, regions, i, concepts, dino, cache, scene, seed_tag):
    """Jeden wycinek: DINO do wlasnego konceptu i najlepszy wsrod pozostalych w scenie."""
    own = concepts[regions[i]["task_idx"]]
    s_own = _cross_cos(cut_emb, refs_for(own["images_dir"], dino, cache))
    others = [concepts[o["task_idx"]] for j, o in enumerate(regions) if j != i]
    s_oth = max((_cross_cos(cut_emb, refs_for(o["images_dir"], dino, cache))
                 for o in others), default=float("nan"))
    return {"scene": scene, "seed": seed_tag, "region": regions[i]["v"],
            "concept": own["concept_id"], "own": s_own, "best_other": s_oth}


def score_solo(scene_dir, scene, regions, concepts, dino, cache):
    """Kontrola `--solo`: kazdy region wyrenderowany OSOBNO, ta sama ramka, prompt i ziarno.
    Obrazy leza w `<scena>/solo_<V>/<i>.png` i kazdy zawiera dokladnie jeden koncept, wiec
    tniemy tylko jego ramke. Roznica wobec kompozycji jest wtedy efektem samego skladania."""
    rows = []
    for sd in sorted(glob.glob(os.path.join(scene_dir, "solo_*"))):
        v = os.path.basename(sd)[len("solo_"):]
        idx = [j for j, r in enumerate(regions) if r["v"] == v]
        if not idx:
            raise SystemExit(f"BLAD: {sd} nie ma odpowiednika w manifescie sceny {scene}")
        i = idx[0]
        for img_path in sorted(glob.glob(os.path.join(sd, "*.png"))):
            img = Image.open(img_path).convert("RGB")
            e = embed_pils(dino, [crop(img, regions[i]["box"])])
            seed_tag = os.path.splitext(os.path.basename(img_path))[0]
            rows.append(score_one(e, regions, i, concepts, dino, cache, scene, seed_tag))
    return rows


def score_dir(d, concepts, dino, cache):
    rows = []
    for man_path in sorted(glob.glob(os.path.join(d, "*", "manifest.json"))):
        scene_dir = os.path.dirname(man_path)
        scene = os.path.basename(scene_dir)
        man = json.load(open(man_path))
        regions = man["regions"]
        if glob.glob(os.path.join(scene_dir, "solo_*")):
            rows += score_solo(scene_dir, scene, regions, concepts, dino, cache)
            continue
        imgs = sorted(p for p in glob.glob(os.path.join(scene_dir, "*.png"))
                      if os.path.basename(p)[0].isdigit())
        for img_path in imgs:
            seed_tag = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path).convert("RGB")
            cuts = embed_pils(dino, [crop(img, r["box"]) for r in regions])
            for i in range(len(regions)):
                rows.append(score_one(cuts[i:i + 1], regions, i, concepts,
                                      dino, cache, scene, seed_tag))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dirs", required=True, help="katalogi kompozycji po przecinku")
    ap.add_argument("--out", default="outputs/compose_scenes/score.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    concepts = cfg["concepts"]
    dino = _Dino("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    all_rows = {}
    for d in [x.strip() for x in a.dirs.split(",") if x.strip()]:
        if not os.path.isdir(d):
            print(f"[score] brak {d}, pomijam", flush=True)
            continue
        rows = score_dir(d, concepts, dino, cache)
        all_rows[d] = rows
        n = len(rows)
        mean_own = sum(r["own"] for r in rows) / max(n, 1)
        # region "przegrany" = wlasny koncept nie jest w swojej ramce najsilniejszy
        lost = sum(1 for r in rows if r["own"] <= r["best_other"])
        print(f"\n=== {os.path.basename(d)}: {n} regionow, srednie DINO wlasne {mean_own:.4f}, "
              f"przegranych {lost} ({100.0 * lost / max(n, 1):.0f}%)", flush=True)
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene"], []).append(r["own"])
        for sc in sorted(by_scene):
            v = by_scene[sc]
            print(f"    {sc:5s} n={len(v):2d}  {sum(v) / len(v):.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(all_rows, open(a.out, "w"), indent=1)
    print("\n[score] zapisane do", a.out)


if __name__ == "__main__":
    main()
