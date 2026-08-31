"""SUFIT metryki DINO per koncept: jak podobne sa do siebie same zdjecia referencyjne.

Po co: DINO generacji liczymy wobec SREDNIEJ z 4-7 zdjec referencyjnych. Jesli te zdjecia
sa wzajemnie niepodobne (plecak sam na skale vs ten sam plecak na plecach kobiety), srednia
jest rozmyta i zadna generacja nie moze dostac wysokiej oceny - niska liczba mowi wtedy o
DANYCH, nie o modelu. Bez tego sufitu nie da sie odpowiedziec, czy plecak (DINO 0.62 przy
0.72-0.89 dla reszty) jest porazka adaptera, czy granica benchmarku.

Trzy liczby na koncept:
  self_par  -- srednie podobienstwo PAR zdjec referencyjnych (spojnosc konceptu w danych)
  ref_mean  -- srednie podobienstwo zdjecia do SREDNIEJ referencji: to jest wprost sufit,
               bo tak samo liczymy generacje
  min_par   -- najgorsza para, czyli jak daleko odstaje najbardziej odmienne zdjecie

Run: python -u scripts/_ref_selfsim.py --config configs/phaseP/P_ground_gsa_nocap_all.yaml
"""
import argparse
import glob
import itertools
import os
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, ".")
from src.common import load_config
from src.cifc_metrics import _Dino

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_ground_gsa_nocap_all.yaml")
a = ap.parse_args()

cfg = load_config(a.config)
dino = _Dino("cuda")
print(f"[cfg] {a.config}", flush=True)

for c in cfg["concepts"]:
    paths = sorted(glob.glob(os.path.join(c["images_dir"], "*")))
    if len(paths) < 2:
        continue
    f = Fn.normalize(dino.img_feats(paths), dim=-1)          # [N, D]
    pares = [float((f[i] @ f[j]).item()) for i, j in itertools.combinations(range(len(f)), 2)]
    mean_ref = Fn.normalize(f.mean(0, keepdim=True), dim=-1)
    to_mean = [float((f[i] @ mean_ref.t()).item()) for i in range(len(f))]
    print(f"  {c['concept_id']:<18} n={len(paths)} | self_par {sum(pares)/len(pares):.4f} | "
          f"ref_mean {sum(to_mean)/len(to_mean):.4f} | min_par {min(pares):.4f} | "
          f"kategoria {c.get('category')}", flush=True)
print("SELFSIM_DONE", flush=True)
