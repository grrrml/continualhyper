"""Dryf parametrow galezi groundingu miedzy checkpointami zadan.

Po co: forgetting checkpointu z groundingiem wynosi 0.0151 DINO przy s=0.8 wobec 0.0014
z galezia wylaczona na inferencji, wiec galaz uczestniczy w zapominaniu. Nie wiadomo
jednak, czy dryfuje SAMA, czy tylko wzmacnia dryf czynnikow LoRA -- a czlon kotwicy
`regG` mierzony w trakcie treningu Z WLACZONA kotwica tego nie rozstrzyga, bo kotwica
trzymajaca tokeny w miejscu i brak dryfu wygladaja identycznie.

Ten skrypt czyta same state_dicty, bez SD i bez generowania obrazow. Klucze zadan sa
deterministyczne i zamrozone, wiec przy stalym wejsciu dryf parametrow jest gornym
ograniczeniem dryfu wyjscia: jesli wagi galezi stoja, wyjscie tez stoi.

Uzycie:
  python -u scripts/_ground_drift.py --ckpts outputs/phaseP/P_ground_gsa_erode/ckpts
"""
import argparse
import glob
import os
import re

import torch

GROUPS = {
    "ground_head": r"^ground_head\.",
    "ground_film": r"^ground_film\.",
    "ground_gates": r"^ground_gates\.",
    "ground_gsa_mods": r"^ground_gsa_mods\.",
    "heads (LoRA)": r"^heads\.",
}


def rel_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """||b-a|| / ||a||, z zabezpieczeniem na zerowo zainicjalizowane tensory."""
    na = a.float().norm().item()
    d = (b.float() - a.float()).norm().item()
    return d / na if na > 1e-12 else (float("inf") if d > 1e-12 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="katalog z hyper_after_taskNN.pt")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.ckpts, "hyper_after_task*.pt")))
    assert len(paths) >= 2, f"potrzebne co najmniej dwa checkpointy, jest {len(paths)}"
    print(f"checkpointow: {len(paths)}  ({os.path.basename(paths[0])} .. {os.path.basename(paths[-1])})")

    sds = [torch.load(p, map_location="cpu")["manager"] for p in paths]
    first, last = sds[0], sds[-1]

    print(f"\n{'grupa':<16} {'tensorow':>8} {'||param||':>12} {'krok po kroku':>14} {'pierwszy->ostatni':>18}")
    for name, pat in GROUPS.items():
        keys = [k for k in first if re.match(pat, k)]
        if not keys:
            continue
        norm = sum(first[k].float().norm().item() ** 2 for k in keys) ** 0.5
        step = max((max(rel_drift(sds[i][k], sds[i + 1][k]) for k in keys)
                    for i in range(len(sds) - 1)), default=0.0)
        total = max(rel_drift(first[k], last[k]) for k in keys)
        print(f"{name:<16} {len(keys):>8} {norm:>12.4f} {step:>14.6f} {total:>18.6f}")

    # bramki osobno: to skalary, wiec pokazujemy wartosci bezwzglednie
    gk = [k for k in first if k.startswith("ground_gates.")]
    if gk:
        g0 = torch.stack([first[k].float().reshape(-1) for k in gk]).reshape(-1)
        g1 = torch.stack([last[k].float().reshape(-1) for k in gk]).reshape(-1)
        print(f"\nbramki: {len(gk)} skalarow | pierwszy ckpt mean {g0.mean():+.4f} "
              f"absmax {g0.abs().max():.4f} | ostatni mean {g1.mean():+.4f} absmax {g1.abs().max():.4f}"
              f" | max |delta| {(g1 - g0).abs().max():.4f}")

    print("\nInterpretacja: 'krok po kroku' to najwiekszy relatywny dryf pojedynczego tensora\n"
          "miedzy dwoma kolejnymi zadaniami, 'pierwszy->ostatni' przez caly strumien. Jesli\n"
          "grupy galezi maja dryf o rzedy mniejszy niz 'heads (LoRA)', galaz nie jest zrodlem\n"
          "zapominania i pozostaje hipoteza wzmacniania dryfu LoRA przez odczyt uwagi.")


if __name__ == "__main__":
    main()
