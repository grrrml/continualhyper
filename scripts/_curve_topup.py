"""Dogeszczenie krzywej TA/IA/DINO dla JEDNEGO punktu, na obu koncach (T=10 i T=50).

Po co osobny skrypt: `sbatch_sweep.sh` liczy staly zestaw skal (t9: 0.45/0.6/0.75,
t49: +0.9), dobrany tak, zeby punkt zmiescil sie w budzecie. Do scharakteryzowania KANDYDATA
to za malo -- przy plaskim albo krotkim odcinku TA odczyt przy zrownanym TA wypada poza
zmierzonym zakresem i trzeba go ekstrapolowac (zmierzone 2026-09-15: wspolczynniki do +2250).
Dolozenie skal po obu stronach przesuwa odczyt z ekstrapolacji do interpolacji i pokazuje
caly kompromis tozsamosc-tekst, a nie trzy punkty na nim.

Wywolania sa DOKLADNIE takie jak w `sbatch_sweep.sh` (te same `--only_tasks`, `--only_concepts`,
`--num_samples`, `--eval_dtype`), zeby nowe skale byly porownywalne ze starymi co do protokolu.
Skale juz policzone sa pomijane, wiec skrypt jest idempotentny i mozna go puscic ponownie.
Obrazy kasowane od razu po metrykach -- `$SCRATCH` ma limit inodow, a wynikiem jest
`cifc_metrics.json` wazacy 2 KB.

Run (przez scripts/sbatch_py.sh):
  python scripts/_curve_topup.py --root outputs/sweep/p001 --t9 0.3 0.9 1.05 --t49 0.3 1.05
"""
import argparse
import os
import subprocess
import sys

CONCEPTS = ("cifc_dog,cifc_duck_toy,cifc_cat,cifc_backpack,cifc_teddybear,"
            "cifc_painting,cifc_dog2,cifc_drawing,cifc_cat2,cifc_ink_painting")


def scale_dir(s):
    """0.45 -> s045, 0.9 -> s09, 1.05 -> s105 (ta sama konwencja co w sbatch_sweep.sh)."""
    return "s" + str(s).replace("0.", "0", 1).replace(".", "")


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="katalog punktu, np. outputs/sweep/p001")
    ap.add_argument("--t9", nargs="*", type=float, default=[], help="skale do dolozenia przy T=10")
    ap.add_argument("--t49", nargs="*", type=float, default=[], help="skale do dolozenia przy T=50")
    ap.add_argument("--num_samples", type=int, default=10)
    a = ap.parse_args()

    cfg = os.path.join(a.root, "config.yaml")
    ckpts = os.path.join(a.root, "ckpts")
    for p in (cfg, ckpts):
        if not os.path.exists(p):
            raise SystemExit(f"brak {p} -- czy to na pewno katalog punktu sweepu?")

    fail = 0
    for K, scales in ((9, a.t9), (49, a.t49)):
        for s in scales:
            out = os.path.join(a.root, f"curveA_t{K}", scale_dir(s))
            if os.path.exists(os.path.join(out, "cifc_metrics.json")):
                print(f"pomijam t{K} {s} (gotowe)", flush=True)
                continue
            print(f"=== t{K}, skala {s} -> {out}", flush=True)
            if run([sys.executable, "-u", "-m", "src.gen_cifc", "--config", cfg,
                    "--ckpt_dir", ckpts, "--out_root", out,
                    "--num_samples", str(a.num_samples), "--lora_scale", str(s),
                    "--sample_batch", "10", "--eval_dtype", "fp16",
                    "--only_tasks", str(K), "--only_concepts", CONCEPTS]):
                print(f"!!! generacja padla t{K} {s}", flush=True)
                fail = 1
                continue
            if run([sys.executable, "-u", "-m", "src.cifc_metrics",
                    "--config", cfg, "--eval_root", out]):
                print(f"!!! metryki padly t{K} {s}", flush=True)
                fail = 1
                continue
            for root_, _, files in os.walk(out):       # inody: wynikiem jest json, nie obrazy
                for f in files:
                    if f.endswith((".jpg", ".png")):
                        os.remove(os.path.join(root_, f))

    # `_sweep_score.py` nadpisze score.json nowym, gestszym odczytem
    run([sys.executable, "-u", "scripts/_sweep_score.py", "--root", a.root, "--ta", "0.747"])
    print("TOPUP FAILED" if fail else "ALL_DONE", flush=True)
    return fail


sys.exit(main())
