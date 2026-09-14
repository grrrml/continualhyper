"""Rysunek skalowania: co spada z liczba konceptow, a co tylko wyglada na spadek.

Trzy panele, bo z jednej krzywej wyciaga sie zly wniosek:

  (a) srednia po WSZYSTKICH konceptach 0..k. Tak wyglada wynik "na benchmarku" i tak spada
      najmocniej -- ale caly spadek siedzi w skoku 10->20, czyli tam, gdzie do strumienia
      wchodzi CustomConcept101. Miedzy 20 a 40 jest plasko.
  (b) te same punkty rozbite na kohorty: CIFC 0-9 osobno, CC101 osobno. Widac, ze CC101 stoi
      nizej od poczatku, wiec czlon "skladu zbioru" nie jest zapominaniem i zadna metoda CL
      go nie ruszy.
  (c) USTALONE podzbiory ocenione kolejnymi checkpointami -- jedyny panel, ktory faktycznie
      mierzy zapominanie. CIFC 0-9 traci 3x wiecej niz CC101 10-19 na porownywalnym horyzoncie.

Pasek szumu na (a) i (c) pochodzi z trzech ziaren identycznej bazy (sd 0.0133 na DINO przy
T=50) -- bez niego czytelnik wziąłby roznice rzedu 0.01 za wynik.

Run:  python scripts/_fig_scaling.py --data <katalog z tK.json> --out assets/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KS = [9, 19, 29, 39, 49]
SD_DINO_T50 = 0.0133          # z trzech ziaren bazy, patrz REPORT.md 5b.2


def cells(path, K, lo=None, hi=None):
    M = json.load(open(path))["matrix"]
    out = []
    for key, v in M.items():
        k, j = (int(x) for x in key.split(","))
        if k != K:
            continue
        if lo is not None and not (lo <= j <= hi):
            continue
        out.append(v)
    if not out:
        return None
    return {m: sum(x[m] for x in out) / len(out) for m in ("clip_t", "clip_i", "dino_i")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="assets/figures")
    a = ap.parse_args()

    f = {K: os.path.join(a.data, f"t{K}.json") for K in KS}
    T = [K + 1 for K in KS]

    allc = [cells(f[K], K) for K in KS]
    cifc = [cells(f[K], K, 0, 9) for K in KS]
    cc = [cells(f[K], K, 10, 49) for K in KS]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    d = [x["dino_i"] for x in allc]
    ax[0].plot(T, d, "o-", color="#1f77b4", lw=2)
    ax[0].fill_between(T, [y - SD_DINO_T50 for y in d], [y + SD_DINO_T50 for y in d],
                       color="#1f77b4", alpha=0.15, lw=0)
    ax[0].set_title("(a) srednia po konceptach 0..k")
    ax[0].annotate("wejscie CC101", xy=(20, d[1]), xytext=(26, d[1] - 0.045),
                   arrowprops=dict(arrowstyle="->", color="0.4"), color="0.3", fontsize=9)

    ax[1].plot(T, [x["dino_i"] for x in cifc], "o-", label="CIFC 0-9", color="#2ca02c", lw=2)
    ax[1].plot(T[1:], [x["dino_i"] for x in cc[1:]], "s-", label="CC101 10..k",
               color="#d62728", lw=2)
    ax[1].set_title("(b) rozbicie na kohorty")
    ax[1].legend(frameon=False, fontsize=9)

    for lo, hi, first, lab, col in ((0, 9, 0, "CIFC 0-9", "#2ca02c"),
                                    (10, 19, 1, "CC101 10-19", "#d62728"),
                                    (20, 29, 2, "CC101 20-29", "#9467bd")):
        xs, ys = [], []
        for i, K in enumerate(KS[first:], start=first):
            m = cells(f[K], K, lo, hi)
            if m:
                xs.append(K + 1); ys.append(m["dino_i"])
        ax[2].plot(xs, ys, "o-", label=lab, color=col, lw=2)
        ax[2].fill_between(xs, [y - SD_DINO_T50 for y in ys], [y + SD_DINO_T50 for y in ys],
                           color=col, alpha=0.12, lw=0)
    ax[2].set_title("(c) USTALONE podzbiory = zapominanie")
    ax[2].legend(frameon=False, fontsize=9)

    for x in ax:
        x.set_xlabel("liczba nauczonych konceptow T")
        x.set_xticks(T)
        x.grid(alpha=0.25, lw=0.6)
        x.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("DINO (s = 0.45)")
    fig.suptitle("Skalowanie do 50 konceptow: co jest zapominaniem, a co skladem zbioru "
                 f"(pasek = sd miedzy ziarnami, {SD_DINO_T50})", fontsize=10)
    fig.tight_layout()

    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(a.out, f"scaling_T50.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"[fig] {p}")

    print("\nliczby na rysunku:")
    print(f"{'T':>4} {'wszystkie':>10} {'CIFC 0-9':>10} {'CC101':>10}")
    for i, t in enumerate(T):
        c = cc[i]["dino_i"] if cc[i] else float("nan")
        print(f"{t:>4} {allc[i]['dino_i']:10.4f} {cifc[i]['dino_i']:10.4f} {c:10.4f}")


if __name__ == "__main__":
    main()
