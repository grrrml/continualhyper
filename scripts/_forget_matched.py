"""Zapominanie odczytane przy ZROWNANYM TA, a nie przy wybranej skali adaptera.

Po co. Zapominanie rosnie monotonicznie ze skala adaptera dla KAZDEJ metody -- u samego
fine-tuningu idzie 0,17 -> 0,25 -> 0,37 przy skalach 0,6 / 0,8 / 1,0. Porownanie przy tej
samej SKALI nie mowi wiec nic o metodach, tylko o tym, ktora skale wybralismy. Jedyny
uczciwy odczyt jest przy zrownanym text alignment: pytamy, ile kazda metoda zapomina,
gdy trzyma to samo TA.

ODTWORZONY 2026-09-20. Oryginalny skrypt o tej nazwie, ktory policzyl wartosc 0,1839 stojaca
w Figurze 3, ZAGINAL -- nie ma go ani lokalnie, ani na klastrze. Per-skalowe liczby dla ziarna
2024 przetrwaly wylacznie w komentarzu w `figures/make_forgetting.py`. To jest odtworzenie
procedury z opisu, nie kopia tamtego kodu, wiec wynik dla nowych ziaren moze sie nieznacznie
roznic od tamtej liczby metoda interpolacji -- i dlatego skrypt drukuje oba punkty brzegowe,
zeby bylo widac, na czym stoi odczyt.

Run:
  python scripts/_forget_matched.py --target_ta 75.59 \
      --dirs outputs/matrix/finetune_s2025,outputs/matrix/finetune_s2026
"""
import argparse
import glob
import json
import os


def read_points(d):
    """[(TA w procentach, zapominanie DINO, skala)] dla wszystkich skal w katalogu."""
    pts = []
    for f in sorted(glob.glob(os.path.join(d, "s*", "cifc_metrics.json"))):
        tag = os.path.basename(os.path.dirname(f))          # 's08' -> 0.8, 's045' -> 0.45
        digits = tag[1:]
        scale = float(digits[0] + "." + digits[1:]) if len(digits) > 1 else float(digits)
        m = json.load(open(f, encoding="utf-8"))
        pts.append((100.0 * m["average_final"]["clip_t"],
                    m["forgetting"]["dino_i"], scale))
    return sorted(pts)


def interp(pts, target):
    """Liniowa interpolacja zapominania przy TA = target. None, gdy target poza zakresem."""
    for (ta0, f0, s0), (ta1, f1, s1) in zip(pts, pts[1:]):
        if ta0 <= target <= ta1:
            if ta1 == ta0:
                return f0, (s0, s1)
            w = (target - ta0) / (ta1 - ta0)
            return f0 + w * (f1 - f0), (s0, s1)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="katalogi macierzy po przecinku")
    ap.add_argument("--target_ta", type=float, required=True,
                    help="TA, przy ktorym odczytujemy (nasze trzyziarnowe: 75.59)")
    a = ap.parse_args()

    vals = []
    for d in [x.strip() for x in a.dirs.split(",") if x.strip()]:
        pts = read_points(d)
        name = os.path.basename(d.rstrip("/"))
        if not pts:
            print(f"{name}: brak macierzy")
            continue
        print(f"\n=== {name}")
        for ta, fg, sc in pts:
            print(f"    skala {sc:<5} TA {ta:6.2f}  zapominanie {fg:.4f}")
        v, br = interp(pts, a.target_ta)
        if v is None:
            lo, hi = pts[0][0], pts[-1][0]
            print(f"    TA {a.target_ta} POZA zakresem [{lo:.2f}, {hi:.2f}] -- potrzebna "
                  f"{'nizsza' if a.target_ta > hi else 'wyzsza'} skala")
        else:
            print(f"    przy TA {a.target_ta}: **{v:.4f}** (interpolacja miedzy skalami {br[0]} i {br[1]})")
            vals.append(v)

    if len(vals) > 1:
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        print(f"\n{len(vals)} ziaren: srednia {m:.4f}, sd {sd:.4f}")


if __name__ == "__main__":
    main()
