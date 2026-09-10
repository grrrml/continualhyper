"""Krzywa skalowania: te same koncepty ocenione po roznej liczbie zadan, przy ZROWNANYM TA.

Po co przy zrownanym TA, a nie przy stalej skali: skala adaptera przesuwa punkt pracy po
krzywej kompromisu tozsamosc-tekst, wiec sam spadek IA przy ustalonym s nie mowi, czy model
jest gorszy, czy tylko stoi w innym miejscu tej samej krzywej. Przy T=50 i s=0.45 TA rosnie
z 0.748 do 0.786, czyli adapter dziala slabiej -- odczyt wprost zawyzalby strate. Ten skrypt
interpoluje IA i DINO po skalach do wspolnego TA, tak jak kazde porownanie w pracy.

Dane wejsciowe to katalogi `curveA_t<K>/s<XX>/cifc_metrics.json`, po jednym na (checkpoint,
skala). Usredniamy TYLKO komorki (k=K, j), czyli koncepty ocenione tym checkpointem, bo
`average_final` w cifc_metrics bierze ostatnie zadanie konfiguracji, a tutaj kazdy przebieg
ma jeden checkpoint.

Run:  python scripts/_curve.py --root outputs/phaseT/T50_mixed --ta 0.7483
"""
import argparse
import glob
import json
import os


def cells(path, K):
    """Srednia po konceptach ocenionych checkpointem K w jednym pliku metryk."""
    M = json.load(open(path))["matrix"]
    c = [v for k, v in M.items() if int(k.split(",")[0]) == K]
    if not c:
        return None
    return {m: sum(x[m] for x in c) / len(c) for m in ("clip_t", "clip_i", "dino_i")}, len(c)


def at_ta(points, ta):
    """Interpoluj (IA, DINO) do zadanego TA po skalach. points: [(TA, IA, DINO)] rosnaco po TA."""
    pts = sorted(points)
    if len(pts) < 2:
        return None
    for a, b in zip(pts, pts[1:]):
        if a[0] <= ta <= b[0]:
            f = (ta - a[0]) / (b[0] - a[0])
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), "interp"
    a, b = (pts[-2], pts[-1]) if ta > pts[-1][0] else (pts[0], pts[1])
    f = (ta - a[0]) / (b[0] - a[0])
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), "EKSTRAP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="katalog z curveA_t<K>/s<XX>/")
    ap.add_argument("--ta", type=float, default=0.0,
                    help="wspolne TA; 0 = wez TA pierwszego punktu krzywej")
    a = ap.parse_args()

    per = {}
    for d in sorted(glob.glob(os.path.join(a.root, "curveA_t*"))):
        K = int(os.path.basename(d).split("_t")[1])
        for f in sorted(glob.glob(os.path.join(d, "s*", "cifc_metrics.json"))):
            got = cells(f, K)
            if got:
                m, n = got
                per.setdefault(K, []).append((m["clip_t"], m["clip_i"], m["dino_i"],
                                              os.path.basename(os.path.dirname(f)), n))
    if not per:
        raise SystemExit(f"brak danych w {a.root}")

    Ks = sorted(per)
    ta = a.ta or sorted(per[Ks[0]])[0][0]
    print(f"wspolne TA = {ta:.4f}\n")
    print(f"{'T':>4} {'n':>3}  {'skale (TA/IA)':<44} {'IA@TA':>8} {'DINO@TA':>9}  {'':>8}")
    base = None
    for K in Ks:
        rows = sorted(per[K])
        grid = " ".join(f"{r[3]}:{r[0]:.3f}/{r[1]:.3f}" for r in rows)
        got = at_ta([(r[0], r[1], r[2]) for r in rows], ta)
        if got is None:
            print(f"{K+1:>4} {rows[0][4]:>3}  {grid:<44} {'?':>8}")
            continue
        ia, di, how = got
        if base is None:
            base = (ia, di)
        print(f"{K+1:>4} {rows[0][4]:>3}  {grid:<44} {ia:8.4f} {di:9.4f}  "
              f"{ia-base[0]:+7.4f} {di-base[1]:+7.4f} {'' if how=='interp' else how}")


if __name__ == "__main__":
    main()
