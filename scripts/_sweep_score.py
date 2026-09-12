"""Jeden skalar z punktu sweepu: DINO przy T=50 odczytane przy ZROWNANYM TA.

Czemu taki cel, a nie samo zapominanie: optymalizacja po nachyleniu zwraca model zamrozony.
Mamy to zmierzone -- era przy beta=100 ma nachylenie porownywalne z baza (-0.041 wobec -0.038
DINO), a jest bezuzyteczna, bo stoi 0.20 DINO nizej. "DINO przy T=50 przy zrownanym TA" zawiera
i poziom, i nachylenie, i jest dokladnie ta liczba, ktora raportujemy w pracy.

Czemu przy zrownanym TA, a nie przy stalej skali: skala adaptera przesuwa punkt pracy po krzywej
kompromisu tozsamosc-tekst, wiec porownanie przy ustalonym s mieszaloby "gorszy model" z "ten sam
model w innym miejscu tej samej krzywej".

Wypisuje takze IA i punkt T=10, zeby w tabeli sweepu bylo widac, czy dany punkt kupil zatrzymanie
starych konceptow kosztem plastycznosci -- bez tego dwa bardzo rozne modele wygladaja tak samo.

Run:  python scripts/_sweep_score.py --root outputs/sweep/p001 --ta 0.747
"""
import argparse
import glob
import json
import os


def cells(path, K):
    M = json.load(open(path))["matrix"]
    c = [v for k, v in M.items() if int(k.split(",")[0]) == K]
    if not c:
        return None
    return {m: sum(x[m] for x in c) / len(c) for m in ("clip_t", "clip_i", "dino_i")}


def at_ta(points, ta):
    pts = sorted(points)
    if len(pts) < 2:
        return None
    for a, b in zip(pts, pts[1:]):
        if a[0] <= ta <= b[0]:
            f = (ta - a[0]) / (b[0] - a[0])
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), True
    a, b = (pts[-2], pts[-1]) if ta > pts[-1][0] else (pts[0], pts[1])
    f = (ta - a[0]) / (b[0] - a[0])
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), False


def endpoint(root, K, ta):
    pts = []
    for f in sorted(glob.glob(os.path.join(root, f"curveA_t{K}", "s*", "cifc_metrics.json"))):
        m = cells(f, K)
        if m:
            pts.append((m["clip_t"], m["clip_i"], m["dino_i"]))
    if len(pts) < 2:
        return None
    return at_ta(pts, ta), [p[0] for p in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ta", type=float, default=0.747)
    ap.add_argument("--wandb", default=None, help="nazwa runu W&B; bez tego tylko druk")
    a = ap.parse_args()

    out = {}
    for K, tag in ((9, "t10"), (49, "t50")):
        got = endpoint(a.root, K, a.ta)
        if got is None:
            raise SystemExit(f"za malo skal dla checkpointu {K} w {a.root}")
        (ia, dino, interp), tas = got
        out[f"ia_{tag}"], out[f"dino_{tag}"] = ia, dino
        out[f"interp_{tag}"] = interp
        if not interp:
            print(f"[score] UWAGA {tag}: TA={a.ta} poza zmierzonym zakresem "
                  f"[{min(tas):.3f}, {max(tas):.3f}] -- ekstrapolacja", flush=True)

    out["d_ia"] = out["ia_t50"] - out["ia_t10"]
    out["d_dino"] = out["dino_t50"] - out["dino_t10"]
    # Cel sweepu. Punkt z ekstrapolowanym odczytem dostaje kare, zeby BO nie gonilo modeli,
    # ktorych w ogole nie umiemy ustawic na wspolnym TA -- one sa poza protokolem porownania.
    out["objective"] = out["dino_t50"] - (0.0 if out["interp_t50"] and out["interp_t10"] else 0.05)

    print(json.dumps(out, indent=1))
    with open(os.path.join(a.root, "score.json"), "w") as f:
        json.dump(out, f, indent=1)

    if a.wandb:
        import wandb
        wandb.init(project="continualhyper-sweep", name=a.wandb, resume="allow")
        wandb.log(out)
        wandb.finish()


if __name__ == "__main__":
    main()
