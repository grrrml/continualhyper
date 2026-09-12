"""Punkty sweepu metoda Latin Hypercube: wypisuje gotowe linie `--set ...` dla sbatch_sweep.sh.

Czemu LHS, a nie optymalizacja bayesowska: przewaga BO bierze sie z wielu rund SEKWENCYJNYCH,
a przy 6-7 h na punkt i deadlinie 25.09 zrobimy najwyzej cztery. Przy tylu rundach losowanie
stratyfikowane jest praktycznie rownie dobre, a nie wymaga agentow, serwera trzymajacego stan
ani procesu, ktory musi przezyc miedzy generacjami. Kazdy punkt jest niezaleznym zadaniem,
wiec awaria punktu kosztuje punkt, a nie sweep.

Czemu 8 wymiarow mimo 48 punktow: liczba wymiarow prawie nie wplywa na moc wykrycia efektu
GLOWNEGO -- kazdy parametr szacujemy porownaniem N/2 punktow "wysokich" z N/2 "niskimi", i to
jest N/2 na N/2 niezaleznie od d. Wiazaca liczba jest N. Przy zmierzonym rozrzucie celu
sigma ~ 0.035 DINO blad standardowy to sigma*sqrt(4/N) = 0.010, czyli wykrywamy efekty od
~0.02 DINO. Wymiarowosc boli dopiero przy lokalizowaniu optimum z interakcjami -- tego przy
48 punktach na 8 osiach NIE rozstrzygniemy i tak tez trzeba to raportowac.

Run:  python scripts/_sweep_points.py --n 48 > sweep_points.txt
"""
import argparse

import numpy as np

# (sciezka w configu, wartosci albo (lo, hi) dla log-uniform)
SPACE = [
    ("reg.space",                ["factors:str", "dw:str"]),
    ("reg.weight",               ("log", 50.0, 10000.0)),
    ("task_cond.key_dim",        [128, 256, 512]),
    ("hyper.rank",               [2, 4, 8]),
    ("training.weight_decay",    [0.0, 1e-4, 1e-3, 1e-2]),
    ("training.steps_per_task",  [400, 800]),
    ("task_cond.learn_v",        ["false:bool", "true:bool"]),
    ("task_cond.scale_cond",     ["false:bool", "true:bool"]),
]


def lhs(n, d, rng):
    """Latin Hypercube: kazdy wymiar podzielony na n rownych przedzialow, po jednej probce
    w kazdym, kolejnosc przedzialow permutowana niezaleznie per wymiar."""
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--prefix", default="p")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    u = lhs(a.n, len(SPACE), rng)

    for i in range(a.n):
        sets = []
        for j, (path, spec) in enumerate(SPACE):
            if isinstance(spec, tuple):
                lo, hi = spec[1], spec[2]
                v = float(np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo))))
                v = float(f"{v:.3g}")
            else:
                v = spec[min(int(u[i, j] * len(spec)), len(spec) - 1)]
            sets.append(f"{path}={v}")
        print(f"{a.prefix}{i:03d} " + " ".join(sets))


if __name__ == "__main__":
    main()
