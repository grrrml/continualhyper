"""Efekty glowne sweepu: model liniowy po osmiu osiach, DWIE odpowiedzi osobno.

Czemu nie tak, jak planowal REPORT (jeden skalar `dino_t50` przy zrownanym TA):
`_sweep_score.py` czyta DINO interpolujac po skalach do TA=0.747, a dla wiekszosci
konfiguracji TA W OGOLE NIE REAGUJE NA SKALE -- jest plaskie w trzeciej cyfrze. Odczyt staje
sie wtedy ekstrapolacja o wspolczynniku do +2250 (zmierzone 2026-09-15: p016 dostal tak
`dino_t50` = 0.808, czyli wiecej niz cokolwiek zmierzonego w tym projekcie). Taki punkt
wygralby sweep jako artefakt dzielenia przez zero.

Czemu nie regula ograniczeniowa ("max DINO przy TA >= prog") jako odpowiedz modelu:
ona naprawia WARTOSC, ale wycina punkty niedosiegajace progu -- a dopuszczalnosc zalezy od TA,
ktore jest sterowane dokladnie tymi osiami, ktorych efekty szacujemy (mocniejszy adapter
obniza TA i wypycha punkt ponizej progu). Warunkowalibysmy wiec na zmiennej skorelowanej
z wynikiem, co OBCIAZA estymaty, nie tylko odbiera moc. Regula ograniczeniowa jest dobra do
WYBORU kandydata (tam selekcja nie szkodzi) i jest tu liczona osobno.

Co robimy zamiast: modelujemy kazda odpowiedz osobno przy WSPOLNEJ skali, na wszystkich
punktach. Os jest korzystna, jesli podnosi DINO i nie obniza TA -- widac to z dwoch tabel
wspolczynnikow, zamiast byc zwiniete w jedna liczbe, ktora przy plaskim TA wybucha.
Skala domyslnie 0.45, bo to jedyna obecna we WSZYSTKICH punktach; to najslabszy koniec
krzywej, wiec sprzyja TA kosztem tozsamosci -- kontrola przy 0.6 (`--scale s06`) na tych
punktach, ktore ja maja, jest czescia protokolu, a nie dodatkiem.

Run:  python scripts/_sweep_effects.py --root outputs/sweep
      python scripts/_sweep_effects.py --root outputs/sweep --scale s06
"""
import argparse
import glob
import json
import math
import os

import yaml

# (nazwa osi w configu, jak kodujemy)
AXES = [
    ("reg.space", "cat"),              # factors / dw   -> -1 / +1
    ("reg.weight", "log10"),
    ("task_cond.key_dim", "log2"),
    ("hyper.rank", "log2"),
    ("training.weight_decay", "ord"),  # 0, 1e-4, 1e-3, 1e-2 -> 0..3
    ("training.steps_per_task", "log2"),
    ("task_cond.learn_v", "bool"),
    ("task_cond.scale_cond", "bool"),
]
WD_LEVELS = [0.0, 1e-4, 1e-3, 1e-2]


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def encode(cfg, path, kind):
    v = dig(cfg, path)
    if kind == "cat":                       # reg.space; brak klucza = factors
        return 1.0 if str(v) == "dw" else -1.0
    if kind == "bool":
        return 1.0 if v is True else -1.0
    if v is None:
        return None
    if kind == "log10":
        return math.log10(float(v))
    if kind == "log2":
        return math.log2(float(v))
    if kind == "ord":
        v = float(v)
        return float(min(range(len(WD_LEVELS)), key=lambda i: abs(WD_LEVELS[i] - v)))
    raise ValueError(kind)


def curve(root, K):
    """[(skala, TA, IA, DINO)] dla tego konca krzywej."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, f"curveA_t{K}", "s*", "cifc_metrics.json"))):
        m = json.load(open(f))["matrix"]
        c = [v for k, v in m.items() if int(k.split(",")[0]) == K]
        if not c:
            continue
        out.append((os.path.basename(os.path.dirname(f)),
                    sum(x["clip_t"] for x in c) / len(c),
                    sum(x["clip_i"] for x in c) / len(c),
                    sum(x["dino_i"] for x in c) / len(c)))
    return out


def solve(X, y):
    """OLS przez rownania normalne, eliminacja Gaussa z czesciowym wyborem elementu glownego.
    Zwraca (beta, SE). Bez numpy, bo login node Heliosa go nie ma, a venv jest aarch64."""
    n, p = len(X), len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] + [0.0] * p
         for a in range(p)]
    for a in range(p):                      # doklejamy jednostkowa -> Gauss-Jordan daje inwersje
        A[a][p + a] = 1.0
    rhs = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-12:
            raise SystemExit("macierz planu jest osobliwa -- za malo punktow albo os stala")
        A[c], A[piv] = A[piv], A[c]
        rhs[c], rhs[piv] = rhs[piv], rhs[c]
        d = A[c][c]
        A[c] = [x / d for x in A[c]]
        rhs[c] /= d
        for r in range(p):
            if r == c:
                continue
            f = A[r][c]
            if f:
                A[r] = [x - f * y_ for x, y_ in zip(A[r], A[c])]
                rhs[r] -= f * rhs[c]
    beta = rhs
    inv = [row[p:] for row in A]
    fit = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    rss = sum((y[i] - fit[i]) ** 2 for i in range(n))
    dof = n - p
    if dof <= 0:
        return beta, [float("nan")] * p, float("nan")
    s2 = rss / dof
    se = [math.sqrt(max(s2 * inv[a][a], 0.0)) for a in range(p)]
    return beta, se, math.sqrt(s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/sweep")
    ap.add_argument("--scale", default="s045", help="wspolna skala odczytu (domyslnie s045)")
    ap.add_argument("--ta", type=float, default=0.747, help="prog TA do wyboru kandydata")
    a = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "p*"))):
        cfgp = os.path.join(d, "config.yaml")
        if not os.path.exists(cfgp):
            continue
        cfg = yaml.safe_load(open(cfgp, encoding="utf-8"))
        c49 = curve(d, 49)
        at = {s: (ta, ia, di) for s, ta, ia, di in c49}
        rows.append({"name": os.path.basename(d), "cfg": cfg, "c49": c49,
                     "pt": at.get(a.scale)})

    used = [r for r in rows if r["pt"] is not None]
    print(f"punktow z config.yaml: {len(rows)} | z odczytem przy {a.scale}: {len(used)}")
    if len(used) < len(AXES) + 2:
        print("ZA MALO PUNKTOW na model efektow glownych -- potrzeba co najmniej "
              f"{len(AXES) + 2}, jest {len(used)}. Ponizej tylko ranking kandydatow.")

    # ---- wybor kandydata: max DINO@T50 przy TA >= prog (bez interpolacji, bez selekcji na osiach)
    print(f"\n=== kandydaci: max DINO@T50 przy TA >= {a.ta} (T=50, wszystkie skale punktu)")
    cand = []
    for r in rows:
        ok = [di for _, ta, _, di in r["c49"] if ta >= a.ta]
        if ok:
            cand.append((max(ok), r["name"], len(r["c49"])))
    for v, n, k in sorted(cand, reverse=True)[:8]:
        print(f"    {n}  skal={k}  {v:.4f}")
    print(f"    dopuszczalnych {len(cand)} z {len(rows)}")

    if len(used) < len(AXES) + 2:
        return

    # ---- efekty glowne, dwie odpowiedzi osobno
    cols = []
    for path, kind in AXES:
        col = [encode(r["cfg"], path, kind) for r in used]
        if any(v is None for v in col):
            raise SystemExit(f"os {path} ma braki w configach")
        m = sum(col) / len(col)
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) or 1.0
        cols.append([(v - m) / sd for v in col])       # efekt na 1 odchylenie osi
    X = [[1.0] + [cols[j][i] for j in range(len(AXES))] for i in range(len(used))]

    for label, idx in (("DINO@T50", 2), ("TA@T50", 0)):
        y = [r["pt"][idx] for r in used]
        beta, se, sig = solve(X, y)
        print(f"\n=== efekty glowne na {label} przy {a.scale}  (n={len(used)}, "
              f"sigma reszt {sig:.4f})")
        print(f"    {'os':<28}{'efekt':>9}{'SE':>9}   istotne przy 2 SE")
        print(f"    {'(wyraz wolny)':<28}{beta[0]:>9.4f}{se[0]:>9.4f}")
        for j, (path, _) in enumerate(AXES, start=1):
            flag = "  <<<" if abs(beta[j]) > 2 * se[j] else ""
            print(f"    {path:<28}{beta[j]:>9.4f}{se[j]:>9.4f}{flag}")


main()
