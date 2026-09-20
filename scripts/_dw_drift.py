"""Dryf generowanych aktualizacji: o ile dW konceptu j odplywa, gdy siec uczy sie kolejnych zadan.

Po co osobny pomiar. Zapominanie mierzymy dotad na obrazach (DINO peak-final), czyli trzy
warstwy od mechanizmu. Regularyzator wyjscia obiecuje jedna konkretna rzecz: ze dW generowane
dla STARYCH osadzen nie zmieni sie, gdy sieci przybywa zadan. To jest dokladnie ta wielkosc,
i da sie ja policzyc z samych checkpointow, bez generowania ani jednego obrazu.

Definicja. Dla konceptu j i checkpointu k >= j:

    drift(j, k) = || dW_j(phi_k) - dW_j(phi_j) ||_F  /  || dW_j(phi_j) ||_F

liczone per warstwa i usredniane. Rozwijamy kwadrat normy roznicy, zeby nie materializowac
zadnego [in, out]:

    ||X - Y||^2 = <X,X> + <Y,Y> - 2<X,Y>

a kazdy z trzech skladnikow wychodzi z macierzy r x r, ta sama tozsamosc sladu co w
`_spectrum.dw_gram` i w Appendiksie o regularyzatorze. Przy r = 4 to jest darmowe.

Uwaga na cechowanie: dW jest niezmiennicze na (A R, R^-1 B), a normy czynnikow nie sa --
dlatego liczymy wylacznie na dW, nigdy na A i B osobno.

Run:  python scripts/_dw_drift.py --config outputs/sweep/p022/config.yaml \
          --ckpt_dir outputs/sweep/p022/ckpts --tasks 10 --out outputs/sweep/p022/dw_drift.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config, load_hyper          # noqa: E402
from src.injection import DEFAULT_TARGETS               # noqa: E402
from src.manager import build_hyper, _key               # noqa: E402
from src.sd_loader import load_sd                       # noqa: E402


def dw_cross(A1, B1, A2, B2):
    """<dW_i^(1), dW_j^(2)> dla dW = A @ B, bez materializowania [in, out].
    Uogolnienie `_spectrum.dw_gram` na DWA rozne zestawy czynnikow."""
    X = torch.einsum("iar,jas->ijrs", A1, A2)
    Y = torch.einsum("jsb,irb->ijsr", B2, B1)
    return (X * Y.transpose(-1, -2)).sum((-1, -2))


def rel_dw_diff(ref_a, ref_b):
    """Wzgledna roznica ||dW_a - dW_b|| / ||dW_a||, ta sama miara co dryf.
    Uzywana wylacznie do kontroli: referencja wczytana z pliku kontra policzona na miejscu."""
    num, den = 0.0, 0.0
    for n, (aL, aR) in ref_a.items():
        bL, bR = ref_b[n]
        aa = float(dw_cross(aL, aR, aL, aR)[0, 0])
        bb = float(dw_cross(bL, bR, bL, bR)[0, 0])
        ab = float(dw_cross(aL, aR, bL, bR)[0, 0])
        num += max(aa + bb - 2 * ab, 0.0) ** 0.5
        den += max(aa, 1e-30) ** 0.5
    return num / max(den, 1e-30)


def factors(manager, bundle, T, device):
    """{nazwa warstwy: (x_L, x_R)} dla wszystkich T konceptow na biezacym checkpointcie."""
    with torch.no_grad():
        manager.cond_box = None
        if getattr(manager, "time_cond", False):
            manager.cond_t = 0.5
        if getattr(manager, "latent_cond", False):
            manager.cond_latent = None
        dummy = torch.zeros(1, int(bundle.clip_hidden_size), device=device)
        C = torch.cat([manager.condition(dummy, k, use_prompt_mod=False) for k in range(T)], 0)
        out = {}
        for name in manager.layer_names:
            head = manager.heads[_key(name)]
            x_L, x_R = head(manager._head_input(C, name))[1:]   # alpha juz w x_L
            out[name] = (x_L.detach(), x_R.detach())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tasks", type=int, required=True, help="ile konceptow w strumieniu")
    ap.add_argument("--ref_factors", default=None,
                    help="plik z _dw_drift_refs.py: dW_j(phi_j) dla checkpointow, ktorych tu nie ma")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp32")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))

    refs = {}          # j -> {warstwa: (x_L_j, x_R_j)} zlapane na checkpointcie j
    drift = {}         # "j,k" -> sredni wzgledny dryf po warstwach

    # Referencje z innego klastra. Strumien wznawiany ma checkpointy rozdzielone miedzy maszyny
    # (p022_v2: 00-09 na Heliosie, 09-49 na Athenie), a dW_j(phi_j) dla brakujacych j inaczej
    # nie istnieje -- i to wlasnie dalo pusty by_lag w jobie 3184647.
    pre = {}
    if a.ref_factors:
        pre = {int(j): {n: (aL.to(device), aR.to(device)) for n, (aL, aR) in d.items()}
               for j, d in torch.load(a.ref_factors, map_location="cpu").items()}
        refs.update(pre)
        print(f"[drift] wczytane referencje dla zadan {sorted(pre)} z {a.ref_factors}", flush=True)

    for k in range(a.tasks):
        ck = os.path.join(a.ckpt_dir, f"hyper_after_task{k:02d}.pt")
        if not os.path.exists(ck):
            # brak checkpointu k oznacza tylko, ze nie policzymy PAR konczacych sie na k;
            # pary dla pozniejszych k zyja dalej, o ile mamy referencje dla ich j
            print(f"[drift] brak {ck}, pomijam k={k}", flush=True)
            continue
        load_hyper(manager, ck, map_location=str(device))
        # checkpoint po zadaniu k niesie warunkowanie kanoniczne tylko dla zadan 0..k,
        # wiec pytamy o tyle konceptow, ile ten checkpoint widzial -- nie o caly strumien
        F = factors(manager, bundle, k + 1, device)
        here = {n: (v[0][k:k + 1].clone(), v[1][k:k + 1].clone()) for n, v in F.items()}
        if k in pre:
            # ten checkpoint jest i tutaj, i w pliku referencji -- jedyny moment, w ktorym
            # da sie sprawdzic, czy przenoszenie referencji miedzy klastrami w ogole wolno robic
            print(f"[drift] KONTROLA k={k}: referencja z pliku vs policzona tu, "
                  f"wzgledna roznica {rel_dw_diff(pre[k], here):.2e}", flush=True)
        refs[k] = here

        for j in range(k):
            if j not in refs:
                continue
            num, den = 0.0, 0.0
            for n, (xL, xR) in F.items():
                aL, aR = refs[j][n]                       # dW_j(phi_j)
                bL, bR = xL[j:j + 1], xR[j:j + 1]         # dW_j(phi_k)
                aa = float(dw_cross(aL, aR, aL, aR)[0, 0])
                bb = float(dw_cross(bL, bR, bL, bR)[0, 0])
                ab = float(dw_cross(aL, aR, bL, bR)[0, 0])
                num += max(aa + bb - 2 * ab, 0.0) ** 0.5
                den += max(aa, 1e-30) ** 0.5
            drift[f"{j},{k}"] = num / max(den, 1e-30)
        print(f"[drift] checkpoint {k}: policzone {k} par", flush=True)

    # podsumowanie po opoznieniu k - j
    by_lag = {}
    for key, v in drift.items():
        j, k = (int(x) for x in key.split(","))
        by_lag.setdefault(k - j, []).append(v)
    print("\nsredni wzgledny dryf dW starego konceptu, wg liczby zadan od jego wlasnego:")
    for lag in sorted(by_lag):
        vs = by_lag[lag]
        print(f"  po {lag:2d} zadaniach: {sum(vs) / len(vs):.4f}  (n={len(vs)})")

    if a.out:
        json.dump({"drift": drift,
                   "by_lag": {str(l): sum(v) / len(v) for l, v in by_lag.items()}},
                  open(a.out, "w"), indent=1)
        print("[drift] zapisane do", a.out)


if __name__ == "__main__":
    main()
