"""Ile wymiarow naprawde uzywa hipersiec: widmo LoRA emitowanych dla kolejnych konceptow.

Po co: `HyperHead` to Linear(cond, hidden) -> SiLU -> Linear(hidden, out), wiec nieliniowosc
stoi PRZED warstwa wyjsciowa i emitowana LoRA lezy dokladnie w zbiorze afinicznym
`b2 + span(kolumny W2)` o wymiarze co najwyzej `hidden`. Przy `head_hidden: 50` wszystkie
adaptery, jakie ta siec moze kiedykolwiek wyprodukowac, mieszcza sie w 50-wymiarowej
podprzestrzeni -- niezaleznie od tego, ile konceptow jej pokazemy. Jesli koncepty potrzebuja
kierunkow w przyblizeniu niezaleznych (a pomiar z Fazy F mowi, ze tak: przy T=10 kolumny x_L
rozpinaly 39 z 40 wymiarow), to przy T rzedu `hidden` glowica musi zaczac je sklejac.

Skrypt liczy to wprost. Dla checkpointu po zadaniu K bierze warunkowanie kazdego konceptu
0..K, emituje LoRA, uklada per warstwa macierz [T, wymiar] i patrzy na widmo wartosci
osobliwych. Macierz jest CENTROWANA po konceptach, bo stala `b2` jest wspolna dla wszystkich
i nie niesie informacji o rozroznianiu ich miedzy soba.

Raportujemy rzad efektywny: najmniejsze r, dla ktorego suma kwadratow r pierwszych wartosci
osobliwych pokrywa 99% (i 95%) energii. Porownanie K=9 z K=49 odpowiada na pytanie, czy
koncepty zaczely dzielic kierunki, czy wciaz maja swoje wlasne.

Mierzymy dwie rzeczy:
  * hidden -- aktywacje SiLU(W1 c + b1), [T, hidden]. To jest samo waskie gardlo.
  * lora   -- splaszczone x_L i x_R, [T, in*r] i [T, r*out]. To, co z gardla wychodzi.

Run:  python -m scripts._spectrum --config configs/phaseT/T50_mixed.yaml \
          --ckpt outputs/phaseT/T50_mixed/ckpts/hyper_after_task49.pt --tasks 50
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


def eff_rank(M, frac):
    """Najmniejsze r pokrywajace `frac` energii widma macierzy juz wycentrowanej."""
    s = torch.linalg.svdvals(M.double())
    e = (s ** 2).cumsum(0) / (s ** 2).sum().clamp_min(1e-30)
    return int((e < frac).sum().item()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint albo kilka po przecinku; model ladowany raz")
    ap.add_argument("--tasks", required=True,
                    help="ile konceptow na checkpoint, po przecinku w tej samej kolejnosci")
    ap.add_argument("--out", default=None, help="plik json z widmami")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp32")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    hidden = int(cfg.get("hyper", {}).get("head_hidden", 0))
    ckpts = a.ckpt.split(",")
    tasks = [int(x) for x in str(a.tasks).split(",")]
    if len(ckpts) != len(tasks):
        raise SystemExit(f"--ckpt ma {len(ckpts)} pozycji, --tasks {len(tasks)}")

    out_all = []
    for ckpt, T in zip(ckpts, tasks):
        load_hyper(manager, ckpt, map_location=str(device))
        manager.eval()
        out_all.append(one(manager, bundle, ckpt, T, hidden, device))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out_all, open(a.out, "w"), indent=1)
        print(f"[spectrum] zapisane do {a.out}")


def one(manager, bundle, ckpt, T, hidden, device):
    with torch.no_grad():
        # warunkowanie jest STALE per zadanie i ignoruje `pooled` (patrz manager.condition),
        # wiec podajemy zera; kotwice w treningu powstaja dokladnie tak samo (use_prompt_mod
        # wylaczone, box na pelna klatke).
        manager.cond_box = None
        if getattr(manager, "time_cond", False):
            manager.cond_t = 0.5
        if getattr(manager, "latent_cond", False):
            manager.cond_latent = None
        dummy = torch.zeros(1, int(bundle.clip_hidden_size), device=device)
        C = torch.cat([manager.condition(dummy, k, use_prompt_mod=False) for k in range(T)], 0)

        rows = []
        for name in manager.layer_names:
            head = manager.heads[_key(name)]
            x = manager._head_input(C, name)
            hL = head.left[1](head.left[0](x))            # SiLU(W1 c + b1), [T, hidden]
            hR = head.right[1](head.right[0](x))
            x_L, x_R = head(x)[1:]
            rows.append({
                "layer": name,
                "hidden_L": eff_rank(hL - hL.mean(0, keepdim=True), 0.99),
                "hidden_R": eff_rank(hR - hR.mean(0, keepdim=True), 0.99),
                "lora_L99": eff_rank((x_L.flatten(1) - x_L.flatten(1).mean(0, keepdim=True)), 0.99),
                "lora_R99": eff_rank((x_R.flatten(1) - x_R.flatten(1).mean(0, keepdim=True)), 0.99),
                "lora_L95": eff_rank((x_L.flatten(1) - x_L.flatten(1).mean(0, keepdim=True)), 0.95),
                # Ile wymiarow R^in zajmuja LACZNIE kolumny x_L wszystkich konceptow. To jest
                # dokladnie to, co musi rozpiac wspolna baza U przy basis_q, wiec ta liczba mowi,
                # jak duze q nie zaczyna wiazac przy danym T. NIE centrujemy: baza ma pokryc
                # rzeczywiste kolumny, razem ze skladowa wspolna.
                "basis_L99": eff_rank(x_L.permute(1, 0, 2).reshape(x_L.shape[1], -1), 0.99),
                "basis_R99": eff_rank(x_R.permute(0, 1, 2).reshape(-1, x_R.shape[2]), 0.99),
            })

    def med(k):
        v = sorted(r[k] for r in rows)
        return v[len(v) // 2]

    cap = min(T - 1, hidden)     # po centrowaniu tracimy jeden stopien swobody
    print(f"\nckpt {os.path.basename(ckpt)} | T={T} | head_hidden={hidden} | "
          f"gorna granica rzedu po centrowaniu = {cap}")
    print(f"{'wielkosc':>10} {'mediana':>8} {'min':>5} {'max':>5}   (po {len(rows)} warstwach)")
    for k in ("hidden_L", "hidden_R", "lora_L99", "lora_R99", "lora_L95",
              "basis_L99", "basis_R99"):
        v = [r[k] for r in rows]
        print(f"{k:>10} {med(k):8d} {min(v):5d} {max(v):5d}")
    print(f"\nwykorzystanie gardla (mediana hidden_L / {cap}): "
          f"{med('hidden_L') / cap:.1%}")

    return {"ckpt": ckpt, "T": T, "hidden": hidden, "cap": cap, "layers": rows}


if __name__ == "__main__":
    main()
