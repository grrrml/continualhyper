"""Czy wygenerowane LoRA roznych konceptow sa wzajemnie ortogonalne?

Po co: praca ortogonalizuje WEJSCIA hipersieci (task embeddings), Orthogonal Adaptation
ortogonalizuje WAGI adapterow. Pytanie recenzenta: czy ortogonalne wejscia daja ortogonalne
wyjscia, i czy to w ogole potrzebne do niskiego zapominania. Mierzymy wprost: dla checkpointu
po zadaniu K bierzemy dW_i = alpha_i A_i B_i kazdego konceptu i liczymy kosinusy par
<dW_i, dW_j>_F / (||dW_i|| ||dW_j||) per warstwa, bez materializowania dW (tozsamosc sladu,
ta sama co w `_spectrum.dw_gram`). Do tego kosinusy na czynnikach osobno (A, B) -- te maja
wolnosc cechowania, wiec sa slabsza miara, ale porownywalna z tym, co ortogonalizuje OA.

Odniesienie: te same ksztalty wypelnione N(0,1) -- kosinus losowych macierzy jest rzedu
1/sqrt(in*out) i mowi, jaka wartosc "ortogonalny" znaczy w tej liczbie wymiarow.

Run:  python scripts/_lora_ortho.py --config outputs/sweep/p022/config.yaml \
          --ckpt outputs/sweep/p022/ckpts/hyper_after_task09.pt,outputs/sweep/p022/ckpts/hyper_after_task49.pt \
          --tasks 10,50 --out outputs/sweep/p022/lora_ortho.json
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
from scripts._spectrum import dw_gram                   # noqa: E402


def cos_from_gram(G):
    d = G.diagonal().clamp_min(1e-30).sqrt()
    return G / (d[:, None] * d[None, :])


def offdiag_stats(Cm):
    T = Cm.shape[0]
    mask = ~torch.eye(T, dtype=torch.bool, device=Cm.device)
    v = Cm[mask]
    return {"mean_abs": float(v.abs().mean()), "mean": float(v.mean()),
            "max_abs": float(v.abs().max()), "frac_abs_gt_0.1": float((v.abs() > 0.1).float().mean())}


def group_of(name):
    for g in ("to_q", "to_k", "to_v", "to_out"):
        if g in name:
            return g
    return "other"


def one(manager, bundle, ckpt, T, device, gen):
    with torch.no_grad():
        manager.cond_box = None
        if getattr(manager, "time_cond", False):
            manager.cond_t = 0.5
        if getattr(manager, "latent_cond", False):
            manager.cond_latent = None
        dummy = torch.zeros(1, int(bundle.clip_hidden_size), device=device)
        C = torch.cat([manager.condition(dummy, k, use_prompt_mod=False) for k in range(T)], 0)
        per_layer, cos_sum, cos_n = [], None, 0
        rnd_sum = 0.0
        for name in manager.layer_names:
            head = manager.heads[_key(name)]
            x = manager._head_input(C, name)
            x_L, x_R = head(x)[1:]                          # alpha juz wliczona w x_L
            Cd = cos_from_gram(dw_gram(x_L, x_R))           # [T, T] kosinusy dW
            CL = cos_from_gram(x_L.flatten(1) @ x_L.flatten(1).T)
            CR = cos_from_gram(x_R.flatten(1) @ x_R.flatten(1).T)
            # odniesienie losowe o tych samych ksztaltach
            rL = torch.randn(x_L.shape, generator=gen, device=device)
            rR = torch.randn(x_R.shape, generator=gen, device=device)
            Cr = cos_from_gram(dw_gram(rL, rR))
            per_layer.append({"layer": name, "group": group_of(name),
                              "dw": offdiag_stats(Cd), "A": offdiag_stats(CL), "B": offdiag_stats(CR),
                              "random_dw": offdiag_stats(Cr),
                              "shape": [int(x_L.shape[1]), int(x_R.shape[2])]})
            cos_sum = Cd.clone() if cos_sum is None else cos_sum + Cd
            cos_n += 1
            rnd_sum += offdiag_stats(Cr)["mean_abs"]
        mean_cos = (cos_sum / cos_n).cpu()
    summary = {}
    for g in ("to_q", "to_k", "to_v", "to_out"):
        rows = [r for r in per_layer if r["group"] == g]
        if rows:
            summary[g] = {k: sum(r[k]["mean_abs"] for r in rows) / len(rows) for k in ("dw", "A", "B", "random_dw")}
    summary["all"] = {k: sum(r[k]["mean_abs"] for r in per_layer) / len(per_layer) for k in ("dw", "A", "B", "random_dw")}
    print(f"\nckpt {os.path.basename(ckpt)} | T={T} | {len(per_layer)} warstw | sredni |cos| par (poza przekatna):")
    print(f"  {'grupa':<8}{'dW':>8}{'A':>8}{'B':>8}{'losowe dW':>12}")
    for g, v in summary.items():
        print(f"  {g:<8}{v['dw']:>8.3f}{v['A']:>8.3f}{v['B']:>8.3f}{v['random_dw']:>12.4f}")
    # najbardziej podobne pary wg sredniego kosinusa dW po warstwach
    M = mean_cos.clone(); M.fill_diagonal_(-2)
    top = torch.topk(M.flatten(), min(5, T * T - T)).indices
    print("  najblizsze pary (sredni cos dW po warstwach): " +
          ", ".join(f"({int(i // T)},{int(i % T)}) {M.flatten()[i]:.3f}" for i in top))
    return {"ckpt": ckpt, "T": T, "summary": summary, "per_layer": per_layer,
            "mean_cos_dw": mean_cos.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint albo kilka po przecinku")
    ap.add_argument("--tasks", required=True, help="ile konceptow na checkpoint, po przecinku")
    ap.add_argument("--out", default=None, help="plik json z kosinusami")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp32")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    gen = torch.Generator(device=device).manual_seed(a.seed)
    ckpts = a.ckpt.split(",")
    tasks = [int(x) for x in str(a.tasks).split(",")]
    if len(ckpts) != len(tasks):
        raise SystemExit(f"--ckpt ma {len(ckpts)} pozycji, --tasks {len(tasks)}")
    out_all = []
    for ckpt, T in zip(ckpts, tasks):
        load_hyper(manager, ckpt, map_location=str(device))
        manager.eval()
        out_all.append(one(manager, bundle, ckpt, T, device, gen))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out_all, open(a.out, "w"), indent=1)
        print(f"[ortho] zapisane do {a.out}")


if __name__ == "__main__":
    main()
