"""Co lezy MIEDZY task embeddingami i POZA nimi: interpolacja, wektor spoza bazy, norma jako suwak.

Po co: embeddingi sa losowe i ortogonalne, hipersiec widziala tylko wierzcholki. Trzy sondy,
wszystkie inference-only:
  * interpolacja  h = (1-a) h_i + a h_j dla par konceptow -- czy przejscie tozsamosci jest gladkie
    (ciagla mapa embedding -> adapter), czy skokowe (koncepty w osobnych "komorkach");
  * spoza bazy    losowy wektor jednostkowy ortogonalny do WSZYSTKICH z_i, przeskalowany do mediany
    normy kluczy -- co generuje siec dla konceptu, ktorego nie uczyla;
  * norma         s * h_i dla kilku s -- czy dlugosc embeddingu dziala jak sila adaptera.

Adapter dostaje mieszany klucz przez podmiane `manager.condition`; galaz umiejscowienia jest
wylaczona (kappa=0), zeby patrzec tylko na tor LoRA. Prompt bez slowa klasy i bez maski tokenow,
zeby to adapter, a nie tekst, decydowal o tozsamosci.

Run:  python scripts/_embed_probe.py --config outputs/sweep/p022/config.yaml \
          --ckpt outputs/sweep/p022/hyper.pt --pairs 0:2,0:6,2:8,3:4 --out outputs/sweep/p022/embed_probe
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config, load_hyper          # noqa: E402
from src.injection import DEFAULT_TARGETS               # noqa: E402
from src.manager import build_hyper                     # noqa: E402
from src.sampling import ddim_sample                    # noqa: E402
from src.sd_loader import load_sd                       # noqa: E402


def to_pil(img):
    return Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def sheet(rows, cell=256, pad=4):
    W = max(len(r) for r in rows) * (cell + pad)
    H = len(rows) * (cell + pad)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    for r, row in enumerate(rows):
        for k, im in enumerate(row):
            out.paste(im.resize((cell, cell), Image.LANCZOS), (k * (cell + pad), r * (cell + pad)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", default="0:2,0:6,2:8,1:4", help="pary indeksow zadan i:j")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--norms", default="0,0.5,1,1.5,2", help="mnozniki normy dla --norm_task")
    ap.add_argument("--norm_task", type=int, default=0)
    ap.add_argument("--offbasis", type=int, default=4, help="ile losowych wektorow spoza bazy")
    ap.add_argument("--prompt", default="a photo")
    ap.add_argument("--n", type=int, default=2, help="ziarna na komorke")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp16")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    load_hyper(manager, a.ckpt, map_location=str(device))
    manager.eval()
    manager.lora_scale = a.scale
    manager.ground_gain_base = 0.0                     # tylko tor LoRA
    T = int(manager.basis_count.item())
    names = [c["concept_id"] for c in cfg["concepts"]]

    with torch.no_grad():
        manager.cond_box = None
        dummy = torch.zeros(1, int(bundle.clip_hidden_size), device=device)
        keys = torch.cat([manager.condition(dummy, k, use_prompt_mod=False) for k in range(T)], 0)   # [T, D]
    med = keys.norm(dim=-1).median()
    real_condition = manager.condition
    state = {"h": None}

    def fake_condition(pooled, task_idx=None, use_prompt_mod=True):
        # Klucze zostaja w fp32: glowice hipersieci sa fp32, a `pooled` z fp16 backbone jest half.
        return state["h"].expand(pooled.shape[0], -1)

    manager.condition = fake_condition
    cond_hidden, pooled, _ = bundle.encode_text([a.prompt])
    uncond_hidden, uncond_pooled, _ = bundle.encode_text([""])

    def gen(h, seed):
        state["h"] = h.unsqueeze(0)
        g = torch.Generator(device=device).manual_seed(31337 + seed)
        img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled, num_inference_steps=a.steps,
                          guidance_scale=7.5, batch_size=1, generator=g, task_idx=0, token_mask=None,
                          uncond_pooled=uncond_pooled)[0]
        return to_pil(img)

    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]
    for pair in a.pairs.split(","):
        i, j = (int(x) for x in pair.split(":"))
        rows = [[gen((1 - al) * keys[i] + al * keys[j], s) for al in alphas] for s in range(a.n)]
        sheet(rows).save(os.path.join(a.out, f"interp_{i:02d}_{names[i]}__{j:02d}_{names[j]}.jpg"), quality=90)
        print(f"[probe] interpolacja {names[i]} -> {names[j]}: alfa {alphas}", flush=True)

    norms = [float(x) for x in a.norms.split(",")]
    rows = [[gen(sc * keys[a.norm_task], s) for sc in norms] for s in range(a.n)]
    sheet(rows).save(os.path.join(a.out, f"norm_{a.norm_task:02d}_{names[a.norm_task]}.jpg"), quality=90)
    print(f"[probe] norma {names[a.norm_task]}: mnozniki {norms}", flush=True)

    if a.offbasis:
        basis = manager.ortho_basis[:T].to(device)
        g = torch.Generator(device=device).manual_seed(4242)
        rows = []
        for s in range(a.n):
            row = []
            for k in range(a.offbasis):
                v = torch.randn(keys.shape[1], generator=g, device=device)
                v = v - basis.t() @ (basis @ v)
                v = v / v.norm() * med
                row.append(gen(v, s))
            rows.append(row)
        sheet(rows).save(os.path.join(a.out, "offbasis.jpg"), quality=90)
        print(f"[probe] {a.offbasis} wektory spoza bazy (norma = mediana kluczy {med:.2f})", flush=True)
    manager.condition = real_condition
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
