"""(1) Mnoznik kappa na gate'ach przy inferencji + (2) surowsza sonda CWIARTKOWA.
Kryterium: argmax DINO po 4 cwiartkach == cwiartka ramki (przypadek = 25%).
Dla kazdego kappa: 7 konceptow x 4 cwiartki x n probek; przy okazji zapis probek psa i kaczki.
"""
import os, sys, glob, torch
sys.path.insert(0, ".")
import torch.nn.functional as Fn
from PIL import Image, ImageDraw
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import set_grounded
from src.cifc_metrics import _Dino

KAPPAS = [1.0, 2.0, 4.0, 8.0]
N = 3
Q = {"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
     "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5)}

cfg = load_config("configs/phaseP/P_ground_gsa.yaml")
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, "outputs/phaseP/P_ground_gsa/hyper.pt", map_location="cuda")
manager.eval(); manager.lora_scale = 0.7
set_grounded(bundle.unet, manager)
dino = _Dino("cuda")
os.makedirs("outputs/probe/gain", exist_ok=True)

def quad_crops(img):
    H, W = img.shape[-2:]
    return {"TL": img[:, :H//2, :W//2], "TR": img[:, :H//2, W//2:],
            "BL": img[:, H//2:, :W//2], "BR": img[:, H//2:, W//2:]}

for kap in KAPPAS:
    manager.ground_gain = kap
    ok = tot = 0
    for j, c in enumerate(cfg["concepts"]):
        if c["category"] == "style":
            continue
        ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
        cls = c["class_word"]; prompt = f"a photo of {cls}"
        ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([""])
        tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
        for qname, box in Q.items():
            manager.cond_box = box
            for i in range(N):
                g = torch.Generator(device="cuda").manual_seed(31337 + i)
                img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=30,
                                  guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
                sims = {}
                for qn, crop in quad_crops(img).items():
                    pil = Image.fromarray((crop.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
                    x = dino.tf(pil).unsqueeze(0).to("cuda")
                    with torch.no_grad():
                        f = Fn.normalize(dino.m(x), dim=-1)
                    sims[qn] = float((f @ ref.t()).item())
                hit = max(sims, key=sims.get) == qname
                ok += int(hit); tot += 1
                if c["concept_id"] in ("cifc_dog", "cifc_duck_toy") and i == 0:
                    pil = Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
                    W, H = pil.size; cx, cy, bw, bh = box
                    ImageDraw.Draw(pil).rectangle([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H],
                                                  outline=(255, 0, 0), width=6)
                    pil.save(f"outputs/probe/gain/k{int(kap)}_{c['concept_id']}_{qname}.png")
    print(f"KAPPA {kap}: cwiartki {ok}/{tot} = {ok/tot:.1%} (przypadek 25%)", flush=True)
print("GAIN_PROBE_DONE", flush=True)
