"""Wizualny grid placementu GSA: koncepty obiektowe x ramka L/P, czerwony box na generacji.
Generacja 1:1 jak w _box_probe.py (te same seedy 31337+i). -> assets/figures/ground_grid_gsa.jpg
"""
import os, sys, torch
sys.path.insert(0, ".")
from PIL import Image, ImageDraw
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import set_grounded

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--gain", type=float, default=1.0)
_ap.add_argument("--sched", type=float, default=1.0)
_ap.add_argument("--out", default="assets/figures/ground_grid_gsa.jpg")
_ap.add_argument("--scene", default="", help="dopisek sceny do promptu, np. 'on a beach'")
_ap.add_argument("--quads", action="store_true", help="ramki = 4 cwiartki zamiast polowek")
_a = _ap.parse_args()
CKPT = "outputs/phaseP/P_ground_gsa/hyper.pt"
CFGP = "configs/phaseP/P_ground_gsa.yaml"
N = 2                                        # probki na (koncept, strona)
BOXES = {"L": (0.25, 0.5, 0.5, 1.0), "P": (0.75, 0.5, 0.5, 1.0)}
if _a.quads:
    BOXES = {"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
             "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5)}
    N = 1

cfg = load_config(CFGP)
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, CKPT, map_location="cuda"); manager.eval(); manager.lora_scale = 0.7
manager.ground_gain_base = _a.gain; manager.ground_sched_frac = _a.sched
set_grounded(bundle.unet, manager)

rows = []
for j, c in enumerate(cfg["concepts"]):
    if c["category"] == "style":
        continue
    cls = c["class_word"]
    prompt = f"a photo of {cls} {_a.scene}".strip() if _a.scene else f"a photo of {cls}"
    ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([""])
    tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
    row = []
    for side, box in BOXES.items():
        manager.cond_box = box
        for i in range(N):
            g = torch.Generator(device="cuda").manual_seed(31337 + i)
            img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=30,
                              guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
            pil = Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
            W, H = pil.size; cx, cy, bw, bh = box
            ImageDraw.Draw(pil).rectangle(
                [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
                outline=(255, 0, 0), width=6)
            row.append(pil.resize((256, 256), Image.LANCZOS))
    rows.append((c["concept_id"], row))
    print(f"[grid] {c['concept_id']}: {len(row)} obrazkow", flush=True)

cell, pad, lab = 256, 6, 120
Wg = lab + (cell + pad) * 2 * N
Hg = (cell + pad) * len(rows)
sheet = Image.new("RGB", (Wg, Hg), (255, 255, 255))
d = ImageDraw.Draw(sheet)
for r, (cid, row) in enumerate(rows):
    d.text((6, r * (cell + pad) + cell // 2 - 6), cid.replace("cifc_", ""), fill=(0, 0, 0))
    for k, im in enumerate(row):
        sheet.paste(im, (lab + k * (cell + pad), r * (cell + pad)))
os.makedirs("assets/figures", exist_ok=True)
sheet.save(_a.out, quality=92)
print(f"GRID_DONE -> {_a.out}", flush=True)
