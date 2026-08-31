"""Siatka PORÓWNAWCZA kandydatów na punkt pracy: wiersz = koncept, kolumna = wariant x ziarno.

Po co osobny skrypt: `_ground_grid.py` robi siatkę JEDNEJ konfiguracji, a pytanie brzmi
"czy tlo jest gorsze i o ile", wiec warianty musza stac obok siebie na tym samym ziarnie
i tej samej ramce. Pierwsza kolumna to KONTROLA bez ramki (pelny kadr, kappa=1) -- czyli
jak tlo wyglada, gdy grounding nie dziala przestrzennie. Bez tej kolumny nie ma z czym
porownac "plaskiego tla".

Dwa ziarna na wariant, bo ocena wizualna z jednej probki jest bezwartosciowa.

Run: python -u scripts/_ground_compare.py --box TL --kappa 4 --sched 0.15
"""
import os, sys, argparse
import torch
sys.path.insert(0, ".")
from PIL import Image, ImageDraw
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import set_grounded

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_ground_gsa.yaml")
ap.add_argument("--ckpt_base", default="outputs/phaseP/P_ground_gsa/hyper.pt")
ap.add_argument("--ckpt_nocap", default="outputs/phaseP/P_ground_gsa_nocap/hyper.pt")
ap.add_argument("--box", default="TL", choices=["TL", "TR", "BL", "BR", "L", "R"])
ap.add_argument("--kappa", type=float, default=4.0)
ap.add_argument("--sched", type=float, default=0.15)
ap.add_argument("--confine", type=float, default=3.0)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--scale", type=float, default=0.7)
ap.add_argument("--seed0", type=int, default=31337)
# Domyslnie na scratch (results/ jest symlinkiem i jest gitignorowane), NIE do assets/.
# Job zapisujacy plik do katalogu SLEDZONEGO gitem blokuje pozniej 'git pull --ff-only'
# w klonie na klastrze, jesli ta sama sciezka zostanie zacommitowana lokalnie - i od tego
# momentu kazdy run.sh startuje po cichu na starym kodzie. Do repo figura trafia z lokalnej
# kopii, po scp.
ap.add_argument("--out", default="results/figures/ground_compare.jpg")
a = ap.parse_args()

BOXES = {"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
         "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5),
         "L": (0.25, 0.5, 0.5, 1.0), "R": (0.75, 0.5, 0.5, 1.0)}
BOX = BOXES[a.box]
FULL = (0.5, 0.5, 1.0, 1.0)

# (etykieta, checkpoint, box, kappa, sched, confine, tail)
VARIANTS = [
    ("kontrola bez ramki", a.ckpt_base, FULL, 1.0, 1.0, 0.0, False),
    ("base", a.ckpt_base, BOX, a.kappa, a.sched, 0.0, False),
    ("base+tail", a.ckpt_base, BOX, a.kappa, a.sched, a.confine, True),
    ("nocap+tail", a.ckpt_nocap, BOX, a.kappa, a.sched, a.confine, True),
]

cfg = load_config(a.config)
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
set_grounded(bundle.unet, manager)
print(f"[env] {torch.cuda.get_device_name(0)} | box {a.box} | kappa {a.kappa} | "
      f"sched {a.sched} | confine {a.confine}", flush=True)

concepts = [(j, c) for j, c in enumerate(cfg["concepts"]) if c.get("category") != "style"]
cells = {}
loaded = None
for vi, (lab, ckpt, box, kap, sch, conf, tail) in enumerate(VARIANTS):
    if ckpt != loaded:                        # ten sam manager, inne wagi
        load_hyper(manager, ckpt, map_location="cuda")
        manager.eval()
        manager.lora_scale = a.scale
        loaded = ckpt
    manager.ground_gain_base = kap
    manager.ground_sched_frac = sch
    manager.ground_confine = conf
    manager.ground_confine_tail = tail
    manager.cond_box = box
    for j, c in concepts:
        cls = c["class_word"]
        prompt = "a photo of " + cls
        ch, pooled, _ = bundle.encode_text([prompt])
        uh, _, _ = bundle.encode_text([""])
        tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() \
            if cfg.get("token_mask_lora") else None
        for i in range(a.n):
            g = torch.Generator(device="cuda").manual_seed(a.seed0 + i)
            img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=a.steps,
                              guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
            pil = Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
            if box != FULL:               # kontrola bez ramki nie dostaje prostokata
                W, H = pil.size
                cx, cy, bw, bh = box
                ImageDraw.Draw(pil).rectangle(
                    [(cx - bw / 2) * W, (cy - bh / 2) * H,
                     (cx + bw / 2) * W, (cy + bh / 2) * H],
                    outline=(255, 0, 0), width=5)
            cells[(vi, j, i)] = pil.resize((256, 256), Image.LANCZOS)
    print(f"[cmp] wariant '{lab}' gotowy", flush=True)

cell, pad, lab_w, hdr = 256, 6, 130, 26
cols = len(VARIANTS) * a.n
sheet = Image.new("RGB", (lab_w + (cell + pad) * cols, hdr + (cell + pad) * len(concepts)),
                  (255, 255, 255))
d = ImageDraw.Draw(sheet)
for vi, (lab, *_) in enumerate(VARIANTS):
    d.text((lab_w + vi * a.n * (cell + pad) + 4, 8), lab, fill=(0, 0, 0))
for r, (j, c) in enumerate(concepts):
    y = hdr + r * (cell + pad)
    d.text((6, y + cell // 2), c["concept_id"].replace("cifc_", ""), fill=(0, 0, 0))
    for vi in range(len(VARIANTS)):
        for i in range(a.n):
            sheet.paste(cells[(vi, j, i)], (lab_w + (vi * a.n + i) * (cell + pad), y))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
sheet.save(a.out, quality=92)
print(f"COMPARE_DONE -> {a.out}", flush=True)
