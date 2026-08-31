"""GO/NO-GO ramki: generacja z cond_box = lewa/prawa polowa, BEZ masek przestrzennych runtime.
Metryka: DINO kazdej polowy obrazu wobec referencji; zgodnosc = wlasciwa polowa wygrywa.
Prog sukcesu (zarejestrowany z gory): >=80% przy pelnokadrowym DINO w szumie od F_base."""
import os, glob, argparse, torch
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.cifc_metrics import _Dino

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_box.yaml")
ap.add_argument("--ckpt",   default="outputs/phaseP/P_box/hyper.pt")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--scale", type=float, default=0.7)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--sched", type=float, default=1.0)
a = ap.parse_args()

cfg = load_config(a.config)
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, a.ckpt, map_location="cuda"); manager.eval(); manager.lora_scale = a.scale
manager.ground_gain_base = a.gain; manager.ground_sched_frac = a.sched
if getattr(manager, "ground_cond", False):
    from src.regional import set_grounded
    set_grounded(bundle.unet, manager)
dino = _Dino("cuda")

BOXES = {"lewa": (0.25, 0.5, 0.5, 1.0), "prawa": (0.75, 0.5, 0.5, 1.0)}
ok = tot = 0
for j, c in enumerate(cfg["concepts"]):
    if c["category"] == "style":
        continue                      # styl nie ma "polozenia" - pomijamy w gonogo
    ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
    cls = c["class_word"]
    prompt = f"a photo of {cls}"
    ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([""])
    tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
    for side, box in BOXES.items():
        manager.cond_box = box
        if getattr(manager, "ground_cond", False):
            manager.set_ground(j, box)
        for i in range(a.n):
            g = torch.Generator(device="cuda").manual_seed(31337 + i)
            img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=30,
                              guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
            W = img.shape[-1]
            halves = {"lewa": img[:, :, :W // 2], "prawa": img[:, :, W // 2:]}
            from PIL import Image
            import torch.nn.functional as Fn
            sim = {}
            for s2, h in halves.items():
                pil = Image.fromarray((h.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
                x = dino.tf(pil).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    f = Fn.normalize(dino.m(x), dim=-1)
                sim[s2] = float((f @ ref.t()).item())
            hit = sim[side] > sim["prawa" if side == "lewa" else "lewa"]
            ok += int(hit); tot += 1
    print(f"[{c['concept_id']}] zgodnosc dotychczas: {ok}/{tot}", flush=True)
print(f"\nGO/NO-GO: {ok}/{tot} = {ok/tot:.1%}  (prog: 80%)", flush=True)
