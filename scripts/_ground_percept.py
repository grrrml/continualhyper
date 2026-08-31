"""Mini-sweep per koncept: (kappa, sched) x [cwiartki + DINO] ROZBITE PER KONCEPT.
Cel: tabelka punktow pracy per koncept (poprawki 1: kappa per koncept, 2: krotszy harmonogram).
Zapisuje probki kaczki z kazdego punktu.
"""
import os, sys, glob, torch
sys.path.insert(0, ".")
import torch.nn.functional as Fn
from PIL import Image
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import set_grounded
from src.cifc_metrics import _Dino

GRID = [(2.0, 0.3), (4.0, 0.2), (4.0, 0.15), (4.0, 0.1)]
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
os.makedirs("outputs/probe/percept", exist_ok=True)

def quads(img):
    H, W = img.shape[-2:]
    return {"TL": img[:, :H//2, :W//2], "TR": img[:, :H//2, W//2:],
            "BL": img[:, H//2:, :W//2], "BR": img[:, H//2:, W//2:]}

for kap, sched in GRID:
    manager.ground_gain_base = kap; manager.ground_sched_frac = sched
    print(f"=== kappa={kap} sched={sched}", flush=True)
    for j, c in enumerate(cfg["concepts"]):
        if c["category"] == "style":
            continue
        ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
        cls = c["class_word"]; prompt = f"a photo of {cls}"
        ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([""])
        tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
        ok = tot = 0; dsum = 0.0
        for qname, box in Q.items():
            manager.cond_box = box
            for i in range(N):
                g = torch.Generator(device="cuda").manual_seed(31337 + i)
                img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=30,
                                  guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
                sims = {}
                for qn, crop in quads(img).items():
                    pil = Image.fromarray((crop.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
                    with torch.no_grad():
                        f = Fn.normalize(dino.m(dino.tf(pil).unsqueeze(0).to("cuda")), dim=-1)
                    sims[qn] = float((f @ ref.t()).item())
                ok += int(max(sims, key=sims.get) == qname); tot += 1
                pil = Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
                with torch.no_grad():
                    f = Fn.normalize(dino.m(dino.tf(pil).unsqueeze(0).to("cuda")), dim=-1)
                dsum += float((f @ ref.t()).item())
                if c["concept_id"] == "cifc_duck_toy" and i == 0:
                    pil.save(f"outputs/probe/percept/k{kap}_s{sched}_{qname}.png")
        print(f"  {c['concept_id']:<16} cwiartki {ok}/{tot} = {ok/tot:.0%} | DINO {dsum/tot:.4f}", flush=True)
print("PERCEPT_DONE", flush=True)
