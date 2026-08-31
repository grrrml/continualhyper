"""Test A/B groundingu: ta sama generacja z ramka lewo/prawo/brak (jeden seed, jeden ckpt).
Rozstrzyga: piksele identyczne = grounding nie strzela przy inferencji (bug okablowania);
rozne = strzela (dalej: czy steruje polozeniem). Generacja skopiowana 1:1 z _box_probe.py."""
import sys, torch
sys.path.insert(0, ".")
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask

cfg = load_config("configs/phaseP/P_ground.yaml")
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, "outputs/phaseP/P_ground/hyper.pt", map_location="cuda")
manager.eval(); manager.lora_scale = 0.7
from src.regional import set_grounded
n = set_grounded(bundle.unet, manager)
print(f"grounded procesory: {n}")

c = cfg["concepts"][0]; cls = c["class_word"]
prompt = f"a photo of {cls}"
ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([""])
tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None

imgs = {}
for tag, box in [("L", (0.25, 0.5, 0.5, 1.0)), ("R", (0.75, 0.5, 0.5, 1.0)), ("OFF", None)]:
    manager.cond_box = box
    g = torch.Generator(device="cuda").manual_seed(31337)
    img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=30,
                      guidance_scale=7.5, generator=g, task_idx=0, token_mask=tm)[0]
    imgs[tag] = img.float().cpu()
    from PIL import Image
    Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    ).save(f"outputs/probe/ab_{tag}.png")

dLR = float((imgs["L"] - imgs["R"]).abs().mean())
dLO = float((imgs["L"] - imgs["OFF"]).abs().mean())
print(f"|L-R| = {dLR:.5f}  |L-OFF| = {dLO:.5f}  (0 = grounding NIE dziala przy inferencji)")
