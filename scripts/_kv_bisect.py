"""Bisekcja artefaktow kv3: (a) procesor bez regionow, (b) pelne polowy, (c) ciasne ramki."""
import os, torch
from torchvision.utils import save_image
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.tokens import token_span_mask, register_ortho_tokens, apply_learned_tokens
from src.regional import set_region_kv

NEG=("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
     "worst quality, low quality")
cfg = load_config("configs/phaseR/R_tail_s2024.yaml")
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
register_ortho_tokens(bundle, len(cfg["concepts"]), key_dim=128)
blob = load_hyper(manager, "outputs/phaseR/R_tail_s2024/hyper.pt", map_location="cuda")
apply_learned_tokens(bundle, blob["learned_tokens"])
manager.eval(); manager.lora_scale = 0.7
torch.set_grad_enabled(False)

prompt = "a <V1> dog and a <V7> dog sitting on the beach"
gh,_,_ = bundle.encode_text([prompt]); uh,_,_ = bundle.encode_text([NEG])
def reg(box, j, w):
    rp = f"a {w} sitting on the beach"; h,_,_ = bundle.encode_text([rp])
    return {"task_idx": j, "hidden": h, "box": box,
            "token_mask": token_span_mask(bundle.tokenizer, [rp], w).cuda()}
FULL = [reg((0.,0.,.5,1.), 0, "<V1> dog"), reg((.5,0.,1.,1.), 6, "<V7> dog")]
TIGHT= [reg((.05,.2,.45,.9), 0, "<V1> dog"), reg((.55,.2,.95,.9), 6, "<V7> dog")]
os.makedirs("outputs/kv_bisect", exist_ok=True)
for tag, regs in (("a_noregion", []), ("b_fullbox", FULL), ("c_tightbox", TIGHT)):
    g = torch.Generator(device="cuda").manual_seed(4242)
    sch = bundle.ddim_scheduler; sch.set_timesteps(50, device="cuda")
    lat = torch.randn(1,4,64,64, generator=g, device="cuda", dtype=torch.float16)*sch.init_noise_sigma
    for t in sch.timesteps:
        inp = sch.scale_model_input(lat, t)
        set_region_kv(bundle.unet, regs if regs else None, manager)
        ec = bundle.unet(inp, t, encoder_hidden_states=gh.half().cuda()).sample
        set_region_kv(bundle.unet, None, manager)
        with manager.no_lora():
            eu = bundle.unet(inp, t, encoder_hidden_states=uh.half().cuda()).sample
        lat = sch.step(eu + 7.5*(ec-eu), t, lat).prev_sample
    with manager.no_lora():
        img = bundle.vae.decode(lat/bundle.vae.config.scaling_factor).sample
    img=(img/2+0.5).clamp(0,1); save_image(img[0], f"outputs/kv_bisect/{tag}.png")
    print(f"{tag}: zapisany, std {float(img.std()):.3f}", flush=True)
print("DONE")
