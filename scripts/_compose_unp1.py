"""Kompozycja metoda CIDM (region noise estimation, U+1 przebiegow) z NASZYMI adapterami.
Kazdy region to pelna generacja jednokonceptowa -> dziedziczy nasza jakosc jednokonceptowa.
`--regional_steps` obcina drogi fragment: po tylu krokach zostaje tylko przebieg globalny."""
import os, argparse, torch
from torchvision.utils import save_image
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import compose_sample_regions
from src.regional import set_region_kv, set_regional_self
from src.tokens import token_span_mask

NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseR/R_tail_s2024.yaml")
ap.add_argument("--ckpt",   default="outputs/phaseR/R_tail_s2024/hyper.pt")
ap.add_argument("--out",    default="outputs/compose_unp1")
ap.add_argument("--scale",  type=float, default=0.7)
ap.add_argument("--n",      type=int, default=3)
ap.add_argument("--regional_steps", type=int, default=-1)   # -1 = wszystkie kroki
ap.add_argument("--kv", type=int, default=0)                # 1 = podmiana K/V, 1 przebieg
ap.add_argument("--bootstrap", type=int, default=0)         # kroki z neutralizacja tla regionu
a = ap.parse_args()

cfg = load_config(a.config)
bundle = (load_sd(model_id=cfg["sd_model_id"], device="cuda", dtype=torch.float16)
          if cfg.get("sd_model_id") else load_sd(device="cuda", dtype=torch.float16))
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
if (cfg.get("task_cond") or {}).get("ortho_tokens"):
    from src.tokens import register_ortho_tokens
    register_ortho_tokens(bundle, len(cfg["concepts"]),
                          key_dim=int(cfg["task_cond"].get("key_dim", 128)))
blob = load_hyper(manager, a.ckpt, map_location="cuda")
if blob.get("learned_tokens"):
    from src.tokens import apply_learned_tokens
    apply_learned_tokens(bundle, blob["learned_tokens"])
manager.eval(); manager.lora_scale = a.scale

C = {c["concept_id"]: (i, " ".join(x for x in (c.get("identifier", ""), c["class_word"]) if x))
     for i, c in enumerate(cfg["concepts"])}
L, R = (0.05, 0.20, 0.45, 0.90), (0.55, 0.20, 0.95, 0.90)   # ciasne, 24% kadru kazda, tlo = prompt globalny
PAIRS = [("cifc_dog", "cifc_cat", "a {a} and a {b} sitting on the beach", L, R),
         ("cifc_dog", "cifc_dog2", "a {a} and a {b} sitting on the beach", L, R)]
rs = None if a.regional_steps < 0 else a.regional_steps
os.makedirs(a.out, exist_ok=True)

for id1, id2, tmpl, b1, b2 in PAIRS:
    (j1, w1), (j2, w2) = C[id1], C[id2]
    prompt = tmpl.format(a=w1, b=w2)
    gh, gp, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([NEG])
    regs = []
    for (j, w, box) in ((j1, w1, b1), (j2, w2, b2)):
        rp = f"a {w} sitting on the beach"          # prompt WLASNY regionu (jednokonceptowy)
        h, pl, _ = bundle.encode_text([rp])
        regs.append({"task_idx": j, "hidden": h, "pooled": pl,
                     "token_mask": token_span_mask(bundle.tokenizer, [rp], w).cuda(), "box": box})
    tag = f"{id1.replace('cifc_','')}+{id2.replace('cifc_','')}"
    d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
    print(f"[{tag}] globalny: '{prompt}' | regiony: {[f'a {w} ...' for w in (w1,w2)]}", flush=True)
    for i in range(a.n):
        g = torch.Generator(device="cuda").manual_seed(4242 + i)
        if a.kv:
            sch = bundle.ddim_scheduler; sch.set_timesteps(50, device="cuda")
            lat = torch.randn(1, 4, 64, 64, generator=g, device="cuda",
                              dtype=torch.float16) * sch.init_noise_sigma
            set_region_kv(bundle.unet, regs, manager)
            set_regional_self(bundle.unet, [r["box"] for r in regs], leak=0.0)
            with torch.no_grad():
                for t in sch.timesteps:
                    inp = sch.scale_model_input(lat, t)
                    ec = bundle.unet(inp, t, encoder_hidden_states=gh.half().cuda()).sample
                    set_region_kv(bundle.unet, None, manager)
                    set_regional_self(bundle.unet, None)
                    with manager.no_lora():
                        eu = bundle.unet(inp, t, encoder_hidden_states=uh.half().cuda()).sample
                    set_region_kv(bundle.unet, regs, manager)
                    set_regional_self(bundle.unet, [r["box"] for r in regs], leak=0.0)
                    lat = sch.step(eu + 7.5 * (ec - eu), t, lat).prev_sample
                set_region_kv(bundle.unet, None, manager)
                set_regional_self(bundle.unet, None)
                img = bundle.vae.decode(lat / bundle.vae.config.scaling_factor).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        else:
            img = compose_sample_regions(bundle, manager, regs, gh, uh, gp,
                                         num_inference_steps=50, guidance_scale=7.5,
                                         generator=g, regional_steps=rs,
                                         bootstrap_steps=a.bootstrap)
        save_image(img[0], os.path.join(d, f"{i}.png"))
    print(f"   {a.n} obrazow -> {d}", flush=True)
print("DONE", flush=True)
