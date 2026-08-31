"""Audyt compose_sample_regions: geometria masek, suma wag, dzialanie confine,
oraz DEKODOWANIE pojedynczych predykcji regionalnych (czy podmiot ladu je w ramce)."""
import torch, os
from torchvision.utils import save_image
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.tokens import token_span_mask, register_ortho_tokens, apply_learned_tokens
from src.regional import set_regional

cfg = load_config("configs/phaseR/R_tail_s2024.yaml")
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
register_ortho_tokens(bundle, len(cfg["concepts"]), key_dim=128)
blob = load_hyper(manager, "outputs/phaseR/R_tail_s2024/hyper.pt", map_location="cuda")
apply_learned_tokens(bundle, blob["learned_tokens"])
manager.eval(); manager.lora_scale = 0.7

print("1) manager podpiety do UNetu:", getattr(bundle.unet, "hyper", None) is manager)
print("   lora_enabled:", manager.lora_enabled)

lh = lw = 64
def mk(box):
    m = torch.zeros(lh, lw, device="cuda")
    x0,y0,x1,y1 = box
    m[int(y0*lh):max(int(y0*lh)+1,int(round(y1*lh))), int(x0*lw):max(int(x0*lw)+1,int(round(x1*lw)))] = 1
    return m[None,None]
L,R = (0.,0.,.5,1.), (.5,0.,1.,1.)
m1,m2 = mk(L), mk(R)
print(f"2) maski: pokrycie {float(m1.mean()):.3f}/{float(m2.mean()):.3f}, "
      f"nachodzenie {float((m1*m2).sum()):.0f}, suma pokrycia {float(torch.clamp(m1+m2,0,1).mean()):.3f}")
a=0.1
w = a + (1-a)*float(m1[0,0,0,0]) + (1-a)*float(m2[0,0,0,0])
print(f"3) suma wag w pikselu lewej polowy: {w:.3f} (ma byc 1.000)")

# 4) confine: czy podmiot ladu je w ramce? generujemy JEDEN region
NEG=("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
     "worst quality, low quality")
w1 = "<V1> dog"
rp = f"a {w1} sitting on the beach"
h,pl,_ = bundle.encode_text([rp]); uh,_,_ = bundle.encode_text([NEG])
tm = token_span_mask(bundle.tokenizer, [rp], w1).cuda()
sch = bundle.ddim_scheduler; sch.set_timesteps(50, device="cuda")
os.makedirs("outputs/verify", exist_ok=True)
torch.set_grad_enabled(False)
for tag, use_confine in (("bez_confine", False), ("z_confine", True)):
    g = torch.Generator(device="cuda").manual_seed(4242)
    lat = torch.randn(1,4,lh,lw, generator=g, device="cuda", dtype=torch.float16)*sch.init_noise_sigma
    for t in sch.timesteps:
        inp = sch.scale_model_input(lat, t)
        with manager.no_lora():
            eu = bundle.unet(inp, t, encoder_hidden_states=uh).sample
        manager.set_context(pl.cuda(), task_idx=0, token_mask=tm); manager.compute_and_cache_loras()
        if use_confine:
            set_regional(bundle.unet, [(L, tm, False)], confine=True)
        ec = bundle.unet(inp, t, encoder_hidden_states=h).sample
        set_regional(bundle.unet, None)
        lat = sch.step(eu + 7.5*(ec-eu), t, lat).prev_sample
    with manager.no_lora():
        img = bundle.vae.decode(lat/bundle.vae.config.scaling_factor).sample
    img = (img/2+0.5).clamp(0,1)
    save_image(img[0], f"outputs/verify/{tag}.png")
    half = img.shape[-1]//2
    l = float(img[0,:,:,:half].std()); r = float(img[0,:,:,half:].std())
    print(f"4) {tag}: std lewa {l:.4f} | prawa {r:.4f} | stosunek {l/max(r,1e-6):.2f}")
print("DONE")
