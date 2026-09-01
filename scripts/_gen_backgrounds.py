"""Jednorazowe tla do segmentowanej wklejki groundingu: rozne naturalne sceny BEZ zwierzat
i obiektow pierwszoplanowych (baza SD-1.5, bez adapterow). ~100 obrazow @512 -> data/backgrounds.
Run: python -m scripts._gen_backgrounds
"""
import argparse, os, torch
from contextlib import contextmanager
from torchvision.utils import save_image
from src.sampling import ddim_sample
from src.sd_loader import load_sd, load_sdxl

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/backgrounds")
ap.add_argument("--size", type=int, default=512, help="rozdzielczosc; SDXL wymaga 1024")
ap.add_argument("--sdxl", action="store_true",
                help="generuj bazowym SDXL zamiast SD-1.5. Tla musza pochodzic z tego samego "
                     "backbone'u i tej samej rozdzielczosci co trening, ktory je wkleja - "
                     "upsampling 512 -> 1024 wnosi bias tekstury, a wlasnie ustalilismy, ze "
                     "brzegi i faktura wklejek maja znaczenie (obwodki, erozja alfy)")
ap.add_argument("--steps", type=int, default=30)
a = ap.parse_args()


class _NoLoRA:
    lora_enabled = False
    lora_scale = 1.0
    def get_cached_lora(self, name): return None
    def get_token_mask(self): return None
    def enable_lora(self): pass
    def disable_lora(self): pass
    @contextmanager
    def no_lora(self):
        yield
    def set_context(self, *a, **kw): pass
    def compute_and_cache_loras(self, *a, **kw): pass


SCENES = [
    "a sandy beach with gentle waves, empty",
    "a green meadow with wildflowers, empty",
    "a forest clearing with soft sunlight",
    "a quiet city street with cobblestones, empty",
    "a cozy living room interior, empty floor",
    "a wooden kitchen table by a window",
    "a park lawn with trees in the background",
    "a snowy field under an overcast sky",
    "a desert landscape with distant dunes",
    "a stone patio in a garden, empty",
    "a lakeside shore at golden hour, empty",
    "a minimalist studio with plain backdrop",
    "a rustic barn interior with hay, empty",
    "a mountain trail with rocks and grass",
    "a library room with bookshelves, empty floor",
    "an autumn park with fallen leaves, empty",
    "a brick wall alley with soft light, empty",
    "a bathroom with tiled floor, empty",
    "a wooden pier over calm water, empty",
    "a grassy hill under a blue sky, empty",
]
PER_SCENE = 5

out = a.out
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")
b = load_sdxl(device=dev, dtype="fp16") if a.sdxl else load_sd(device=dev, dtype="fp16")
side = a.size // 8
print(f"[bg] backbone {'SDXL' if a.sdxl else 'SD-1.5'} | {a.size}px | {a.steps} krokow "
      f"-> {out}", flush=True)
mgr = _NoLoRA()
neg = "person, animal, object in foreground, text, watermark"
n_total = 0
for si, scene in enumerate(SCENES):
    ctx, pooled, _ = b.encode_text([scene])
    unc, _, _ = b.encode_text([neg])
    lat = torch.stack([torch.randn((b.latent_channels, side, side),
                       generator=torch.Generator(device=dev).manual_seed(5000 + si * 100 + j),
                       device=dev, dtype=b.dtype) for j in range(PER_SCENE)])
    imgs = ddim_sample(b, mgr, ctx, unc, pooled, num_inference_steps=a.steps,
                       guidance_scale=7.5, batch_size=PER_SCENE, scheduler=b.dpm_scheduler,
                       height=a.size, width=a.size, latents=lat)
    for j in range(PER_SCENE):
        save_image(imgs[j], os.path.join(out, f"bg_{si:02d}_{j}.jpg"))
        n_total += 1
    print(f"[bg] {si + 1}/{len(SCENES)}: {scene}", flush=True)
print(f"BG_DONE {n_total} -> {out}", flush=True)
