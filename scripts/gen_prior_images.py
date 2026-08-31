"""Generate DreamBooth prior-preservation images with the BASE model (no adapters).

L2DM: 200 class images per concept, 50 sampled per step for L_PR.
Run: python -m scripts.gen_prior_images --config configs/baseline_l2dm.yaml --n 200
"""
import argparse, os, torch
from contextlib import contextmanager
from torchvision.utils import save_image
from src.common import load_config
from src.sampling import ddim_sample
from src.sd_loader import load_sd


class _NoLoRA:                      # sampler expects a manager-like object
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="data/prior")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    a = p.parse_args()

    cfg = load_config(a.config)
    dev = torch.device("cuda")
    _mid = cfg.get("sd_model_id", "")
    if "xl" in str(_mid).lower():
        from src.sd_loader import load_sdxl
        b = load_sdxl(model_id=_mid, device=dev, dtype="bf16")
    else:
        b = load_sd(model_id=_mid, device=dev, dtype="fp16")
    mgr = _NoLoRA()
    classes = sorted({c["class_word"] for c in cfg["concepts"]})
    print(f"[prior] {len(classes)} klas x {a.n} obrazow", flush=True)
    for cls in classes:
        d = os.path.join(a.out, cls.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        if have >= a.n:
            print(f"[prior] {cls}: {have} juz jest, pomijam", flush=True); continue
        prompt = f"a photo of {cls}"
        ctx, pooled, _ = b.encode_text([prompt])
        unc, _, _ = b.encode_text([""])
        i = have
        while i < a.n:
            bs = min(a.batch, a.n - i)
            side = int(getattr(b, "default_resolution", 512)) // 8
            lat = torch.stack([torch.randn((b.latent_channels, side, side),
                               generator=torch.Generator(device=dev).manual_seed(1000 + i + j),
                               device=dev, dtype=b.dtype) for j in range(bs)])
            res = int(getattr(b, "default_resolution", 512))
            imgs = ddim_sample(b, mgr, ctx, unc, pooled, num_inference_steps=a.steps,
                               guidance_scale=7.5, batch_size=bs, scheduler=b.dpm_scheduler,
                               height=res, width=res, latents=lat)
            for j in range(bs):
                save_image(imgs[j], os.path.join(d, f"{i + j:04d}.jpg"))
            i += bs
        print(f"[prior] {cls}: {i} obrazow -> {d}", flush=True)
    print("PRIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
