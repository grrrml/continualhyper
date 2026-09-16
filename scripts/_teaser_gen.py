"""Dogeneracja obrazow do teasera: jeden koncept, wlasne prompty, wybrane skale LoRA.

Po co osobny skrypt: `gen_cifc` chodzi po dwudziestu promptach benchmarku i po wszystkich konceptach,
a `src.infer` nie wystawia skali adaptera. Tutaj: jeden `task_idx`, lista promptow, lista skal,
galaz umiejscowienia wylaczona (teaser (b) jest bez ramki), maska tokenow na slowie klasy jak
w treningu i ewaluacji.

Run:  python scripts/_teaser_gen.py --config outputs/sweep/p022/config.yaml \
          --ckpt outputs/sweep/p022/ckpts/hyper_after_task49.pt --task 45 \
          --prompts "A person wearing glasses, in the forest" "A person with glasses, on the street" \
          --scales 0.8 1.0 --n 12 --out outputs/teaser/p022_more/task45
"""
import argparse
import os
import re
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config, load_hyper          # noqa: E402
from src.injection import DEFAULT_TARGETS               # noqa: E402
from src.manager import build_hyper                     # noqa: E402
from src.sampling import ddim_sample                    # noqa: E402
from src.sd_loader import load_sd                       # noqa: E402
from src.tokens import token_span_mask                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", type=int, required=True, help="indeks zadania (0-based)")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.8])
    ap.add_argument("--n", type=int, default=12, help="obrazow na prompt")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--ground_gain", type=float, default=None,
                    help="nadpisuje kappa galezi umiejscowienia; brak = jak w configu, "
                         "czyli tak jak robi `gen_cifc` (pelna ramka, galaz czynna)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp16")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    load_hyper(manager, a.ckpt, map_location=str(device))
    manager.eval()
    if getattr(manager, "ground_cond", False):
        # Bez tego galaz umiejscowienia NIE ISTNIEJE w UNecie, a niesie czesc tozsamosci:
        # `gen_cifc` instaluje ja zawsze, takze przy generacji bez ramki (pelny kadr, maska = 1).
        from src.regional import set_grounded
        set_grounded(bundle.unet, manager)
        print("[teaser] galaz umiejscowienia zainstalowana (pelny kadr)", flush=True)
    # Bez ramki, ale galaz umiejscowienia zostaje czynna przy pelnym kadrze -- dokladnie tak
    # generuje `gen_cifc`, a galaz niesie czesc tozsamosci (wylaczenie jej kosztuje 8.2 IA
    # i 11.1 DINO). Zerowanie jej dawalo obrazy slabsze niz pasek teasera.
    if a.ground_gain is not None:
        manager.ground_gain_base = a.ground_gain
    manager.cond_box = None
    c = cfg["concepts"][a.task]
    cls = c["class_word"]
    print(f"[teaser] zadanie {a.task}: {c['concept_id']} (slowo klasy {cls!r})", flush=True)

    uncond_hidden, uncond_pooled, _ = bundle.encode_text([""])
    for sc in a.scales:
        manager.lora_scale = sc
        tag = "s" + f"{sc:.2f}".replace(".", "").rstrip("0")
        for prompt in a.prompts:
            if cls not in prompt.lower():
                print(f"[teaser] UWAGA: prompt bez slowa klasy: {prompt!r}", flush=True)
            ch, pooled, _ = bundle.encode_text([prompt])
            tm = (token_span_mask(bundle.tokenizer, [prompt], cls).to(device)
                  if cfg.get("token_mask_lora") else None)
            slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:48]
            d = os.path.join(a.out, tag, slug)
            os.makedirs(d, exist_ok=True)
            for i in range(a.n):
                g = torch.Generator(device=device).manual_seed(a.seed0 + i)
                with torch.no_grad():
                    img = ddim_sample(bundle, manager, ch, uncond_hidden, pooled,
                                      num_inference_steps=a.steps, guidance_scale=7.5,
                                      batch_size=1, generator=g, task_idx=a.task,
                                      scheduler=bundle.dpm_scheduler,   # jak `gen_cifc`: DPM++,
                                      # nie DDIM. Przy tym samym checkpointie i skali DDIM dawal
                                      # inne obrazy (manekin bez sukienki przy zadaniu 40), wiec
                                      # teaser musi isc tym samym samplerem co ewaluacja.
                                      token_mask=tm, uncond_pooled=uncond_pooled)[0]
                arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                Image.fromarray(arr).save(os.path.join(d, f"{i:02d}.jpg"), quality=92)
            print(f"[teaser] {tag} | {prompt} -> {d} ({a.n})", flush=True)
    print("TEASER_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
