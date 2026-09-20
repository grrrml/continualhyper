"""Przerenderowanie figur z wybranych checkpointow, zadan i skal adaptera -- model ladowany RAZ.

Po co osobny skrypt, skoro jest `_teaser_gen.py`. Tamten bierze JEDEN checkpoint i JEDNO zadanie,
a tu potrzebne sa dwa przemiaty, ktore roznia sie tym, co jest petla zewnetrzna:

* siatka zapominania (fig:forgetgrid): jedno zadanie, SZESC checkpointow, TRZY skale;
* pasek teasera (fig:teaser b): jeden checkpoint, DZIESIEC zadan, jedna skala.

Szesnascie wywolan `_teaser_gen.py` to szesnascie ladowan SDXL-a. Tutaj bundle wstaje raz,
a `load_hyper` miedzy checkpointami jest darmowy.

Sciezka renderowania jest skopiowana z `_teaser_gen.py` i musi taka zostac, bo kazdy z tych
szczegolow byl tam ustalony osobno i zmiana ktoregokolwiek daje inne obrazy:

* galaz umiejscowienia INSTALOWANA takze przy pelnym kadrze -- niesie czesc tozsamosci,
  wylaczenie jej kosztowalo 8,2 IA i 11,1 DINO;
* sampler DPM++, nie DDIM -- tak generuje `gen_cifc`, a przy tym samym checkpointcie DDIM
  dawal inne obrazy;
* maska tokenow na slowie klasy, jak w treningu i ewaluacji.

Po co w ogole ten przemiat po skalach. Probki `forgetting/`, z ktorych zlozona jest obecna
Figura 6, sa zapisywane przez `train_cl.py` przy DOMYSLNEJ siłe adaptera (`manager.lora_scale`
= 1.0), a podpis figury mowi 0.45. To jest blad rzeczowy, a przy okazji prawdopodobna przyczyna
tego, co na tej figurze widac: nasza wlasna Figura 9 pokazuje, ze przy 1.5 obraz sie rozpada,
wiec 1.0 jest juz w drodze tam, dwukrotnie powyzej punktu pracy.

Run:
  # siatka zapominania, koncept 0, szesc checkpointow, trzy skale
  python scripts/_fig_rerender.py --config configs/phaseT/T50_v2.yaml \\
      --ckpt_dir outputs/sweep/p022_v2/ckpts --ckpts 0,9,19,29,39,49 --tasks 0 \\
      --scales 0.45,0.8,1.0 --n 2 --out outputs/figrender/forget_v2

  # pasek teasera, ostatni checkpoint, co piaty koncept
  python scripts/_fig_rerender.py --config configs/phaseT/T50_v2.yaml \\
      --ckpt_dir outputs/sweep/p022_v2/ckpts --ckpts 49 \\
      --tasks 0,5,10,15,20,25,30,35,40,45 --scales 0.8 --n 4 --out outputs/figrender/teaser_v2
"""
import argparse
import os
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


def ints(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--ckpts", required=True, help="indeksy zadan po przecinku, np. 0,9,19,29,39,49")
    ap.add_argument("--tasks", required=True, help="indeksy konceptow po przecinku")
    ap.add_argument("--scales", default="0.8")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp16")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    grounded = False
    uncond_hidden, uncond_pooled, _ = bundle.encode_text([""])

    for ck in ints(a.ckpts):
        path = os.path.join(a.ckpt_dir, f"hyper_after_task{ck:02d}.pt")
        if not os.path.exists(path):
            raise SystemExit(f"BLAD: brak {path}")
        load_hyper(manager, path, map_location=str(device))
        manager.eval()
        if getattr(manager, "ground_cond", False) and not grounded:
            from src.regional import set_grounded
            set_grounded(bundle.unet, manager)      # raz: patch czyta z zywego managera
            grounded = True
            print("[fig] galaz umiejscowienia zainstalowana (pelny kadr)", flush=True)
        manager.cond_box = None

        for t in ints(a.tasks):
            if t > ck:
                print(f"[fig] checkpoint {ck} nie widzial zadania {t}, pomijam", flush=True)
                continue
            c = cfg["concepts"][t]
            prompt = c.get("prompt") or f"a photo of {c['class_word']}"
            cls = c["class_word"]
            ch, pooled, _ = bundle.encode_text([prompt])
            tm = (token_span_mask(bundle.tokenizer, [prompt], cls).to(device)
                  if cfg.get("token_mask_lora") else None)
            for sc in floats(a.scales):
                manager.lora_scale = sc
                tag = "s" + f"{sc:.2f}".replace(".", "")
                d = os.path.join(a.out, f"ck{ck:02d}", f"task{t:02d}", tag)
                os.makedirs(d, exist_ok=True)
                for i in range(a.n):
                    g = torch.Generator(device=device).manual_seed(a.seed0 + i)
                    with torch.no_grad():
                        img = ddim_sample(bundle, manager, ch, uncond_hidden, pooled,
                                          num_inference_steps=a.steps, guidance_scale=7.5,
                                          batch_size=1, generator=g, task_idx=t,
                                          scheduler=bundle.dpm_scheduler,
                                          token_mask=tm, uncond_pooled=uncond_pooled)[0]
                    arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    Image.fromarray(arr).save(os.path.join(d, f"{i:02d}.jpg"), quality=92)
                print(f"[fig] ck{ck:02d} task{t:02d} {tag} -> {d} ({a.n})", flush=True)

    print("FIG_RERENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
