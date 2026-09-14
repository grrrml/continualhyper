"""Kompozycja wielokonceptowa na SCENACH Z PRACY CIDM (arXiv:2410.17594, Rys. 3 i 12).

Po co osobny skrypt zamiast `_compose_probe.py` / `_compose_unp1.py`: tamte maja zaszyte
zabawkowe pary (pies+kot na plazy, ramki na pol kadru). Tutaj sceny, prompty i geometria ramek
pochodza z `assets/composition/scenes.json`, czyli z odczytu ICH zrodla TeX i wektorow w ICH
PDF-ach -- porownanie jest wiec like-for-like, a nie "podobna scena".

Przepis: rownanie 4-5 z ich pracy (region noise estimation) NASZYMI adapterami, plus bootstrap
wnetrza ramki (MultiDiffusion; w ich pracy NIEOPISANY, a konieczny -- zmierzone 2026-08-10:
2/3 probek z dwoma podmiotami wobec 0/3 bez niego). Opcjonalnie `--ground 1` dokleja NASZ
grounding GSA zaadresowany ramka regionu, czego ich rownania nie maja.

Potok jest NIEZALEZNY OD CHECKPOINTU: `--dry_run 1` buduje prompty, maski tokenowe, manifesty
i podglady ukladu bez GPU i bez wag. To jest tez sposob na sprawdzenie parowania koncept-ramka,
o ktore prosza notatki (assets/cidm_composition_notes.md, sekcja 7).

Przyklady:
  python scripts/_compose_scenes.py --dry_run 1 --out outputs/compose_scenes_dry
  python -u scripts/_compose_scenes.py --only 12.1,12.4 --n 3 --scale 0.4
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, ".")

NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")

# Staly kolor per koncept, zeby podglad ukladu dalo sie czytac miedzy scenami.
COLOR = {"V1": (214, 39, 40), "V2": (255, 127, 14), "V3": (44, 160, 44),
         "V4": (148, 103, 189), "V5": (140, 86, 75), "V6": (127, 127, 127),
         "V7": (31, 119, 180), "V8": (127, 127, 127), "V9": (227, 119, 194),
         "V10": (127, 127, 127)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="assets/composition/scenes.json")
    # Domyslnie `ground_800aug`, a NIE `X_sdxl_best`: ten drugi ma `paste_flush` i jest
    # o 1.7-2.0 IA gorszy (REPORT sek. 4). Nazwa "best" jest historyczna i mylaca.
    ap.add_argument("--config", default="configs/phaseX/X_sdxl_ground_800aug.yaml")
    ap.add_argument("--ckpt", default="outputs/sdxl/X_sdxl_ground_800aug/hyper.pt")
    ap.add_argument("--out", default="outputs/compose_scenes")
    ap.add_argument("--only", default="", help="lista id scen po przecinku, np. 12.1,12.4")
    ap.add_argument("--scene35", default="", choices=["", "v7", "v9", "literal"],
                    help="nadpisuje odczyt spornego trzeciego regionu sceny 3.5. Domyslnie v7, "
                         "bo ich wlasny panel 'Ours' rysuje tam psa zgodnego z miniatura V7; "
                         "'literal' odtwarza figure tak, jak jest wydrukowana")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=4242)
    ap.add_argument("--scale", type=float, default=0.7,
                    help="lora_scale. Punkt zrownanego TA wobec ich 80.0: 0.4 na SDXL "
                         "(ground_800aug), 0.45 na SD-1.5 (P_paper)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.5, help="ich s = 7.5")
    ap.add_argument("--alpha", type=float, default=0.1, help="ich alpha = 0.1")
    ap.add_argument("--bootstrap", type=int, default=15,
                    help="kroki z neutralnym tlem poza ramka regionu (0 = ich rownania golenkie)")
    ap.add_argument("--regional_steps", type=int, default=-1, help="-1 = wszystkie kroki")
    ap.add_argument("--ground", type=int, default=0,
                    help="1 = dolacz nasz grounding GSA zaadresowany ramka regionu")
    ap.add_argument("--res", type=int, default=0, help="0 = natywna dla backbone'u")
    ap.add_argument("--dry_run", type=int, default=0,
                    help="1 = bez GPU i bez wag: prompty, manifesty i podglady ukladu")
    return ap.parse_args()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def resolve_scene(scene, concepts, cfg_concepts, scene35):
    """Zwraca liste regionow z gotowym promptem i indeksem taska, albo None gdy scena zablokowana.

    Identyfikator bierzemy Z CONFIGU, nie ze specyfikacji: `R_tail` uczy sie z '<V1> dog',
    a `P_paper`/`X_sdxl` z golego 'dog'. Ten sam plik scen obsluguje wiec obie rodziny.
    """
    regions = []
    for r in scene["regions"]:
        v, task, cw, prompt = r["v"], r["task"], r["class_word"], r["region_prompt"]
        alt = (next((x for x in r.get("alternatives", []) if x["reading"] == scene35), None)
               if scene35 else None)
        if alt is not None:              # jawnie wybrany inny odczyt spornego regionu (3.5)
            v, task, cw, prompt = alt["v"], alt["task"], alt["class_word"], alt["region_prompt"]
        elif prompt is None:             # sporny i nierozstrzygniety -> scena odpada
            return None
        if concepts[v]["task"] != task:
            raise SystemExit(f"scena {scene['id']}: {v} ma task {task}, a slownik konceptow "
                             f"{concepts[v]['task']}")
        if task >= len(cfg_concepts):
            raise SystemExit(f"scena {scene['id']}: task {task} poza configiem "
                             f"({len(cfg_concepts)} konceptow)")
        cc = cfg_concepts[task]
        if cc["concept_id"] != concepts[v]["concept_id"]:
            raise SystemExit(f"scena {scene['id']}: task {task} to {cc['concept_id']} w configu, "
                             f"a {concepts[v]['concept_id']} w specyfikacji scen")
        ident = (cc.get("identifier") or "").strip()
        phrase = f"{ident} {cw}" if ident else cw
        if cw not in prompt:
            raise SystemExit(f"scena {scene['id']}: '{cw}' nie wystepuje w '{prompt}'")
        prompt = prompt.replace(cw, phrase, 1)              # tylko pierwsze wystapienie
        regions.append({"v": v, "task_idx": task, "class_word": cw, "phrase": phrase,
                        "prompt": prompt, "box": tuple(r["box"]),
                        "rtp_segment": r["rtp_segment"]})
    return regions


def draw_layout(scene, regions, path, side=512):
    """Nasz odpowiednik ich panelu 'Region Boxes' -- i zarazem kontrola parowania koncept-ramka."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (side, side), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, side - 1, side - 1], outline=(0, 0, 0), width=3)
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        col = COLOR.get(r["v"], (0, 0, 0))
        d.rectangle([x0 * side, y0 * side, x1 * side - 1, y1 * side - 1], outline=col, width=4)
        d.text((x0 * side + 6, y0 * side + 6), f"{r['v']} {r['class_word']}", fill=col)
    d.text((6, 6), f"{scene['id']}: {scene['itp']}", fill=(0, 0, 0))
    img.save(path)


def main():
    a = parse_args()
    spec = json.load(open(a.scenes, encoding="utf-8"))
    concepts = spec["concepts"]
    wanted = [s.strip() for s in a.only.split(",") if s.strip()]

    from src.common import load_config
    cfg = load_config(a.config)
    cfg_concepts = cfg["concepts"]

    scenes = [s for s in spec["scenes"] if not wanted or s["id"] in wanted]
    if wanted:
        missing = set(wanted) - {s["id"] for s in scenes}
        if missing:
            raise SystemExit(f"nie ma takich scen: {sorted(missing)}")

    plan = []
    for s in scenes:
        regions = resolve_scene(s, concepts, cfg_concepts, a.scene35)
        if regions is None:
            print(f"[{s['id']}] POMIJAM: {s.get('conflict', 'scena zablokowana')}", flush=True)
            continue
        plan.append((s, regions))
    if not plan:
        raise SystemExit("nic do wygenerowania")

    os.makedirs(a.out, exist_ok=True)
    commit = git_commit()
    print(f"[compose] {len(plan)} scen | config {a.config} | commit {commit}", flush=True)

    if a.dry_run:
        bundle = manager = None
    else:
        import torch
        from src.common import load_hyper
        from src.sd_loader import load_sd
        from src.manager import build_hyper
        from src.injection import DEFAULT_TARGETS
        mid = str(cfg.get("sd_model_id", ""))
        if "xl" in mid.lower():
            from src.sd_loader import load_sdxl
            bundle = load_sdxl(model_id=mid, device="cuda", dtype=torch.float16)
        else:
            bundle = (load_sd(model_id=mid, device="cuda", dtype=torch.float16) if mid
                      else load_sd(device="cuda", dtype=torch.float16))
        manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules",
                                                                   DEFAULT_TARGETS)),
                              n_tasks=len(cfg_concepts), task_cond=cfg.get("task_cond"),
                              **cfg.get("hyper", {}))
        if (cfg.get("task_cond") or {}).get("ortho_tokens"):
            from src.tokens import register_ortho_tokens
            register_ortho_tokens(bundle, len(cfg_concepts),
                                  key_dim=int(cfg["task_cond"].get("key_dim", 128)))
        blob = load_hyper(manager, a.ckpt, map_location="cuda")
        if blob.get("learned_tokens"):
            from src.tokens import apply_learned_tokens
            apply_learned_tokens(bundle, blob["learned_tokens"])
            print(f"[compose] wgrano {len(blob['learned_tokens'])} wierszy tokenow", flush=True)
        manager.eval()
        manager.lora_scale = a.scale
        if a.ground:
            if not getattr(manager, "ground_cond", False):
                raise SystemExit("--ground 1 wymaga checkpointu z ground_cond: true")
            from src.regional import set_grounded
            print(f"[compose] grounding na {set_grounded(bundle.unet, manager)} warstwach",
                  flush=True)

    # w dry_run nie ma bundle'a, wiec rozdzielczosc bierzemy z configu -- inaczej manifest
    # dla SD-1.5 klamalby, ze 1024
    res = a.res or (bundle.default_resolution if bundle is not None
                    else int(cfg.get("resolution", 1024)))
    rs = None if a.regional_steps < 0 else a.regional_steps

    for s, regions in plan:
        d = os.path.join(a.out, s["id"])
        os.makedirs(d, exist_ok=True)
        draw_layout(s, regions, os.path.join(d, "layout.png"))
        print(f"\n[{s['id']}] ITP '{s['itp']}'", flush=True)
        for r in regions:
            print(f"    {r['v']:>3s} task {r['task_idx']} box {r['box']} <- '{r['prompt']}'",
                  flush=True)

        manifest = {
            "scene": s["id"], "figure": s["figure"], "itp": s["itp"], "rtp": s["rtp"],
            "regions": [{k: r[k] for k in ("v", "task_idx", "class_word", "phrase",
                                           "prompt", "box", "rtp_segment")} for r in regions],
            "ours": {"config": a.config, "ckpt": a.ckpt, "commit": commit,
                     "backbone": str(cfg.get("sd_model_id", "")), "resolution": res,
                     "steps": a.steps, "guidance_scale": a.cfg, "alpha": a.alpha,
                     "lora_scale": a.scale, "bootstrap_steps": a.bootstrap,
                     "regional_steps": rs, "ground": bool(a.ground),
                     "scheduler": "DDIM", "negative_prompt": NEG,
                     "seeds": [a.seed0 + i for i in range(a.n)],
                     "scene35_reading": a.scene35 or None},
            "theirs": {"paper": spec["paper"],
                       "not_stated_by_authors": ["resolution", "sampler", "step count", "seed",
                                                 "negative prompt"],
                       "note": "ITP/RTP i geometria ramek sa ich; rozdzielczosc, sampler, liczba "
                               "krokow i ziarno sa nasze -- praca ich nie podaje (tekst mowi "
                               "seed 0, wydany kod ma na sztywno 2024)"},
        }
        json.dump(manifest, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        if a.dry_run:
            continue

        import torch
        from torchvision.utils import save_image
        from src.sampling import compose_sample_regions
        from src.tokens import token_span_mask
        gh, gp, _ = bundle.encode_text([s["itp"]])
        uh, up, _ = bundle.encode_text([NEG])
        regs = []
        for r in regions:
            h, pl, _ = bundle.encode_text([r["prompt"]])
            tm = token_span_mask(bundle.tokenizer, [r["prompt"]], r["phrase"]).cuda()
            if int(tm.sum()) == 0:
                raise SystemExit(f"scena {s['id']}: pusty span '{r['phrase']}' w '{r['prompt']}'")
            regs.append({"task_idx": r["task_idx"], "hidden": h, "pooled": pl, "box": r["box"],
                         "token_mask": tm if cfg.get("token_mask_lora") else None})
        for i in range(a.n):
            g = torch.Generator(device="cuda").manual_seed(a.seed0 + i)
            img = compose_sample_regions(bundle, manager, regs, gh, uh, gp,
                                         num_inference_steps=a.steps, guidance_scale=a.cfg,
                                         alpha=a.alpha, height=res, width=res, generator=g,
                                         regional_steps=rs, bootstrap_steps=a.bootstrap,
                                         uncond_pooled=up, ground=bool(a.ground))
            save_image(img[0], os.path.join(d, f"{i}.png"))
        print(f"    {a.n} obrazow -> {d}", flush=True)
    print("\nDONE", flush=True)


main()
