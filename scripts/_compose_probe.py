"""Najtansza mozliwa proba kompozycji na istniejacym F_base -- ZERO treningu.
Strona tekstowa (to_k/to_v): adapter kazdego konceptu maskowany do WLASNEGO spanu (rozlacznie).
Strona obrazowa (to_q/to_out): suma delt (brak osi tokenow, wiec nie ma czego maskowac).
Kontrole jednokonceptowe generowane tym samym seedem -> roznica jest efektem kompozycji."""
import os, sys, torch, argparse
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import (set_regional, set_regional_self, derive_masks,
                          reset_attn_acc)

NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")

def occ_mask(tokenizer, prompt, phrase, which):
    """maska [1,77] dla KTOREGO wystapienia frazy (0 = pierwsze, 1 = drugie)"""
    m = token_span_mask(tokenizer, [prompt], phrase)          # wszystkie wystapienia
    idx = (m[0] > 0).nonzero().flatten().tolist()
    if not idx: raise RuntimeError(f"brak '{phrase}' w '{prompt}'")
    groups, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            groups.append(cur)          # nowa lista, bez aliasu
            cur = [b]
    groups.append(cur)
    out = torch.zeros_like(m)
    out[0, groups[min(which, len(groups) - 1)]] = 1.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phaseF/F_base.yaml")
    ap.add_argument("--ckpt",   default="outputs/phaseF/F_base/hyper.pt")
    ap.add_argument("--out",    default="outputs/compose")
    ap.add_argument("--scale",  type=float, default=0.7)
    ap.add_argument("--n",      type=int, default=3)
    ap.add_argument("--mode",   default="full", choices=["tokenonly","full","soft"])
    ap.add_argument("--strength", type=float, default=8.0)
    ap.add_argument("--self_leak", type=float, default=-1.0,
                    help=">=0 wlacza regionalna samo-uwage z tym przeciekiem")
    ap.add_argument("--no_boxes", type=int, default=0,
                    help="1 = ZERO bboxow: uklad z modelu bazowego, maski tylko z uwagi")
    ap.add_argument("--attn_masks", type=int, default=0,
                    help="krok, od ktorego bboxy zastepujemy maskami z uwagi (0=nigdy)")
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
        print(f"[compose] wgrano {len(blob['learned_tokens'])} wierszy tokenow", flush=True)
    manager.eval(); manager.lora_scale = a.scale
    C = {c["concept_id"]: (i, " ".join(x for x in (c.get("identifier", ""), c["class_word"]) if x))
         for i, c in enumerate(cfg["concepts"])}
    print("koncepty:", {k: v[0] for k, v in C.items()}, flush=True)

    L, R, ALL = (0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    ALLBOX = ALL
    # (id1, id2, szablon, box1, box2, czy2global) -- styl dostaje slot GLOBALNY
    PAIRS = [("cifc_dog", "cifc_cat", "a {a} and a {b} sitting on the beach", L, R, False),
             ("cifc_dog", "cifc_dog2", "a {a} and a {b} sitting on the beach", L, R, False),
             ("cifc_cat", "cifc_painting", "a {a} in the style of a {b}", ALL, ALL, True)]
    os.makedirs(a.out, exist_ok=True)
    for id1, id2, tmpl, b1, b2, g2 in PAIRS:
        if id1 not in C or id2 not in C: print(f"pomijam {id1}+{id2}"); continue
        (j1, w1), (j2, w2) = C[id1], C[id2]
        prompt = tmpl.format(a=w1, b=w2)
        m1 = occ_mask(bundle.tokenizer, prompt, w1, 0)
        m2 = occ_mask(bundle.tokenizer, prompt, w2, 1 if w1 == w2 else 0)
        ov = float((m1 * m2).sum())
        print(f"\n[{id1}+{id2}] '{prompt}' | span1 {int(m1.sum())} tok, span2 {int(m2.sum())} tok, "
              f"nachodzenie {ov:.0f}", flush=True)
        ch, pooled, _ = bundle.encode_text([prompt]); uh, _, _ = bundle.encode_text([NEG])
        tag = f"{id1.replace('cifc_','')}+{id2.replace('cifc_','')}"
        for name, setup in (("compose", "multi"), (f"solo1_{id1[5:]}", j1), (f"solo2_{id2[5:]}", j2)):
            d = os.path.join(a.out, f"{a.mode}__{tag}__{name}"); os.makedirs(d, exist_ok=True)
            for i in range(a.n):
                g = torch.Generator(device="cuda").manual_seed(4242 + i)
                if setup == "multi":
                    e1 = {"task_idx": j1, "token_mask": m1.cuda(),
                          "box": None if (a.no_boxes or a.mode == "tokenonly") else b1}
                    e2 = {"task_idx": j2, "token_mask": m2.cuda(),
                          "box": None if (g2 or a.no_boxes or a.mode == "tokenonly") else b2}
                    manager.set_multi_context([e1, e2])
                    if a.mode != "tokenonly":
                        nobox = a.attn_masks and a.no_boxes
                        set_regional(bundle.unet,
                                     None if nobox else [(b1, m1, False), (b2, m2, g2)],
                                     strength=(a.strength if a.mode == "soft" else None),
                                     collect=bool(a.attn_masks) and not g2)
                        if nobox:      # zbieraj uwage bez narzucania ukladu
                            set_regional(bundle.unet, [(ALLBOX, m1, False), (ALLBOX, m2, False)],
                                         strength=0.0, collect=True)
                        if a.self_leak >= 0 and not g2 and not nobox:
                            set_regional_self(bundle.unet, [b1, b2], leak=a.self_leak)
                    if a.attn_masks and not g2:
                        reset_attn_acc()
                        state = {"on": False}

                        def _hook(step, total, _s=state):
                            # first `attn_masks` steps: rectangular boxes fix the layout while
                            # attention is still noise. After that, switch to masks derived from
                            # cross-attention -- they follow each subject's actual silhouette, so
                            # a dog spilling past its box keeps its own identity instead of the
                            # neighbour's.
                            if step < a.attn_masks:
                                return
                            dm = derive_masks(2)
                            if dm is None:
                                return
                            manager.set_multi_context(
                                [{"task_idx": j1, "token_mask": m1.cuda(), "box": dm[0]},
                                 {"task_idx": j2, "token_mask": m2.cuda(), "box": dm[1]}])
                            set_regional(bundle.unet, [(dm[0], m1, False), (dm[1], m2, g2)],
                                         strength=(a.strength if a.mode == "soft" else None),
                                         collect=True)
                            if a.self_leak >= 0:
                                set_regional_self(bundle.unet, [dm[0], dm[1]], leak=a.self_leak)
                            if not _s["on"]:
                                print(f"      krok {step}: maski z uwagi "
                                      f"(pokrycie {float(dm[0].mean()):.2f}/{float(dm[1].mean()):.2f})",
                                      flush=True)
                                _s["on"] = True
                        manager._step_hook = _hook
                    img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=50,
                                      guidance_scale=7.5, generator=g, task_idx=j1, token_mask=None)
                    set_regional(bundle.unet, None)
                    set_regional_self(bundle.unet, None)
                    manager._step_hook = None
                    manager.clear_multi()
                else:
                    manager.clear_multi()
                    mm = m1 if setup == j1 else m2
                    img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=50,
                                      guidance_scale=7.5, generator=g, task_idx=setup, token_mask=mm.cuda())
                from torchvision.utils import save_image
                save_image(img[0], os.path.join(d, f"{i}.png"))
            print(f"   {name}: {a.n} obrazow -> {d}", flush=True)
    print("\nDONE", flush=True)

main()
