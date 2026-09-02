"""Generate CIDM/CIFC evaluation images for the continual-learning forgetting matrix.

For each per-task checkpoint k, generate the CIFC recontextualization eval set for every concept
j<=k (lower-triangular CIL matrix). Per concept: read its category's prompt file
(datasets/evaluation_prompts/test_<category>.txt), substitute `<TOK>`:
  * generation prompt   -> "V{j+1} <class>"   (conditions our hypernet, same as training)
  * CLIP-T candidate    -> "<class>"          (natural text, written to prompts.json)
Output layout matches CIFC evaluate.py: <out>/after_task{k}/task{j}_{cid}/{samples/*.jpg, prompts.json}.

Loads SD once; swaps the hyper checkpoint per task (load_state_dict), so it is efficient.

Run:  python -u -m src.gen_cifc --config configs/cl_unhype.yaml \
          --ckpt_dir outputs/cl_unhype/ckpts --out_root outputs/cl_unhype/cifc_eval --num_samples 4
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torchvision.utils import save_image

from .common import load_config, load_hyper
from .injection import DEFAULT_TARGETS
from .manager import build_hyper
from .sampling import ddim_sample
from .sd_loader import load_sd

EVAL_DIR = "data/CIFC/datasets/evaluation_prompts"
CAT_FILE = {"pet": "test_pet.txt", "plushy": "test_plushy.txt", "style": "test_style.txt"}
NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")


def _read_prompts(category):
    with open(os.path.join(EVAL_DIR, CAT_FILE[category])) as f:
        return [ln.strip() for ln in f if ln.strip()]


@torch.no_grad()
def gen_concept(bundle, manager, gen_repl, clipt_repl, category, out_dir, n, steps, gscale, seed,
                task_idx=None, mask_phrase=None, lora_start_frac=0.0, sample_batch=1):
    prompts = _read_prompts(category)
    sdir = os.path.join(out_dir, "samples")
    os.makedirs(sdir, exist_ok=True)
    uncond_hidden, _, _ = bundle.encode_text([NEG])
    res = int(getattr(bundle, "default_resolution", 512))     # 512 (SD-1.5) / 1024 (SDXL)
    lat_shape = (bundle.latent_channels, res // 8, res // 8)
    info, count = [], 0
    for p in prompts:
        gen_prompt = p.replace("<TOK>", gen_repl)      # generation: "<eval_prefix> V<k> <class>"
        clipt_text = p.replace("<TOK>", clipt_repl)    # CLIP-T candidate: "<eval_prefix> <class>"
        cond_hidden, pooled, _ = bundle.encode_text([gen_prompt])
        token_mask = None
        if mask_phrase:
            from .tokens import token_span_mask
            token_mask = token_span_mask(bundle.tokenizer, [gen_prompt], mask_phrase)
        done = 0
        while done < n:
            bs = min(sample_batch, n - done)
            # one generator PER IMAGE, seeded exactly as in the unbatched path -> the drawn
            # latents (and therefore the images) are identical regardless of sample_batch
            lat = torch.stack([
                torch.randn(lat_shape, generator=torch.Generator(device=bundle.device)
                            .manual_seed(seed + count + i), device=bundle.device,
                            dtype=bundle.dtype)
                for i in range(bs)])
            imgs = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                               num_inference_steps=steps, guidance_scale=gscale, batch_size=bs,
                               height=res, width=res,
                               scheduler=bundle.dpm_scheduler, task_idx=task_idx,
                               token_mask=token_mask, lora_start_frac=lora_start_frac,
                               latents=lat)
            for i in range(bs):
                save_image(imgs[i], os.path.join(sdir, f"{count}.jpg"))
                info.append({str(count): clipt_text})
                count += 1
            done += bs
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump(info, f)
    return count


def parse_args():
    p = argparse.ArgumentParser(description="CIDM/CIFC eval-image generation (forgetting matrix)")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_dir", required=True, help="dir with hyper_after_task{k:02d}.pt")
    p.add_argument("--out_root", default=None)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--final_only", action="store_true",
                   help="only the final checkpoint over all concepts (no full matrix)")
    p.add_argument("--ground_gain", type=float, default=None,
                   help="nadpisuje kappa galezi groundingu (ground_gain_base); 0 wylacza "
                        "wstrzykniecie i izoluje dryf tej galezi od zmiany rozkladu "
                        "treningowego")
    p.add_argument("--only_concepts", default=None,
                   help="comma-separated concept_ids to (re)generate; others left untouched")
    p.add_argument("--lora_scale", type=float, default=1.0,
                   help="inference-time LoRA strength (1.0 = trained strength)")
    p.add_argument("--eval_dtype", default=None, choices=["fp32", "fp16", "bf16"],
                   help="backbone dtype for generation (hyper heads stay fp32); default: config")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 matmuls (fp32 path only) -- large speedup on A100")
    p.add_argument("--sample_batch", type=int, default=10,
                   help="images generated per UNet batch (identical output, ~4x faster)")
    p.add_argument("--lora_scale_map", default=None,
                   help='per-group scales, e.g. "attn2.to_k=0.4,attn2.to_v=0.4,attn2.to_q=0.8"; '
                        'patterns as in target_modules, fallback --lora_scale')
    p.add_argument("--lora_start_frac", type=float, default=0.0,
                   help="enable LoRA only after this fraction of denoising steps")
    p.add_argument("--only_tasks", default=None,
                   help="comma-separated checkpoint indices k to generate (shard the matrix "
                        "across jobs); default: all")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_root = args.out_root or os.path.join(os.path.dirname(args.ckpt_dir.rstrip("/")), "cifc_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[gen] TF32 matmuls enabled", flush=True)
    dtype = args.eval_dtype or cfg.get("weight_dtype", "fp32")

    _mid = cfg.get("sd_model_id", "")
    if "xl" in str(_mid).lower():
        from .sd_loader import load_sdxl
        bundle = load_sdxl(model_id=_mid, device=device, dtype=dtype)
    else:
        bundle = (load_sd(model_id=_mid, device=device, dtype=dtype)
                  if _mid else load_sd(device=device, dtype=dtype))
    if (cfg.get("task_cond") or {}).get("ortho_tokens"):
        from .tokens import register_ortho_tokens
        register_ortho_tokens(bundle, len(cfg["concepts"]),
                              key_dim=int(cfg["task_cond"].get("key_dim", 128)))
    bl_cfg = cfg.get("baseline", {}) or {}
    svd_cfg = cfg.get("svdiff", {}) or {}
    if svd_cfg.get("enabled"):
        # SVDiff backbone (L2DM): spectral shifts modify the weights; no adapter cache involved
        from .svdiff import inject_svdiff, load_shift_state, set_scale
        from .train_l2dm import _Plain
        swapped = inject_svdiff(bundle.unet)
        if svd_cfg.get("text_encoder", True):
            swapped += inject_svdiff(bundle.text_encoder)
        manager = _Plain()
        bundle.unet.hyper = manager
        set_scale(swapped, float(args.lora_scale))     # same knob as our LoRA scale
        print(f"[gen] SVDiff: {len(swapped)} warstw, skala {args.lora_scale}", flush=True)
    elif bl_cfg:
        from .baselines import StaticLoRABank
        from .injection import inject_lora
        wrappers = inject_lora(bundle.unet, tuple(cfg.get("target_modules", DEFAULT_TARGETS)))
        for _, w in wrappers:
            w.set_parent(bundle.unet)
        method = bl_cfg.get("method", "finetune")
        manager = StaticLoRABank(wrappers, rank=int(cfg.get("hyper", {}).get("rank", 4)),
                                 per_task=method in ("clora", "lora_m", "lora_c", "lora_solo"),
                                 n_tasks=len(cfg["concepts"])).to(device)
        bundle.unet.hyper = manager
        print(f"[gen] baseline bank: {method}, {len(wrappers)} layers", flush=True)
    else:
        manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                              n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                              **cfg.get("hyper", {}))
    if getattr(manager, "ground_cond", False):
        from .regional import set_grounded
        set_grounded(bundle.unet, manager)
        if args.ground_gain is not None:
            # ground_gain_base, NIE ground_gain: sampling.py nadpisuje ground_gain na
            # KAZDYM kroku odszumiania z base i harmonogramu, wiec ustawienie samego
            # ground_gain jest no-opem (job 21766671 zwrocil wyniki identyczne do
            # czwartego miejsca wlasnie dlatego).
            manager.ground_gain_base = float(args.ground_gain)
            manager.ground_gain = float(args.ground_gain)
            print(f"[gen] ground_gain_base={manager.ground_gain_base}", flush=True)
    manager.eval()
    if getattr(manager, "scale_cond", False):
        # scale is an INPUT to the head here, not a multiplier -- multiplying as well would
        # apply the knob twice and make the sweep meaningless
        manager.cond_scale_val = float(args.lora_scale)
        manager.lora_scale = 1.0
    else:
        manager.lora_scale = float(args.lora_scale)
    if args.lora_scale_map:
        manager.lora_scale_map = [(kv.split("=")[0], float(kv.split("=")[1]))
                                  for kv in args.lora_scale_map.split(",")]
    if args.lora_scale != 1.0 or args.lora_start_frac > 0 or args.lora_scale_map:
        print(f"[gen] LoRA knobs: scale={args.lora_scale} map={args.lora_scale_map} "
              f"start_frac={args.lora_start_frac}", flush=True)

    concepts = cfg["concepts"]
    n_tasks = len(concepts)
    tok_cfg = cfg.get("learned_tokens", {}) or {}
    use_tokens = bool(tok_cfg.get("enabled", False))
    if use_tokens and not (cfg.get("task_cond") or {}).get("ortho_tokens"):
        from .tokens import add_learned_tokens
        add_learned_tokens(bundle, [(c.get("identifier", f"V{i + 1}"), c["class_word"])
                                    for i, c in enumerate(concepts)], init_from_class=False)
    task_ks = [n_tasks - 1] if args.final_only else list(range(n_tasks))
    if args.only_tasks:
        keep = {int(x) for x in args.only_tasks.split(",")}
        task_ks = [k for k in task_ks if k in keep]
    only = set(args.only_concepts.split(",")) if args.only_concepts else None

    for k in task_ks:
        ckpt = os.path.join(args.ckpt_dir,
                            f"shifts_after_task{k:02d}.pt" if svd_cfg.get("enabled")
                            else (f"bank_after_task{k:02d}.pt" if bl_cfg
                                  else f"hyper_after_task{k:02d}.pt"))
        if not os.path.exists(ckpt):
            print(f"[gen] MISSING {ckpt} — skipping task {k}", flush=True)
            continue
        if svd_cfg.get("enabled"):
            from .svdiff import load_shift_state
            blob = torch.load(os.path.join(args.ckpt_dir, f"shifts_after_task{k:02d}.pt"),
                              map_location=str(device))
            load_shift_state(swapped, blob["shifts"])
        elif bl_cfg:
            blob = torch.load(os.path.join(args.ckpt_dir, f"bank_after_task{k:02d}.pt"),
                              map_location=str(device))
            manager.load_state_dict(blob["bank"])
            manager._active = int(blob.get("active", 1))
            # LoRA-C composes the adapters of all concepts seen so far, one UNet pass each
            manager.compose_tasks = (list(range(manager._active))
                                     if bl_cfg.get("method") == "lora_c" else None)
        else:
            blob = load_hyper(manager, ckpt, map_location=str(device))
        if use_tokens and blob.get("learned_tokens"):
            from .tokens import apply_learned_tokens
            apply_learned_tokens(bundle, blob["learned_tokens"])   # rows as of THIS checkpoint
        for j in range(k + 1):                              # concept j was seen by task k
            c = concepts[j]
            if only is not None and c["concept_id"] not in only:
                continue                                    # leave already-generated cells untouched
            if bl_cfg.get("method") == "lora_solo":
                # THE control baseline: ten independently trained LoRAs, the right one picked by
                # oracle task index. Same deployment size and same task-id requirement as ours --
                # if it matches us, the hypernetwork buys nothing on this benchmark.
                manager.solo_task = j
            prefix = c.get("eval_prefix", "").strip()       # e.g. "yellow rubber" (duck), "red" (backpack)
            ident, cls, cat = c.get("identifier", f"V{j+1}"), c["class_word"], c["category"]
            gen_repl = " ".join(x for x in (prefix, ident, cls) if x)   # ident may be "" (no-id mode)
            clipt_repl = " ".join(x for x in (prefix, cls) if x)
            out_dir = os.path.join(out_root, f"after_task{k:02d}", f"task{j:02d}_{c['concept_id']}")
            nimg = gen_concept(bundle, manager, gen_repl, clipt_repl, cat, out_dir,
                               args.num_samples, args.steps, args.guidance_scale, args.seed,
                               task_idx=j,
                               mask_phrase=(" ".join(x for x in (ident, cls) if x)
                                            if cfg.get("token_mask_lora") else None),
                               lora_start_frac=float(args.lora_start_frac),
                               sample_batch=int(args.sample_batch))
            print(f"[gen] after_task{k:02d} / {c['concept_id']} ({cat}, '{gen_repl}'): {nimg} imgs",
                  flush=True)
    print(f"[gen] DONE -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
