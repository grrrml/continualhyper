"""Continual-learning trainer for the LoRA-hypernetwork (sequential over concepts).

Each concept is one task: prompt (CLIP pooled) -> hypernet -> LoRA -> diffusion reconstruction.
With `reg.weight == 0` this is the no-regularization baseline (catastrophic forgetting). With
`reg.weight > 0` it adds the von-Oswald hypernetwork output-regularization (arXiv:1906.00695),
two-stage (lookahead):

  Stage 1: DeltaTheta = -lookahead_lr * grad(L_recon)            (candidate step, detached)
  Stage 2: L = L_recon(Theta) + beta * mean_{t<k} || H_{Theta*}(c*_t) - H_{Theta+DeltaTheta}(c*_t) ||^2

i.e. the hypernet may move to fit the new concept, but its LoRA output for old concepts' prompts
(snapshotted at the start of the task) must stay put -> old concepts are not forgotten.

Forgetting is tracked by sampling each concept right after it is learned and re-sampling the first
concept after every task; per-task checkpoints feed the CIDM forgetting-matrix eval.

Run:  python -u -m src.train_cl --config configs/cl_unhype.yaml                 # baseline
      python -u -m src.train_cl --config configs/cl_unhype_reg.yaml             # with reg
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .common import load_config, save_hyper, set_seed
from .data import ConceptDataset, collate_fn, specs_from_config, enhance, scene_for_bg
from .injection import DEFAULT_TARGETS
from .losses import reconstruction_loss
from .manager import build_hyper
from .sampling import ddim_sample
from .sd_loader import load_sd


@torch.no_grad()
def _gen_one(bundle, manager, prompt, steps, gscale, seed, task_idx=None, mask_phrase=None,
             box=None):
    """Sample one image for `prompt`; returns [3,H,W] in [0,1] on cpu.
    `box` jest ustawiany JAWNIE (None = pelny kadr) - wczesniej probki diag dziedziczyly
    resztkowa losowa ramke z ostatniego kroku treningu."""
    manager.eval()
    manager.cond_box = box
    cond_hidden, pooled, _ = bundle.encode_text([prompt])
    uncond_hidden, _, _ = bundle.encode_text([""])
    token_mask = None
    if mask_phrase:
        from .tokens import token_span_mask
        token_mask = token_span_mask(bundle.tokenizer, [prompt], mask_phrase)
    gen = torch.Generator(device=bundle.device).manual_seed(seed)
    res = int(getattr(bundle, "default_resolution", 512))   # SDXL at 512 tiles into garbage
    img = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                      num_inference_steps=steps, guidance_scale=gscale, batch_size=1,
                      height=res, width=res,
                      generator=gen, scheduler=bundle.dpm_scheduler,
                      task_idx=task_idx, token_mask=token_mask)[0].clamp(0, 1).cpu()
    manager.train()
    return img


@torch.no_grad()
def _sample(bundle, manager, prompt, out_dir, n, steps, gscale, seed, task_idx=None, mask_phrase=None):
    """Generate `n` images for `prompt` into out_dir/sample_{i}.png."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        img = _gen_one(bundle, manager, prompt, steps, gscale, seed + i, task_idx=task_idx,
                       mask_phrase=mask_phrase)
        save_image(img, os.path.join(out_dir, f"sample_{i:02d}.png"))


def _paste_scale(lo: float, hi: float, full_p: float) -> float:
    """Skala wklejanego obiektu; z prawdopodobienstwem full_p z trybu pelnokadrowego."""
    if full_p > 0 and float(torch.rand(1).item()) < full_p:
        return float(torch.empty(1).uniform_(0.9, 1.0).item())
    return float(torch.empty(1).uniform_(lo, hi).item())


def _reg_mse(now, targets):
    """von-Oswald output reg: mean over layers of MSE on (x_L, x_R) between the current LoRA
    `now` and the start-of-task snapshot `targets`. F.mse_loss averages over anchors+elements."""
    terms = [F.mse_loss(now[n][0], targets[n][0]) + F.mse_loss(now[n][1], targets[n][1]) for n in targets]
    return torch.stack(terms).mean()


def parse_args():
    p = argparse.ArgumentParser(description="ContinualHyper continual trainer (optional von-Oswald reg)")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--reg_weight", type=float, default=None, help="override reg.weight (von-Oswald beta)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2024)))

    resolution = int(cfg.get("resolution", 512))
    output_dir = args.output_dir or cfg.get("output_dir", "./outputs/cl")
    train = cfg.get("training", {})
    steps_per_task = int(train.get("steps_per_task", 800))
    batch_size = int(train.get("batch_size", 2))
    lr = float(train.get("lr", 1e-4))
    grad_clip = float(train.get("grad_clip", 1.0))
    log_every = int(train.get("log_every", 50))
    diag_freq = int(train.get("diagnostic_freq", 200))   # wandb diagnostic generations every N steps
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))
    reg_cfg = cfg.get("reg", {})
    reg_weight = float(args.reg_weight if args.reg_weight is not None else reg_cfg.get("weight", 0.0))
    scale_min = float((cfg.get("task_cond") or {}).get("scale_min", 0.3))
    scale_kappa = float((cfg.get("task_cond") or {}).get("scale_kappa", 1e-3))
    lookahead_lr = float(reg_cfg.get("lookahead_lr", lr))   # von-Oswald candidate-step size (default = lr)
    # Domyslnie WYLACZONE, zeby wszystkie dotychczasowe configi odtwarzaly sie bez zmian.
    ground_anchor = bool(reg_cfg.get("ground_anchor", False))
    ground_anchor_gates = bool(reg_cfg.get("ground_anchor_gates", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fp32 backbone for the baseline (no grad-scaler headaches; hyper grads stay clean).
    _mid = cfg.get("sd_model_id", "")
    _wd = cfg.get("weight_dtype", "fp32")
    if "xl" in str(_mid).lower():
        from .sd_loader import load_sdxl
        bundle = load_sdxl(model_id=_mid, device=device, dtype=_wd)
    else:
        bundle = (load_sd(model_id=_mid, device=device, dtype=_wd)
                  if _mid else load_sd(device=device, dtype=_wd))

    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    manager.train()

    specs = specs_from_config(cfg["concepts"])
    if (cfg.get("task_cond") or {}).get("ortho_tokens"):
        from .tokens import register_ortho_tokens
        reg_ids = register_ortho_tokens(bundle, len(specs),
                                        key_dim=int(cfg["task_cond"].get("key_dim", 128)))
        print(f"[CL] ortho tokens registered: {list(reg_ids)[:3]}... ({len(reg_ids)})", flush=True)
    n_tasks = len(specs)

    # Option C: learned identifier tokens ("<Vk>") -- TI-style rows in the CLIP input embedding.
    tok_cfg = cfg.get("learned_tokens", {}) or {}
    use_tokens = bool(tok_cfg.get("enabled", False))
    token_ids, emb_weight, tok_lr = {}, None, 0.0
    ortho_on = bool((cfg.get("task_cond") or {}).get("ortho_tokens"))
    # dims [0, tail_lock) of an ortho row are the TASK KEY -- never trained, or routing breaks.
    tail_lock = int((cfg.get("task_cond") or {}).get("key_dim", 128)) if ortho_on else 0
    if use_tokens:
        if ortho_on:
            token_ids = {f"<V{k + 1}>": bundle.tokenizer.convert_tokens_to_ids(f"<V{k + 1}>")
                         for k in range(len(specs))}
            print(f"[CL] learned TAIL on ortho rows: dims [{tail_lock}:768) train, "
                  f"[0:{tail_lock}) frozen (key)", flush=True)
        else:
            from .tokens import add_learned_tokens
            token_ids = add_learned_tokens(bundle, [(sp.identifier, sp.class_word) for sp in specs],
                                           init_from_class=bool(tok_cfg.get("init_from_class", True)))
        emb_weight = bundle.text_encoder.get_input_embeddings().weight
        emb_weight.requires_grad_(True)      # grads row-masked to the current task's token
        tok_lr = float(tok_cfg.get("lr", 5e-3))
        print(f"[CL] learned tokens ON: {sorted(token_ids)} (tok_lr={tok_lr})", flush=True)

    box_aug_p = float(cfg.get("training", {}).get("box_aug_p", 0.5))
    # Protokol CIFC (data/CIFC/options/cidm/task_*.yml) trenuje wszystkie metody
    # porownawcze z EnhanceText. Domyslnie wylaczone, zeby stare configi sie odtwarzaly.
    # Skala wklejanego obiektu jako frakcja dluzszego wymiaru kadru. Domyslnie jak dotad.
    _ps = cfg.get("training", {}).get("paste_scale", [0.45, 0.85])
    paste_lo, paste_hi = float(_ps[0]), float(_ps[1])
    # Prawdopodobienstwo pobrania skali z drugiego trybu przy pelnym kadrze: bez niego
    # kompozyty nigdy nie pokazuja obiektu wypelniajacego kadr, a metryki calo-obrazowe
    # wlasnie to nagradzaja.
    paste_full_p = float(cfg.get("training", {}).get("paste_scale_full_p", 0.0))
    prompt_aug = bool(cfg.get("training", {}).get("prompt_aug", False))
    paste_caption = bool(cfg.get("training", {}).get("paste_caption", False))
    # segmentowana wklejka (objective v2): caly wyciety obiekt (RGBA) na naturalne tlo,
    # strata BEZ maski -> ramka jest informacyjnie konieczna przy wysokim szumie
    # `alpha_erode`: gradient alfy w wycinkach (isnet + feather ~10 px) wmieszuje kolor tla
    # ZE ZDJECIA ZRODLOWEGO w brzeg obiektu, wiec kazda wklejka nosi poswiate po sylwetce.
    # Adapter uczy sie jej jako czesci konceptu i przy wlaczonym groundingu obiekty dostaja
    # widoczne OBWODKI (zmierzone 2026-08-31: rant obecny przy ramce, nieobecny bez niej).
    # Min-pooling cofa brzeg do wnetrza obiektu, wiec skrajne piksele maja kolor OBIEKTU.
    # 0 = zachowanie dotychczasowe (stare configi bez zmian).
    alpha_erode = int(cfg.get("training", {}).get("alpha_erode", 0))
    seg_dir = cfg.get("training", {}).get("seg_dir")
    bg_dir = cfg.get("training", {}).get("bg_dir")
    seg_bank, bg_bank, bg_scenes = {}, [], []
    if seg_dir and bg_dir:
        import glob as _glob
        from PIL import Image as _PIL
        import numpy as _np
        res0 = int(cfg.get("resolution", 512))
        for f in sorted(_glob.glob(os.path.join(bg_dir, "*"))):
            im = _PIL.open(f).convert("RGB").resize((res0, res0), _PIL.LANCZOS)
            t = torch.from_numpy(_np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1.0
            bg_bank.append(t)
            bg_scenes.append(scene_for_bg(f))
        for spec_ in specs:
            fs = sorted(_glob.glob(os.path.join(seg_dir, spec_.concept_id, "*.png")))
            cuts = []
            for f in fs:
                im = _PIL.open(f).convert("RGBA")
                t = torch.from_numpy(_np.asarray(im)).permute(2, 0, 1).float()
                cuts.append((t[:3] / 127.5 - 1.0, t[3:4] / 255.0))   # (rgb[-1,1], alfa[0,1])
            if cuts:
                seg_bank[spec_.concept_id] = cuts
        print(f"[CL] seg-paste ON: tla {len(bg_bank)}, wycinki "
              f"{ {k: len(v) for k, v in seg_bank.items()} }", flush=True)
    if getattr(manager, "ground_cond", False):
        from .regional import set_grounded
        ng = set_grounded(bundle.unet, manager)
        print(f"[CL] grounded attention na {ng} warstwach attn2", flush=True)
    tm_enabled = bool(cfg.get("token_mask_lora", False))
    if tm_enabled:
        from .tokens import token_span_mask
        print("[CL] token-masked LoRA ON (delta only at concept-token positions)", flush=True)

    def _tok_extra():
        if not use_tokens:
            return None
        with torch.no_grad():
            return {"learned_tokens": {t: emb_weight[i].detach().cpu().clone()
                                       for t, i in token_ids.items()}}
    inf = cfg.get("infer", {})
    gsteps = int(inf.get("steps", 50)); gscale = float(inf.get("guidance_scale", 7.5))
    n_eval = int(inf.get("n_images", 4))
    seed0 = int(cfg.get("seed", 2024))
    unet = bundle.unet

    regmsg = (f"von-Oswald reg beta={reg_weight} (lookahead_lr={lookahead_lr})" if reg_weight > 0 else "NO reg")
    tcmsg = ("task_cond ON (learned V_t + Gram-Schmidt ortho)" if manager.task_cond_enabled else "task_cond OFF")
    print(f"[CL] {n_tasks} tasks (sequential, {regmsg}, {tcmsg}) | {steps_per_task} steps/task"
          f" | bs={batch_size} | lr={lr}", flush=True)
    print("[CL] task order: " + ", ".join(f"{i}:{s.concept_id}('{s.diag_prompt}')" for i, s in enumerate(specs)),
          flush=True)

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            run_name = wandb_cfg.get("name")
            if reg_weight > 0:
                run_name = (run_name or "cl") + f"_b{reg_weight}"
            _wandb.init(project=wandb_cfg.get("project", "ContinualHyper"), name=run_name,
                        config={**cfg, "reg_weight": reg_weight, "lookahead_lr": lookahead_lr})
            wandb = _wandb
        except Exception as e:
            print(f"[CL] wandb disabled (init failed: {e})", flush=True)

    named = list(manager.heads.named_parameters())       # (name, param), stable across tasks
    params = [p for _, p in named]
    anchor_conds = []   # hyper conditioning of each learned concept's canonical prompt (reg anchors)
    gstep = 0
    _clip_img = None                      # lazy: only built when sem_dim is on
    for k, spec in enumerate(specs):
        # Network weights PERSIST across tasks; fresh optimizer per task (clean per-task LR).
        # With task_cond: the CURRENT task's embedding V_k trains too (old V_i stay frozen).
        task_params = manager.task_parameters(k)
        cur_tid = token_ids.get(spec.identifier) if use_tokens else None
        opt_groups = [{"params": manager.hyper_parameters() + task_params}]
        if cur_tid is not None:
            # whole embedding matrix in the group (grads are row-masked); TI-style separate LR
            opt_groups.append({"params": [emb_weight], "lr": tok_lr, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(opt_groups, lr=lr,
                                      weight_decay=float(train.get("weight_decay", 0.0)))
        # task conditioning source (constant per task): the canonical prompt's pooled embedding,
        # or -- ablation `task_cond.key_prompt: identifier` -- just the identifier ("V1"): the key
        # only has to be fixed and GS-separable; generation prompts stay the full canonical text.
        with torch.no_grad():
            tc_cfg = cfg.get("task_cond", {}) or {}
            mode = str(tc_cfg.get("key_prompt", "canonical"))
            # "index": synthetic distinct key per task -- REQUIRED when prompts carry no
            # identifier (canonical prompts of same-class tasks are then identical text).
            key_text = {"identifier": spec.identifier or f"V{k + 1}",
                        "index": f"V{k + 1}"}.get(mode, spec.diag_prompt)
            _, pooled_canon, _ = bundle.encode_text([key_text])
            sem_vec = None
            if getattr(manager, "sem_dim", 0):
                # semantic block = mean CLIP IMAGE embedding of this concept's reference photos.
                # Images, not text: two same-class concepts have identical prompts here (the
                # identifier is empty), so text carries no instance information at all.
                import glob as _glob
                paths = sorted(_glob.glob(os.path.join(spec.images_dir, "*")))
                if not paths:
                    raise RuntimeError(f"sem_dim set but no images for {spec.concept_id}")
                if _clip_img is None:
                    from .cifc_metrics import _Clip
                    _clip_img = _Clip(device)
                sem_vec = _clip_img.img_feats(paths).mean(0).cpu()
                print(f"[CL] semantic key from {len(paths)} images of {spec.concept_id}", flush=True)
            manager.set_canonical(k, pooled_canon[0], sem_vec=sem_vec)
        loader = DataLoader(ConceptDataset(spec, resolution, augment=bool(train.get("augment", False))),
                            batch_size=batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_fn,
                            num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)
        # Captiony ida do treningu DOKLADNIE w tej formie. Drukujemy je, bo cichy blad
        # w podmianie (attr_strip, class_word) nie ma zadnego innego objawu niz atrybut,
        # ktorego adapter sie nie uczy: kolor kaczki szedl przez prompt, nie przez hipersiec.
        _ds = loader.dataset
        _caps = sorted({_ds._caption(os.path.splitext(os.path.basename(q))[0]) for q in _ds.paths})
        print(f"[CL] captiony {spec.concept_id}: " + " | ".join(_caps), flush=True)

        # von-Oswald: snapshot the hypernet output on OLD concepts at the START of this task (Theta*)
        targets, anchors = None, None
        ground_targets, ground_ids = None, list(range(len(anchor_conds)))
        gate_targets = None
        if reg_weight > 0 and anchor_conds:
            anchors = torch.stack(anchor_conds, 0)           # [k, clip_size], already conditioned
            with torch.no_grad():
                targets = {n: (a.detach(), b.detach())
                           for n, (a, b) in manager.generate_lora(anchors).items()}
                if ground_anchor:
                    gt = manager.generate_ground(ground_ids)
                    ground_targets = None if gt is None else [t.detach() for t in gt]
                if ground_anchor_gates and getattr(manager, 'ground_gates', None) is not None:
                    gate_targets = {n: p.detach().clone()
                                    for n, p in manager.ground_gates.named_parameters()}

        for step in range(steps_per_task):
            batch = next(data_iter)
            images = batch["pixel_values"].to(device)
            captions = batch["captions"]
            bsz = images.shape[0]

            # `box_cond`: with prob box_aug_p shrink the photo and paste it at a random spot;
            # the box is KNOWN exactly (that's the whole point of paste-supervision), and the
            # diffusion loss is masked to the box so the delta learns nothing about the filler.
            loss_mask = None
            if getattr(manager, "box_cond", False) or getattr(manager, "ground_cond", False):
                manager.cond_box = None
                cuts = seg_bank.get(spec.concept_id)
                if cuts and torch.rand(1).item() < box_aug_p:
                    # OBJECTIVE v2: caly obiekt (miekka alfa) na losowym naturalnym tle,
                    # strata na CALYM obrazie -- poza ramka nadzorem jest tlo, wiec przy
                    # wysokim szumie ramka to jedyne zrodlo pozycji
                    import torch.nn.functional as Fnn
                    H = images.shape[-1]
                    comp = []
                    box = None
                    for bi in range(bsz):
                        rgb, al = cuts[int(torch.randint(0, len(cuts), (1,)).item())]
                        ch, cw = rgb.shape[-2:]
                        sc = _paste_scale(paste_lo, paste_hi, paste_full_p)
                        r = sc * H / max(ch, cw)
                        nh, nw = max(8, int(ch * r)), max(8, int(cw * r))
                        rgb = Fnn.interpolate(rgb[None], size=(nh, nw), mode="bilinear",
                                              align_corners=False)[0]
                        al = Fnn.interpolate(al[None], size=(nh, nw), mode="bilinear",
                                             align_corners=False)[0].clamp(0, 1)
                        if alpha_erode > 0:
                            # UWAGA na nazwe: `k` to w tej petli INDEKS TASKA. Uzycie go na
                            # rozmiar jadra nadpisywalo task numerem 2*erode+1 (job 21744547
                            # padl na "canonical conditioning for task 7 not set"); gdyby
                            # kolizja trafila po tasku 7, trening uczylby cicho zlych konceptow.
                            ksz = 2 * alpha_erode + 1
                            al = -Fnn.max_pool2d(-al[None], ksz, stride=1,
                                                 padding=alpha_erode)[0]
                        if box is None:      # JEDNA ramka na krok (set_ground jest per krok)
                            x0 = int(torch.randint(0, H - nw + 1, (1,)).item())
                            y0 = int(torch.randint(0, H - nh + 1, (1,)).item())
                            box = ((x0 + nw / 2) / H, (y0 + nh / 2) / H, nw / H, nh / H)
                        else:                # pozycja wspolna, wymiar moze sie minimalnie rozjechac
                            x0 = min(max(int(box[0] * H - nw / 2), 0), H - nw)
                            y0 = min(max(int(box[1] * H - nh / 2), 0), H - nh)
                        bgi = int(torch.randint(0, len(bg_bank), (1,)).item())
                        if paste_caption and bg_scenes[bgi]:
                            # podpis opisuje tlo, na ktorym obiekt NAPRAWDE stoi, w formie
                            # promptow ewaluacyjnych ("A <TOK>, in the forest")
                            captions[bi] = f"{spec.replacement}, {bg_scenes[bgi]}"
                        bg = bg_bank[bgi].clone()
                        reg = bg[:, y0:y0 + nh, x0:x0 + nw]
                        bg[:, y0:y0 + nh, x0:x0 + nw] = reg * (1 - al) + rgb * al
                        comp.append(bg)
                    images = torch.stack(comp).to(device)
                    manager.cond_box = box
                elif not cuts and torch.rand(1).item() < box_aug_p and not (seg_dir and bg_dir):
                    # stara wklejka prostokatna (objective v1) - tylko gdy seg-paste wylaczone
                    import torch.nn.functional as Fnn
                    sc = _paste_scale(paste_lo, paste_hi, paste_full_p)
                    H = images.shape[-1]; hw = max(8, int(round(H * sc)) // 8 * 8)
                    small = Fnn.interpolate(images, size=(hw, hw), mode="bilinear",
                                            align_corners=False)
                    x0 = int(torch.randint(0, H - hw + 1, (1,)).item())
                    y0 = int(torch.randint(0, H - hw + 1, (1,)).item())
                    canvas = torch.zeros_like(images)          # szare tlo w [-1,1]
                    canvas[:, :, y0:y0 + hw, x0:x0 + hw] = small
                    images = canvas
                    manager.cond_box = ((x0 + hw / 2) / H, (y0 + hw / 2) / H, hw / H, hw / H)
                    lm = torch.zeros(1, 1, H // 8, H // 8, device=device)
                    lm[:, :, y0 // 8:(y0 + hw) // 8, x0 // 8:(x0 + hw) // 8] = 1.0
                    loss_mask = lm
            cond_hidden, pooled, _ = bundle.encode_text(captions, train_tokens=cur_tid is not None)
            z0 = bundle.encode_images(images)
            noise = torch.randn_like(z0)
            t = torch.randint(0, bundle.num_train_timesteps, (bsz,), device=device)
            if manager.cond_box is not None and float(cfg.get("training", {}).get("box_t_min_frac", 0.0)) > 0:
                # na krokach z ramka: wysoki szum - tam z_t nie zdradza pozycji i ramka
                # jest informacyjnie konieczna (sygnal placementu byl rozcienczany do ~20%)
                lo = int(float(cfg["training"]["box_t_min_frac"]) * bundle.num_train_timesteps)
                t = torch.randint(lo, bundle.num_train_timesteps, (bsz,), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t)

            if prompt_aug:
                captions = [enhance(c) for c in captions]
            tok_mask = (token_span_mask(bundle.tokenizer, captions, spec.replacement).to(device)
                        if tm_enabled else None)
            # `scale_cond`: s is sampled per step and fed to the head instead of multiplying the
            # delta. The s-weighted objective below is what gives s a meaning -- without it the
            # head would ignore the input, since reconstruction alone always prefers max identity.
            if getattr(manager, "latent_cond", False):
                manager.cond_latent = manager.latent_stats(z_t).detach()
            if getattr(manager, "time_cond", False):
                # per-sample, not the batch mean: condition() broadcasts h[1,D] + emb[B,D],
                # so each image gets the adapter belonging to ITS noise level
                manager.cond_t = (t.float() / bundle.num_train_timesteps).to(device)
            s_cur = 1.0
            if getattr(manager, "scale_cond", False):
                s_cur = float(torch.empty(1).uniform_(scale_min, 1.0).item())
                manager.cond_scale_val = s_cur
            if getattr(manager, "ground_cond", False):
                manager.set_ground(k, manager.cond_box)
            manager.set_context(pooled, task_idx=k, token_mask=tok_mask)  # timestep-independent LoRA
            manager.compute_and_cache_loras()
            manager.enable_lora()
            ac = bundle.added_cond(z_t.shape[0], resolution, resolution, pooled=pooled) \
                if getattr(bundle, "is_sdxl", False) else None
            eps_pred = unet(z_t, t, encoder_hidden_states=cond_hidden,
                            added_cond_kwargs=ac).sample
            if loss_mask is not None:
                d2 = (eps_pred.float() - noise.float()) ** 2 * loss_mask
                loss = d2.sum() / (loss_mask.sum() * eps_pred.shape[0] * eps_pred.shape[1]).clamp_min(1.0)
            else:
                loss = reconstruction_loss(eps_pred.float(), noise.float())   # new-concept reconstruction
            if s_cur < 1.0:
                # BUDGET, not output interpolation. A convex mix of eps-targets is equivalent to
                # regressing s*eps + (1-s)*eps_base, which to first order is exactly what scaling
                # the adapter by s already gives -- so it could not beat the status quo. Penalising
                # ||dW|| instead forces the head to CHOOSE where to spend a shrinking budget, which
                # uniform scaling cannot do.
                dw = torch.stack([ (xl.float()**2).sum() + (xr.float()**2).sum()
                                   for xl, xr in (manager.get_cached_lora(n) for n in manager.layer_names) ]).sum()
                loss = loss + scale_kappa * (1.0 - s_cur) * dw / len(manager.layer_names)

            optimizer.zero_grad(set_to_none=True)
            reg_val, reg_val_g = 0.0, 0.0
            emb_params = [emb_weight] if cur_tid is not None else []
            # The prompt-modulation net is NOT part of `params` (which is heads-only) but still
            # needs a gradient: the manual autograd.grad path below bypasses .backward(), so
            # anything omitted here would sit in the optimizer and never move.
            mod_params = ([] if getattr(manager, 'ground_head', None) is None else
                          list(manager.ground_head.parameters()) + list(manager.ground_gates.parameters())
                          + (list(manager.ground_pos_proj.parameters()) + list(manager.ground_box_proj.parameters())
                             if getattr(manager, 'ground_pos_proj', None) is not None else [])
                          + ([manager.ground_geo_a, manager.ground_geo_b]
                             if getattr(manager, 'ground_geo_a', None) is not None else [])
                          + (list(manager.ground_gsa_mods.parameters()) + list(manager.ground_film.parameters())
                             if getattr(manager, 'ground_gsa_mods', None) is not None else [])) \
                         + ([] if getattr(manager, 'box_emb', None) is None else list(manager.box_emb.parameters())) \
                         + ([] if manager.prompt_mod is None else list(manager.prompt_mod.parameters())) \
                         + ([] if manager.prompt_gate is None else list(manager.prompt_gate.parameters())) \
                         + ([] if getattr(manager, 'scale_emb', None) is None else list(manager.scale_emb.parameters())) \
                         + ([] if getattr(manager, 'time_emb', None) is None else list(manager.time_emb.parameters())) \
                         + ([] if getattr(manager, 'latent_emb', None) is None else list(manager.latent_emb.parameters()))
            if targets is not None:
                # Stage 1: candidate step that minimizes ONLY the new-task loss (detached).
                # allow_unused: z ground_boxonly galaz milczy na krokach bez ramki, wiec jej
                # parametry nie sa w grafie i bez tego autograd rzuca wyjatkiem (job 21827099).
                g_all = torch.autograd.grad(loss, params + task_params + emb_params + mod_params,
                                            retain_graph=False, allow_unused=True)
                g = g_all[:len(params)]
                g_task = g_all[len(params):len(params) + len(task_params)]
                g_mod = g_all[len(g_all) - len(mod_params):] if mod_params else []
                delta = {nm: (-lookahead_lr * gi).detach() for (nm, _), gi in zip(named, g)}
                # Stage 2: anchor the hypernet output at the lookahead params Theta + DeltaTheta.
                perturbed = {nm: p + delta[nm] for nm, p in named}
                reg = reg_weight * _reg_mse(manager.lora_from_params(anchors, perturbed), targets)
                g_reg = torch.autograd.grad(reg, params)
                g_reg_g = None
                if ground_targets is not None:
                    cur_g = manager.generate_ground(ground_ids)
                    reg_g = reg_weight * sum(F.mse_loss(c, t)
                                             for c, t in zip(cur_g, ground_targets))
                    # allow_unused: bramki, projekcje odczytu GSA i FiLM warunkowane ramka
                    # nie wchodza w ten czlon, wiec ich gradient jest None.
                    g_reg_g = torch.autograd.grad(reg_g, mod_params, allow_unused=True)
                    reg_val_g = float(reg_g.item())
                if gate_targets is not None:
                    reg_gate = reg_weight * sum(
                        F.mse_loss(p, gate_targets[n])
                        for n, p in manager.ground_gates.named_parameters())
                    g_gate = torch.autograd.grad(reg_gate, mod_params, allow_unused=True)
                    g_reg_g = g_gate if g_reg_g is None else tuple(
                        a if b_ is None else (b_ if a is None else a + b_)
                        for a, b_ in zip(g_reg_g, g_gate))
                    reg_val_g += float(reg_gate.item())
                for p, gi, gr in zip(params, g, g_reg):    # heads: task grad + reg grad
                    p.grad = gi + gr
                for p, gi in zip(task_params, g_task):     # V_k: task grad only (anchors are frozen)
                    p.grad = gi
                for i_mp, (p, gi) in enumerate(zip(mod_params, g_mod)):
                    gg = None if g_reg_g is None else g_reg_g[i_mp]
                    if gi is None and gg is None:
                        continue                      # parametr nieuzyty na tym kroku
                    p.grad = gg if gi is None else (gi if gg is None else gi + gg)
                if emb_params:
                    g_emb = g_all[len(params) + len(task_params)]
                    emb_weight.grad = torch.zeros_like(g_emb)
                    emb_weight.grad[cur_tid] = g_emb[cur_tid]   # only the current identifier row trains
                    if tail_lock:
                        emb_weight.grad[cur_tid, :tail_lock] = 0.0   # key block stays frozen
                reg_val = float(reg.item())
            else:
                loss.backward()
                if emb_params and emb_weight.grad is not None:
                    keep = emb_weight.grad[cur_tid].clone()     # only the current identifier row trains
                    if tail_lock:
                        keep[:tail_lock] = 0.0                 # key block stays frozen
                    emb_weight.grad = torch.zeros_like(emb_weight.grad)
                    emb_weight.grad[cur_tid] = keep
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params + task_params + emb_params + mod_params, grad_clip)
            optimizer.step()

            gstep += 1
            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[CL] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}"
                      + (f" | reg {reg_val:.4f}" if targets is not None else "")
                      + (f" | regG {reg_val_g:.6f}"
                         if (ground_targets is not None or gate_targets is not None) else ""), flush=True)
            if wandb is not None:
                wandb.log({"loss": float(loss.item()), "reg": reg_val, "task": k, "task_step": step,
                           "lora_magnitude": float(manager.current_lora_magnitude().item())}, step=gstep)
                if diag_freq > 0 and gstep % diag_freq == 0:   # 0 -> no mid-training images
                    # diagnostic generations for ALL concepts seen so far -> forgetting visible live
                    logs = {f"diag/{specs[j].concept_id}":
                            wandb.Image(_gen_one(bundle, manager, specs[j].diag_prompt, gsteps, gscale,
                                                 seed0, task_idx=j,
                                                 mask_phrase=specs[j].replacement if tm_enabled else None),
                                        caption=specs[j].diag_prompt)
                            for j in range(k + 1)}
                    if getattr(manager, "ground_cond", False):
                        # diag_bbox: generacje ze STEROWANA ramka + ramka narysowana na obrazku;
                        # obiekt powinien chodzic za prostokatem, styl pomijamy (globalny)
                        from PIL import Image as _PIL, ImageDraw as _Draw
                        from torchvision.transforms.functional import to_pil_image as _to_pil
                        if getattr(spec, "category", None) != "style":
                            for side, bb in (("L", (0.25, 0.5, 0.5, 1.0)), ("R", (0.75, 0.5, 0.5, 1.0))):
                                t = _gen_one(bundle, manager, spec.diag_prompt, gsteps, gscale, seed0,
                                             task_idx=k,
                                             mask_phrase=spec.replacement if tm_enabled else None, box=bb)
                                pil = _to_pil(t); W, H = pil.size
                                cx, cy, bw, bh = bb
                                _Draw.Draw(pil).rectangle(
                                    [(cx - bw / 2) * W, (cy - bh / 2) * H,
                                     (cx + bw / 2) * W, (cy + bh / 2) * H],
                                    outline=(255, 0, 0), width=4)
                                logs[f"diag_bbox/{spec.concept_id}_{side}"] = wandb.Image(
                                    pil, caption=f"box={bb}")
                        gv = [float(torch.tanh(p.detach()).abs().max())
                              for p in manager.ground_gates.values()]
                        gv.sort()
                        logs["ground/gate_max"] = gv[-1]
                        logs["ground/gate_median"] = gv[len(gv) // 2]
                        with torch.no_grad():
                            manager.set_ground(k, (0.25, 0.5, 0.5, 1.0))
                            logs["ground/e_norm"] = float(manager._ground_vec.norm())
                            if getattr(manager, "ground_geo", False):
                                gm = torch.sigmoid(manager.geo_logit(
                                    32, 32, manager._ground_vec.device, torch.float32))
                                logs["ground/geo_gate_map_L"] = wandb.Image(
                                    _to_pil(gm.reshape(1, 32, 32).cpu()),
                                    caption="sigmoid(geo_logit) dla ramki L - powinna zbiegac do maski boxa")
                    wandb.log(logs, step=gstep)

        # just-learned concept (fresh)
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "fresh", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale,
                seed0, task_idx=k, mask_phrase=spec.replacement if tm_enabled else None)
        # forgetting curve: re-sample the FIRST concept after every task
        _sample(bundle, manager, specs[0].diag_prompt,
                os.path.join(output_dir, "forgetting", f"after_task{k:02d}"), n_eval, gsteps, gscale,
                seed0, task_idx=0, mask_phrase=specs[0].replacement if tm_enabled else None)
        # freeze this task's ortho-basis vector z_k BEFORE the checkpoint (ckpt carries the basis),
        # then record the anchor (the task's constant conditioning) for future reg.
        with torch.no_grad():
            manager.freeze_task_basis(k)
            if reg_weight > 0:
                _, pooled_k, _ = bundle.encode_text([spec.diag_prompt])
                if getattr(manager, "time_cond", False):
                    manager.cond_t = 0.5      # anchor a fixed slice of the t-family
                if getattr(manager, "latent_cond", False):
                    manager.cond_latent = None   # anchor the latent-free conditioning
                if getattr(manager, "box_cond", False):
                    manager.cond_box = None      # anchor at the canonical full frame
                anchor_conds.append(manager.condition(pooled_k, k)[0].detach())
        # per-task checkpoint (for the CIDM/CIFC forgetting-matrix eval: each task's model state)
        save_hyper(manager, os.path.join(output_dir, "ckpts", f"hyper_after_task{k:02d}.pt"),
                   extra=_tok_extra())
        print(f"[CL] done task {k}:{spec.concept_id}", flush=True)

    # final sweep: how does EACH concept look after the whole sequence?
    for k, spec in enumerate(specs):
        _sample(bundle, manager, spec.diag_prompt,
                os.path.join(output_dir, "final", f"task{k:02d}_{spec.concept_id}"), n_eval, gsteps, gscale,
                seed0, task_idx=k, mask_phrase=spec.replacement if tm_enabled else None)

    if wandb is not None:
        # single end-of-run image panel (mid-training diagnostics can be off via diagnostic_freq: 0)
        wandb.log({f"final/{s.concept_id}":
                   wandb.Image(_gen_one(bundle, manager, s.diag_prompt, gsteps, gscale, seed0,
                                        task_idx=j,
                                        mask_phrase=s.replacement if tm_enabled else None),
                               caption=s.diag_prompt)
                   for j, s in enumerate(specs)}, step=gstep)

    save_hyper(manager, os.path.join(output_dir, "hyper.pt"), extra=_tok_extra())
    if wandb is not None:
        wandb.finish()
    print(f"[CL] DONE -> {output_dir} (fresh/, forgetting/, final/, hyper.pt)", flush=True)


if __name__ == "__main__":
    main()
