"""Sampling with classifier-free guidance and the LoRA-hypernetwork.

The hypernetwork conditions on the prompt's CLIP **pooler_output**; the LoRA is timestep-
independent, so it is computed ONCE and reused for every denoising step. CFG policy: the
conditional pass runs with LoRA ON, the unconditional pass with LoRA OFF.
"""

from __future__ import annotations

from typing import Optional

import torch

from .regional import box_to_cxcywh


@torch.no_grad()
def ddim_sample(
    bundle,
    manager,
    cond_hidden: torch.Tensor,        # [1 or B, 77, 768] conditional context (last_hidden_state)
    uncond_hidden: torch.Tensor,      # [1 or B, 77, 768] unconditional context ("" / negative)
    clip_pooled: torch.Tensor,        # [1 or B, clip_size] pooled prompt embedding (hyper cond)
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    generator: Optional[torch.Generator] = None,
    scheduler=None,
    task_idx: Optional[int] = None,   # task-conditioning index (learned V_t + ortho basis)
    token_mask: Optional[torch.Tensor] = None,   # [1,77] LoRA application mask (concept tokens)
    lora_start_frac: float = 0.0,   # enable LoRA only after this fraction of steps (early steps
                                    # lay out the prompt's composition, late steps paint identity)
    latents: Optional[torch.Tensor] = None,   # pre-drawn latents (per-image generators); when
                                              # given, no sampling happens here
    bootstrap_steps: int = 0,          # przez pierwsze K krokow poza ramka NIE DZIALA prompt:
                                       # zewnetrze odszumiane bezwarunkowo, wnetrze z pelnym CFG
    bootstrap_bg: Optional[torch.Tensor] = None,   # ABLACJA: podstaw zewnetrzny latent tla
                                       # zamiast trybu bezwarunkowego (wymaga danych na wejsciu,
                                       # wiec nie jest to wersja docelowa - patrz komentarz nizej)
    uncond_pooled: Optional[torch.Tensor] = None,  # SDXL: pooled promptu negatywnego (patrz nizej)
) -> torch.Tensor:
    """Returns images in [0,1], shape [batch_size, 3, H, W]."""
    device, dtype = bundle.device, bundle.dtype
    unet = bundle.unet
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    lh, lw = height // 8, width // 8
    if latents is None:
        latents = torch.randn(batch_size, bundle.latent_channels, lh, lw,
                              generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    cond_seq = cond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
    ac_c = bundle.added_cond(batch_size, height, width, pooled=clip_pooled) \
        if hasattr(bundle, "added_cond") else {}
    ac_u = dict(ac_c)
    if ac_c:
        # SDXL: pipeline zeruje sekwencje I pooled RAZEM (tylko przy pustym negatywie);
        # przy podanym prompcie negatywnym oba pochodza z enkoderow. Do 2026-09-07
        # galaz uncond dostawala sekwencje NEG z zerowym pooled - kombinacje, ktorej model
        # nigdy nie widzial, a blad uncond mnozy sie w CFG przez (1 - guidance).
        # `uncond_pooled=None` zachowuje stare zachowanie (odtwarzalnosc wczesniejszych liczb).
        if uncond_pooled is not None:
            ac_u = bundle.added_cond(batch_size, height, width, pooled=uncond_pooled)
        else:
            ac_u = {**ac_c, "text_embeds": torch.zeros_like(ac_c["text_embeds"])}
    uncond_seq = uncond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    # Timestep-independent LoRA: compute ONCE from the pooled prompt, reuse every step.
    if getattr(manager, "ground_cond", False):
        manager.set_ground(task_idx, getattr(manager, "cond_box", None))
    manager.set_context(clip_pooled.to(device), task_idx=task_idx,
                        token_mask=token_mask.to(device) if token_mask is not None else None)
    manager.compute_and_cache_loras()

    time_cond = bool(getattr(manager, "time_cond", False))
    latent_cond = bool(getattr(manager, "latent_cond", False))
    n_train_t = float(getattr(bundle, "num_train_timesteps", 1000))
    steps_list = list(scheduler.timesteps)
    start_i = int(round(float(lora_start_frac) * len(steps_list)))
    compose = getattr(manager, "compose_tasks", None)   # LoRA-C: average eps over adapters
    # Bootstrap: kary w uwadze maja zmierzony pulap (tail-confine +17 pp i nasycenie), a
    # izolacja self-attention pogarsza sprawe, bo zewnetrze nie dziedziczy obiektu biernie -
    # generuje go samo z promptu. Wiec odbieramy mu prompt: przez pierwsze K krokow, kiedy
    # uklad sie rozstrzyga, ZEWNETRZE ramki jest odszumiane BEZWARUNKOWO (bez guidance), a
    # wnetrze z pelnym CFG. Nic nie pcha konceptu poza ramke, a tlo maluje potem pelny prompt
    # przez pozostale kroki. Predykcja bezwarunkowa i tak jest liczona na potrzeby CFG, wiec
    # koszt to ZERO dodatkowych przebiegow UNetu i ZERO danych wejsciowych - inferencja nie
    # moze zalezec od zewnetrznego banku tel (decyzja uzytkownika 2026-08-31).
    # `bootstrap_bg` zostaje tylko jako ABLACJA (zmierzone: zewnetrzne naturalne tlo daje
    # tlo std 0.188 vs 0.131 bez bootstrapu, zawarcie 0.88, wypelnienie 1.02).
    bs_mask = None
    if bootstrap_steps > 0 and getattr(manager, "cond_box", None) is not None:
        cx, cy, bw, bh = manager.cond_box
        # Maska TWARDA, ale ramka ROZSZERZONA o `bs_dilate` komorek latentu.
        # Dlaczego nie miekka krawedz (sprawdzone i odrzucone 2026-08-31): mieszanka
        # `latents*m + bg_t*(1-m)` sklada DWA NIEZALEZNE losowania szumu, wiec jej wariancja
        # to m^2+(1-m)^2 - przy m=0.5 polowa tego, co powinno byc na danym poziomie szumu.
        # Denoiser widzi wejscie "za malo zaszumione" i zwraca obraz o zdlawionym kontraescie:
        # tla wychodzily SZARE, a detektor przestawal znajdowac podmiot (cat: DINO 0.299,
        # brak detekcji w 7/12). Twarda maska zachowuje wariancje; rozszerzenie ramki daje
        # obiektowi margines, w ktorym moze dokonczyc nogi i glowe, zamiast zostac uciety.
        d = int(getattr(manager, "bs_dilate", 3))
        y0 = max(0, int((cy - bh / 2) * lh) - d)
        y1 = min(lh, max(y0 + 1, int(round((cy + bh / 2) * lh)) + d))
        x0 = max(0, int((cx - bw / 2) * lw) - d)
        x1 = min(lw, max(x0 + 1, int(round((cx + bw / 2) * lw)) + d))
        bs_mask = torch.zeros(1, 1, lh, lw, device=device, dtype=dtype)
        bs_mask[:, :, y0:y1, x0:x1] = 1.0
        if bootstrap_bg is not None:
            bootstrap_bg = bootstrap_bg.to(device=device, dtype=dtype)

    _MISSING = object()
    _prev_gain = getattr(manager, "ground_gain", _MISSING)
    for i, t in enumerate(steps_list):
        hook = getattr(manager, "_step_hook", None)
        if hook is not None:                # composition: refresh region masks from attention
            hook(i, len(steps_list))
        if getattr(manager, "ground_cond", False):
            # UWAGA: to MUTUJE manager.ground_gain, ktory czyta takze trening (regional
            # mnozy przez niego wstrzykniecie GSA). Bez przywrocenia na koncu diagnostyka
            # w trakcie treningu zostawialaby wartosc z ostatniego kroku samplingu.
            # harmonogram kappa (GLIGEN-style): grounding aktywny tylko przez poczatkowa
            # frakcje krokow (uklad rozstrzyga sie przy wysokim szumie); potem czysty model+LoRA
            frac = i / max(1, len(steps_list))
            base = float(getattr(manager, "ground_gain_base", 1.0))
            sched = float(getattr(manager, "ground_sched_frac", 1.0))
            manager.ground_gain = base if frac < sched else 0.0
        if i >= start_i:
            manager.enable_lora()
        else:
            manager.disable_lora()
        if latent_cond:                     # adapter depends on the generation state
            manager.cond_latent = manager.latent_stats(latents).detach()
            manager.compute_and_cache_loras()
        if time_cond:                       # adapter depends on t -> refresh the cache
            manager.cond_t = float(t) / n_train_t
            manager.compute_and_cache_loras()
        if bs_mask is not None and i < bootstrap_steps and bootstrap_bg is not None:
            noise_bg = torch.randn(bootstrap_bg.shape, generator=generator, device=device,
                                   dtype=dtype)
            bg_t = scheduler.add_noise(bootstrap_bg, noise_bg, t.reshape(1))
            latents = latents * bs_mask + bg_t.expand_as(latents) * (1 - bs_mask)
        model_input = scheduler.scale_model_input(latents, t)
        if compose:
            preds = []
            for task in compose:
                manager.solo_task = task
                preds.append(unet(model_input, t, encoder_hidden_states=cond_seq, added_cond_kwargs=ac_c or None).sample)
            manager.solo_task = None
            noise_cond = torch.stack(preds).mean(0)
        else:
            noise_cond = unet(model_input, t, encoder_hidden_states=cond_seq, added_cond_kwargs=ac_c or None).sample
        with manager.no_lora():
            noise_uncond = unet(model_input, t, encoder_hidden_states=uncond_seq, added_cond_kwargs=ac_u or None).sample

        if bs_mask is not None and i < bootstrap_steps and bootstrap_bg is None:
            # guidance TYLKO w ramce: poza nia zostaje czysta predykcja bezwarunkowa
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond) * bs_mask
        else:
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if _prev_gain is _MISSING:
        if hasattr(manager, "ground_gain"):
            del manager.ground_gain
    else:
        manager.ground_gain = _prev_gain
    return bundle.decode_latents(latents)


@torch.no_grad()
def compose_sample_regions(
    bundle, manager, regions, global_hidden, uncond_hidden, global_pooled,
    num_inference_steps: int = 50, guidance_scale: float = 7.5, alpha: float = 0.1,
    height: Optional[int] = None, width: Optional[int] = None, generator=None, scheduler=None,
    regional_steps: Optional[int] = None, bootstrap_steps: int = 0,
    uncond_pooled: Optional[torch.Tensor] = None, ground: bool = False,
):
    """CIDM-style region noise estimation (arXiv 2410.17594 eq. 4-5) with OUR adapters.

    Per step: one shared unconditional pass, one global conditional pass, and one pass per region
    conditioned on that region's own prompt with that region's adapter -- so each region is a full
    single-concept generation. Predictions merge as
        E* = alpha * E_global + sum_u (1-alpha) * E_u * m_u
    Cost is (2 + U) UNet calls per step against 2 for a plain generation.

    `regional_steps` truncates the expensive part: after that many steps only the global pass runs
    (composition is settled early; late steps only refine texture). None = all steps.

    `ground=True`: przebieg regionu dostaje NASZ grounding (GSA) zaadresowany wlasna ramka, czyli
    to, czego rownania CIDM nie maja -- podmiot formuje sie w ramce, zamiast wysrodkowany.
    Wymaga checkpointu z `ground_cond` i zainstalowanych `set_grounded`; przy `ground=False`
    grounding jest JAWNIE czyszczony, bo `_ground_vec` z poprzedniego wywolania przetrwaloby
    i region dostalby cudza ramke.

    regions: [{'task_idx', 'hidden', 'pooled', 'token_mask', 'box'}], box = (x0,y0,x1,y1) w [0,1]
    liczone od LEWEGO GORNEGO rogu, albo gesta maska. UWAGA: `manager.cond_box` uzywa innej
    konwencji, (cx,cy,w,h) -- przeliczenie jest nizej, w jednym miejscu.
    """
    device, dtype = bundle.device, bundle.dtype
    height = height or bundle.default_resolution
    width = width or bundle.default_resolution
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    lh, lw = height // 8, width // 8
    latents = torch.randn(1, bundle.latent_channels, lh, lw, generator=generator,
                          device=device, dtype=dtype) * scheduler.init_noise_sigma

    # SDXL: kazdy przebieg UNetu potrzebuje wlasnego mikro-warunkowania. Galaz uncond dostaje
    # pooled promptu negatywnego, a nie zera -- ta sama poprawka co w ddim_sample (2026-09-07);
    # bez `uncond_pooled` zostaje stare zachowanie, zeby dalo sie odtworzyc wczesniejsze liczby.
    # Na SD-1.5 `added_cond` zwraca {}, wiec caly ten blok jest tam no-opem.
    ac_g = bundle.added_cond(1, height, width, pooled=global_pooled) \
        if hasattr(bundle, "added_cond") else {}
    if ac_g:
        ac_u = bundle.added_cond(1, height, width, pooled=uncond_pooled) if uncond_pooled is not None \
            else {**ac_g, "text_embeds": torch.zeros_like(ac_g["text_embeds"])}
        ac_r = [bundle.added_cond(1, height, width, pooled=r["pooled"]) for r in regions]
    else:
        ac_u, ac_r = {}, [{} for _ in regions]

    def _mask(box):
        m = torch.zeros(lh, lw, device=device, dtype=torch.float32)
        if torch.is_tensor(box):
            m = torch.nn.functional.interpolate(box[None, None].float(), size=(lh, lw),
                                                mode="nearest")[0, 0].to(device)
        else:
            x0, y0, x1, y1 = box
            m[int(y0 * lh):max(int(y0 * lh) + 1, int(round(y1 * lh))),
              int(x0 * lw):max(int(x0 * lw) + 1, int(round(x1 * lw)))] = 1.0
        return m[None, None]

    masks = [_mask(r["box"]) for r in regions]
    z_bg = None
    if bootstrap_steps > 0:
        # MultiDiffusion-style bootstrapping: for the first K steps each region pass sees a
        # latent whose OUTSIDE is a noised flat background, so the subject has nowhere to form
        # except inside its box. Plain conditioning cannot do this (measured: a centred subject
        # forms regardless); the paper's eq. 5 needs this trick and does not mention it.
        # VAE SDXL siedzi w fp32, a `dtype` to bf16 -- kodowanie musi isc w dtype VAE,
        # inaczej wywala sie na niezgodnosci typow przy 1024.
        flat = torch.full((1, 3, height, width), 0.5, device=device, dtype=bundle.vae.dtype)
        z_bg = (bundle.vae.encode(flat * 2 - 1).latent_dist.mean
                * bundle.vae_scale_factor).to(dtype)
    uh = uncond_hidden.to(device=device, dtype=dtype)
    gh = global_hidden.to(device=device, dtype=dtype)

    _MISSING = object()
    _prev_gain = getattr(manager, "ground_gain", _MISSING)
    if not ground:
        manager.set_ground(None)          # patrz docstring: czysci ramke z poprzedniego wywolania

    for i, t in enumerate(scheduler.timesteps):
        inp = scheduler.scale_model_input(latents, t)
        if ground:
            # harmonogram kappa jak w ddim_sample: grounding zyje tylko przez poczatkowa
            # frakcje krokow, bo uklad rozstrzyga sie przy wysokim szumie
            frac = i / max(1, len(scheduler.timesteps))
            manager.ground_gain = (float(getattr(manager, "ground_gain_base", 1.0))
                                   if frac < float(getattr(manager, "ground_sched_frac", 1.0))
                                   else 0.0)
        with manager.no_lora():                                   # unconditional, shared
            eps_u = bundle.unet(inp, t, encoder_hidden_states=uh,
                                added_cond_kwargs=ac_u or None).sample
            eps_g_c = bundle.unet(inp, t, encoder_hidden_states=gh,
                                  added_cond_kwargs=ac_g or None).sample
        eps_global = eps_u + guidance_scale * (eps_g_c - eps_u)

        use_regions = regional_steps is None or i < regional_steps
        if use_regions:
            merged = alpha * eps_global
            for r, m, ac in zip(regions, masks, ac_r):
                if ground:
                    manager.set_ground(r["task_idx"], box_to_cxcywh(r["box"]))
                manager.set_context(r["pooled"].to(device), task_idx=r["task_idx"],
                                    token_mask=r["token_mask"])
                manager.compute_and_cache_loras()
                inp_r = inp
                if z_bg is not None and i < bootstrap_steps:
                    noise = torch.randn(z_bg.shape, generator=generator, device=device,
                                        dtype=dtype)
                    bg_t = scheduler.add_noise(z_bg, noise, t.reshape(1))
                    bg_t = scheduler.scale_model_input(bg_t, t)
                    mm = m.to(dtype)
                    inp_r = inp * mm + bg_t * (1 - mm)
                eps_c = bundle.unet(inp_r, t,
                                    encoder_hidden_states=r["hidden"].to(device=device, dtype=dtype),
                                    added_cond_kwargs=ac or None).sample
                eps_r = eps_u + guidance_scale * (eps_c - eps_u)
                merged = merged + (1.0 - alpha) * eps_r * m.to(dtype)
            covered = torch.clamp(sum(masks), 0, 1).to(dtype)
            merged = merged + (1.0 - alpha) * eps_global * (1.0 - covered)   # background
            eps = merged
        else:
            eps = eps_global
        latents = scheduler.step(eps, t, latents).prev_sample

    if ground:
        manager.set_ground(None)
    if _prev_gain is _MISSING:
        if hasattr(manager, "ground_gain"):
            del manager.ground_gain
    else:
        manager.ground_gain = _prev_gain
    return bundle.decode_latents(latents)        # dzieli na kawalki: VAE fp32 przy 1024 to ~1 GB/obraz


@torch.no_grad()
def compose_sample_single(
    bundle, manager, regions, global_hidden, uncond_hidden, global_pooled,
    num_inference_steps: int = 50, guidance_scale: float = 7.5,
    height: Optional[int] = None, width: Optional[int] = None, generator=None, scheduler=None,
    uncond_pooled: Optional[torch.Tensor] = None, ground: bool = False,
):
    """JEDNO przejscie UNetu na krok. Koszt NIEZALEZNY od liczby komponowanych konceptow --
    to jest ta wlasciwosc, ktora niesie teze pracy; `compose_sample_regions` (ich rown. 4-5)
    potrzebuje 2+U przebiegow i jest tu punktem odniesienia, nie wersja docelowa.

    Protokol wejscia jest DOKLADNIE ich: ITP jako prompt globalny, kazdy RTP zakodowany
    OSOBNO plus ramka. Osobne kodowanie nie jest wygoda, tylko koniecznoscia -- CLIP jest
    przyczynowy, wiec w jednym scalonym prompcie drugi span niesie kontekst pierwszego
    (zmierzone w tym repo, audyt 2885915: drugi span rownoodlegly od obu wzorcow).
    Sekwencje tekstowe nie maja w UNecie ograniczenia dlugosci ani pozycji, wiec U+1 blokow
    kontekstu miesci sie w jednym przebiegu: rozne sa tylko K/V w 16 (SD-1.5) lub 70 (SDXL)
    warstwach attn2, a nie caly UNet.

    `ground=True`: kazdy region dostaje dodatkowo wstrzyk GSA zaadresowany wlasna ramka.
    Sierpniowy werdykt ("regionalna uwaga trasuje tresc, ale nie wymusza liczby podmiotow")
    dotyczyl GOLEGO maskowania uwagi; galaz groundingu powstala pozniej i jest UCZONA pchac
    mase konceptu do ramki, wiec ten wariant nie byl nigdy sprawdzony.

    OGRANICZENIE, ktore trzeba znac przy czytaniu wynikow: `RegionKVAttnProcessor` liczy `q`
    globalnie i `to_out` bez adaptera (zachowanie referencyjne Mix-of-Show), wiec w tym torze
    dziala WYLACZNIE tekstowa polowa naszej LoRA (`to_k`/`to_v`). Nasze checkpointy maja
    delty takze na `to_q`/`to_out.0`. Jesli podmioty znow beda sie zlewac, to jest pierwsza
    dzwignia do sprawdzenia, a nie dowod, ze jedno przejscie nie dziala.

    regions: [{'task_idx', 'hidden', 'token_mask', 'box'}] -- 'pooled' nie jest uzywane,
    bo hipersiec warunkuje sie tu kanonicznym kluczem konceptu, nie promptem regionu.
    """
    from .regional import set_region_kv
    device, dtype = bundle.device, bundle.dtype
    height = height or bundle.default_resolution
    width = width or bundle.default_resolution
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    lh, lw = height // 8, width // 8
    latents = torch.randn(1, bundle.latent_channels, lh, lw, generator=generator,
                          device=device, dtype=dtype) * scheduler.init_noise_sigma

    ac_g = bundle.added_cond(1, height, width, pooled=global_pooled)         if hasattr(bundle, "added_cond") else {}
    if ac_g:
        ac_u = bundle.added_cond(1, height, width, pooled=uncond_pooled)             if uncond_pooled is not None             else {**ac_g, "text_embeds": torch.zeros_like(ac_g["text_embeds"])}
    else:
        ac_u = {}

    # LoRA jest niezalezna od kroku, wiec liczymy ja RAZ na region; procesor potem tylko
    # podmienia wskaznik cache'a. Bez tego hipersiec liczylaby sie w kazdej warstwie attn2,
    # dla kazdego regionu i kazdego kroku.
    for r in regions:
        ti = r["task_idx"]
        manager.set_context(manager.canon_pooled[ti:ti + 1], task_idx=ti,
                            token_mask=r.get("token_mask"))
        manager.compute_and_cache_loras()
        r["lora"] = manager.snapshot_lora()

    gh = global_hidden.to(device=device, dtype=dtype)
    uh = uncond_hidden.to(device=device, dtype=dtype)

    _MISSING = object()
    _prev_gain = getattr(manager, "ground_gain", _MISSING)
    if not ground:
        manager.set_ground(None)          # czysci ramke z poprzedniego wywolania
    set_region_kv(bundle.unet, regions, manager, ground)
    try:
        steps = list(scheduler.timesteps)
        for i, t in enumerate(steps):
            if ground:                    # harmonogram kappa, jak w ddim_sample
                frac = i / max(1, len(steps))
                manager.ground_gain = (float(getattr(manager, "ground_gain_base", 1.0))
                                       if frac < float(getattr(manager, "ground_sched_frac", 1.0))
                                       else 0.0)
            inp = scheduler.scale_model_input(latents, t)
            eps_c = bundle.unet(inp, t, encoder_hidden_states=gh,
                                added_cond_kwargs=ac_g or None).sample
            with manager.no_lora():
                # procesor sam przepuszcza ten przebieg: czysty negatyw w calym kadrze,
                # bez regionow i bez adapterow (bramka `lora_enabled` w RegionKVAttnProcessor)
                eps_u = bundle.unet(inp, t, encoder_hidden_states=uh,
                                    added_cond_kwargs=ac_u or None).sample
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            latents = scheduler.step(eps, t, latents).prev_sample
    finally:
        set_region_kv(bundle.unet, None, manager)
        for r in regions:
            r.pop("lora", None)
        if ground:
            manager.set_ground(None)
        if _prev_gain is _MISSING:
            if hasattr(manager, "ground_gain"):
                del manager.ground_gain
        else:
            manager.ground_gain = _prev_gain
    return bundle.decode_latents(latents)
