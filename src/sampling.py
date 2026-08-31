"""Sampling with classifier-free guidance and the LoRA-hypernetwork.

The hypernetwork conditions on the prompt's CLIP **pooler_output**; the LoRA is timestep-
independent, so it is computed ONCE and reused for every denoising step. CFG policy: the
conditional pass runs with LoRA ON, the unconditional pass with LoRA OFF.
"""

from __future__ import annotations

from typing import Optional

import torch


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
    if ac_c:                              # uncond: zerowe pooled embeds (standard SDXL CFG)
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
    for i, t in enumerate(steps_list):
        hook = getattr(manager, "_step_hook", None)
        if hook is not None:                # composition: refresh region masks from attention
            hook(i, len(steps_list))
        if getattr(manager, "ground_cond", False):
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

        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return bundle.decode_latents(latents)


@torch.no_grad()
def compose_sample_regions(
    bundle, manager, regions, global_hidden, uncond_hidden, global_pooled,
    num_inference_steps: int = 50, guidance_scale: float = 7.5, alpha: float = 0.1,
    height: int = 512, width: int = 512, generator=None, scheduler=None,
    regional_steps: Optional[int] = None, bootstrap_steps: int = 0,
):
    """CIDM-style region noise estimation (arXiv 2410.17594 eq. 4-5) with OUR adapters.

    Per step: one shared unconditional pass, one global conditional pass, and one pass per region
    conditioned on that region's own prompt with that region's adapter -- so each region is a full
    single-concept generation. Predictions merge as
        E* = alpha * E_global + sum_u (1-alpha) * E_u * m_u
    Cost is (2 + U) UNet calls per step against 2 for a plain generation.

    `regional_steps` truncates the expensive part: after that many steps only the global pass runs
    (composition is settled early; late steps only refine texture). None = all steps.

    regions: [{'task_idx', 'hidden', 'pooled', 'token_mask', 'box'}]
    """
    from .regional import set_regional
    device, dtype = bundle.device, bundle.dtype
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    lh, lw = height // 8, width // 8
    latents = torch.randn(1, bundle.latent_channels, lh, lw, generator=generator,
                          device=device, dtype=dtype) * scheduler.init_noise_sigma

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
        flat = torch.full((1, 3, height, width), 0.5, device=device, dtype=dtype)
        z_bg = bundle.vae.encode(flat * 2 - 1).latent_dist.mean * bundle.vae.config.scaling_factor
    uh = uncond_hidden.to(device=device, dtype=dtype)
    gh = global_hidden.to(device=device, dtype=dtype)
    n_reg = len(regions) if regional_steps is None else 0

    for i, t in enumerate(scheduler.timesteps):
        inp = scheduler.scale_model_input(latents, t)
        with manager.no_lora():                                   # unconditional, shared
            eps_u = bundle.unet(inp, t, encoder_hidden_states=uh).sample
            eps_g_c = bundle.unet(inp, t, encoder_hidden_states=gh).sample
        eps_global = eps_u + guidance_scale * (eps_g_c - eps_u)

        use_regions = regional_steps is None or i < regional_steps
        if use_regions:
            merged = alpha * eps_global
            for r, m in zip(regions, masks):
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
                                    encoder_hidden_states=r["hidden"].to(device=device, dtype=dtype)).sample
                eps_r = eps_u + guidance_scale * (eps_c - eps_u)
                merged = merged + (1.0 - alpha) * eps_r * m.to(dtype)
            covered = torch.clamp(sum(masks), 0, 1).to(dtype)
            merged = merged + (1.0 - alpha) * eps_global * (1.0 - covered)   # background
            eps = merged
        else:
            eps = eps_global
        latents = scheduler.step(eps, t, latents).prev_sample

    with manager.no_lora():
        img = bundle.vae.decode(latents / bundle.vae.config.scaling_factor).sample
    return (img / 2 + 0.5).clamp(0, 1)
