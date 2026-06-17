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
) -> torch.Tensor:
    """Returns images in [0,1], shape [batch_size, 3, H, W]."""
    device, dtype = bundle.device, bundle.dtype
    unet = bundle.unet
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    lh, lw = height // 8, width // 8
    latents = torch.randn(batch_size, bundle.latent_channels, lh, lw,
                          generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    cond_seq = cond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
    uncond_seq = uncond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    # Timestep-independent LoRA: compute ONCE from the pooled prompt, reuse every step.
    manager.set_context(clip_pooled.to(device))
    manager.compute_and_cache_loras()

    for t in scheduler.timesteps:
        manager.enable_lora()
        model_input = scheduler.scale_model_input(latents, t)
        noise_cond = unet(model_input, t, encoder_hidden_states=cond_seq).sample
        with manager.no_lora():
            noise_uncond = unet(model_input, t, encoder_hidden_states=uncond_seq).sample

        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return bundle.decode_latents(latents)
