"""Load a frozen Stable Diffusion 1.5 backbone via `diffusers` and expose its parts.

Design notes
------------
* **Self-sufficient.** We use `diffusers` (not the CompVis `ldm` loader) so this folder
  depends only on pip packages. This is a deliberate deviation from the original spec,
  approved to make the project standalone.
* **CLIP text encoder.** SD-1.5's own `CLIPTextModel` (ViT-L/14, hidden=768) produces both
  (a) the `last_hidden_state` sequence used as the UNet cross-attention context and
  (b) the `pooler_output` (one 768-d vector) that conditions the LoRA-hypernetwork.
* **Frozen backbone.** UNet, VAE and the text-encoder transformer are frozen
  (`requires_grad_(False)`, `eval()`); only the hypernetwork heads ever train.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Community-maintained SD-1.5 (auto-downloaded by diffusers on first use). CIDM uses
# SD-1.5, so this is the parity default. Override via config `sd_model_id` with a local
# path or another repo id (e.g. "sd-legacy/stable-diffusion-v1-5").
DEFAULT_SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"

_DTYPES = {"fp32": torch.float32, "float32": torch.float32,
           "fp16": torch.float16, "float16": torch.float16,
           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def resolve_dtype(name: Optional[str]) -> torch.dtype:
    if name is None:
        return torch.float32
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES[str(name).lower()]


@dataclasses.dataclass
class ModelBundle:
    """Frozen SD-1.5 components + the metadata the hyper stack needs."""

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler        # training: add_noise / q-sample
    ddim_scheduler: DDIMScheduler         # inference: deterministic sampling
    dpm_scheduler: DPMSolverMultistepScheduler  # inference: higher-quality DPM++ sampling
    device: torch.device
    dtype: torch.dtype
    cross_attention_dim: int              # LoRA in_dim for attn2.to_k/to_v (=768)
    clip_hidden_size: int                 # CLIP pooled size = hypernet conditioning dim (=768)
    num_train_timesteps: int              # =1000
    vae_scale_factor: float               # latent scaling (=0.18215)
    model_id: str

    @property
    def latent_channels(self) -> int:
        return self.unet.config.in_channels  # 4

    # ------------------------------------------------------------------ encoding
    def encode_text(
        self, prompts: List[str], train_tokens: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts with the frozen CLIP text encoder (UnHype-style conditioning).

        Returns
        -------
        last_hidden_state : [B, 77, 768]  — the UNet cross-attention context.
        pooler_output     : [B, 768]      — the POOLED prompt embedding that conditions the
                                            LoRA-hypernetwork (UnHype feeds `.pooler_output`).
        attention_mask    : [B, 77]        — 1 for real tokens, 0 for padding.

        CLIP is frozen, so this is no-grad -- unless `train_tokens=True` (learned identifier
        tokens, option C): then the forward keeps the graph so gradients reach the learned
        embedding rows (the transformer itself stays frozen).
        """
        batch = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        if train_tokens:
            out = self.text_encoder(input_ids=input_ids)
        else:
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids)
        return out.last_hidden_state, out.pooler_output, attention_mask

    # ------------------------------------------------------------------ vae
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images in [-1, 1], shape [B,3,H,W] -> latents [B,4,H/8,W/8] (scaled)."""
        images = images.to(self.device, dtype=self.vae.dtype)
        posterior = self.vae.encode(images).latent_dist
        z = posterior.sample() * self.vae_scale_factor
        return z

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [B,4,h,w] -> images in [0,1], [B,3,H,W]."""
        latents = latents.to(self.vae.dtype) / self.vae_scale_factor
        images = self.vae.decode(latents).sample
        return (images / 2 + 0.5).clamp(0, 1)


def load_sd(
    model_id: str = DEFAULT_SD15,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = torch.float32,
) -> ModelBundle:
    """Load + freeze SD-1.5. Weights are auto-downloaded by diffusers on first run."""
    device = torch.device(device if torch.cuda.is_available() or "cpu" not in str(device) else "cpu")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    dtype = resolve_dtype(dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    text_encoder: CLIPTextModel = pipe.text_encoder
    tokenizer: CLIPTokenizer = pipe.tokenizer

    # Schedulers share SD-1.5's config (betas etc.); two views for train vs. sample.
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Freeze the whole backbone.
    for module in (unet, vae, text_encoder):
        module.requires_grad_(False)
        module.eval()
        module.to(device)

    del pipe  # we keep only the components we need

    return ModelBundle(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        ddim_scheduler=ddim_scheduler,
        dpm_scheduler=dpm_scheduler,
        device=device,
        dtype=dtype,
        cross_attention_dim=int(unet.config.cross_attention_dim),
        clip_hidden_size=int(text_encoder.config.hidden_size),
        num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
        vae_scale_factor=float(vae.config.scaling_factor),
        model_id=model_id,
    )
