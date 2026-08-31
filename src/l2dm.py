"""L2DM (Sun et al. 2023, arXiv:2309.04430) -- lifelong text-to-image diffusion.

Faithful reimplementation of the parts that act in the SINGLE-concept sequential setting, which
is what the CIFC comparison tables measure. The multi-concept inference modules (CAA / OAA) only
fire on prompts containing several concepts and are therefore out of scope here.

Components (equation numbers follow the paper):

* backbone     : SVDiff spectral shifts (see `src/svdiff.py`); lr 1.5e-3, 500 steps/task, batch 2,
                 rare identifier token ("sks"-style, NOT learned), text encoder also shifted.
* TAME  (Eq.4) : alpha * sum_{l<k} E||eps - eps_theta(z_t^l|c^l)||^2  +  beta * sum_l L_PR
                 -- replay over generated images of past concepts + DreamBooth prior preservation
                 (200 prior images per concept, 50 sampled per step group).
* ECD   (Eq.6) : gamma * sum_{l<k} E||eps_{theta^{k-1}}(z_t^l|c^l) - eps_theta(z_t^l|c^l)||^2
                 -- distillation from the previous task's model on the same memory samples.
* Rainbow memory (Alg.1, Eq.5): the long-term bank stores CLIP features + prompts of the REAL
                 training images; the short-term bank keeps, for every past concept, the
                 generated images maximizing
                     S = mean_{other tasks}(1 - sim) + beta_s * sim(to that concept's real feature)
                 i.e. images that are diverse across tasks yet faithful to their own concept.

The loss weights alpha, beta, gamma are NOT specified in the paper, so they are swept like the
other baselines' lambda.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class MemoryBanks:
    """Long-term: CLIP features + prompts of real images. Short-term: selected generated images."""
    long_feats: Dict[int, torch.Tensor] = field(default_factory=dict)     # task -> [n, d]
    long_prompts: Dict[int, List[str]] = field(default_factory=dict)
    short_images: Dict[int, torch.Tensor] = field(default_factory=dict)   # task -> [m, 3, H, W]
    short_prompts: Dict[int, List[str]] = field(default_factory=dict)

    def has_past(self, k: int) -> bool:
        return any(t < k and len(self.short_prompts.get(t, [])) for t in self.short_prompts)

    def sample(self, k: int, batch: int, device) -> Optional[Tuple[torch.Tensor, List[str]]]:
        """Draw a batch from the short-term memory of tasks before k."""
        pool = [(t, i) for t in range(k) for i in range(len(self.short_prompts.get(t, [])))]
        if not pool:
            return None
        picks = [pool[random.randrange(len(pool))] for _ in range(batch)]
        imgs = torch.stack([self.short_images[t][i] for t, i in picks]).to(device)
        prompts = [self.short_prompts[t][i] for t, i in picks]
        return imgs, prompts


def rainbow_scores(gen_feats: Dict[int, torch.Tensor], real_feats: Dict[int, torch.Tensor],
                   beta_s: float = 1.0) -> Dict[int, torch.Tensor]:
    """Eq. (5): diversity across tasks + fidelity to the concept's own real features.

    gen_feats[t] : [n_t, d] L2-normalised CLIP features of images generated for task t
    real_feats[t]: [m_t, d] the task's real-image features from the long-term bank
    Returns per-task scores [n_t] used to pick the short-term memory."""
    all_gen = torch.cat([v for v in gen_feats.values()], 0) if gen_feats else None
    out = {}
    for t, g in gen_feats.items():
        diversity = (1.0 - (g @ all_gen.t())).mean(dim=1)          # mean over all generated
        fidelity = (g @ real_feats[t].t()).mean(dim=1)             # mean sim to own real images
        out[t] = diversity + beta_s * fidelity
    return out


def tame_replay_loss(bundle, unet, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """First term of Eq. (4): plain diffusion loss on replayed images of past concepts."""
    cond, pooled, _ = bundle.encode_text(list(prompts), train_tokens=True)
    z0 = bundle.encode_images(images)
    noise = torch.randn_like(z0)
    t = torch.randint(0, bundle.num_train_timesteps, (z0.shape[0],), device=z0.device)
    z_t = bundle.noise_scheduler.add_noise(z0, noise, t)
    ac = (bundle.added_cond(z_t.shape[0], images.shape[-1], images.shape[-1], pooled=pooled)
          if getattr(bundle, "is_sdxl", False) else None)
    eps = unet(z_t, t, encoder_hidden_states=cond, added_cond_kwargs=ac).sample
    return F.mse_loss(eps.float(), noise.float()), (z_t, t, cond, ac)


def ecd_loss(unet, teacher_ctx, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
             eps_student: torch.Tensor, ac=None) -> torch.Tensor:
    """Eq. (6): distillation from the previous task's model on the memory samples."""
    with torch.no_grad(), teacher_ctx():
        eps_teacher = unet(z_t, t, encoder_hidden_states=cond, added_cond_kwargs=ac).sample
    return F.mse_loss(eps_student.float(), eps_teacher.detach().float())


def load_prior_batch(prior_dir: str, class_word: str, batch: int, resolution: int,
                     device) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """DreamBooth prior preservation: a batch of class images generated by the BASE model."""
    d = os.path.join(prior_dir, class_word.replace(" ", "_"))
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]
    if not files:
        return None
    picks = random.sample(files, min(batch, len(files)))
    import numpy as np
    ims = []
    for f in picks:
        im = Image.open(os.path.join(d, f)).convert("RGB").resize((resolution, resolution),
                                                                  Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0).permute(2, 0, 1)
        ims.append(arr * 2.0 - 1.0)
    return torch.stack(ims).to(device), [f"a photo of {class_word}"] * len(picks)
