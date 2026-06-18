"""ContinualHyperManager: the LoRA-hypernetwork.

Conditioning is the prompt's CLIP **pooler_output** only; per-layer heads map `clip_pooled ->
(x_L, x_R)` for one cross-attn projection. The LoRA is **timestep-independent** — computed once
per prompt and applied unchanged at every denoising step — and is applied to every context token.

`compute_and_cache_loras()` runs all heads once and fills a per-layer cache that the injected
`CachedLoRALinear` reads via `unet.hyper`. This module just maps a pooled prompt -> LoRA.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hyper_head import HyperHead
from .injection import DEFAULT_TARGETS, inject_lora

LoraPair = Tuple[torch.Tensor, torch.Tensor]


def _key(name: str) -> str:
    return name.replace(".", "__")


class ContinualHyperManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict()           # sanitized layer name -> HyperHead
        self.layer_names: List[str] = []        # dotted names, cache-key order
        self._cache: Dict[str, LoraPair] = {}
        self._ctx: Dict[str, object] = {"pooled": None}
        self.lora_enabled = True

    # --------------------------------------------------------------- setup
    def add_head(self, layer_name: str, head: HyperHead) -> None:
        self.heads[_key(layer_name)] = head
        self.layer_names.append(layer_name)

    # --------------------------------------------------------------- lora toggle
    def enable_lora(self) -> None:
        self.lora_enabled = True

    def disable_lora(self) -> None:
        self.lora_enabled = False

    @contextmanager
    def no_lora(self):
        prev = self.lora_enabled
        self.lora_enabled = False
        try:
            yield
        finally:
            self.lora_enabled = prev

    # --------------------------------------------------------------- context
    def set_context(self, clip_pooled: torch.Tensor) -> None:
        """clip_pooled: [B, clip_size] CLIP pooler_output."""
        self._ctx = {"pooled": clip_pooled}

    def compute_and_cache_loras(self, clip_pooled=None) -> None:
        """Run all heads once: clip_pooled -> per-layer (x_L, x_R). Timestep-independent."""
        pooled = clip_pooled if clip_pooled is not None else self._ctx["pooled"]
        if pooled is None:
            raise RuntimeError("compute_and_cache_loras called without context")
        cond = pooled.to(next(self.parameters()).dtype)         # [B, clip_size]
        cache: Dict[str, LoraPair] = {}
        for name in self.layer_names:
            _, x_L, x_R = self.heads[_key(name)](cond)          # [B,in,r], [B,r,out]
            cache[name] = (x_L, x_R)
        self._cache = cache

    def get_cached_lora(self, layer_name: str) -> Optional[LoraPair]:
        return self._cache.get(layer_name)

    def clear_cache(self) -> None:
        self._cache = {}

    # --------------------------------------------------------------- regularisers
    def current_lora_magnitude(self) -> torch.Tensor:
        """Mean squared magnitude of the currently cached LoRA factors (differentiable)."""
        if not self._cache:
            return torch.zeros((), device=next(self.parameters()).device)
        terms = [x_L.pow(2).mean() + x_R.pow(2).mean() for x_L, x_R in self._cache.values()]
        return torch.stack(terms).mean()

    # --------------------------------------------------------------- von-Oswald reg helpers
    def generate_lora(self, clip_pooled: torch.Tensor) -> Dict[str, LoraPair]:
        """Run the heads on a pooled batch -> {layer: (x_L, x_R)} with the CURRENT params.
        Does NOT touch the training cache (used for reg anchors/targets)."""
        cond = clip_pooled.to(next(self.parameters()).dtype)
        return {name: self.heads[_key(name)](cond)[1:] for name in self.layer_names}

    def lora_from_params(self, clip_pooled: torch.Tensor, params: Dict[str, torch.Tensor]) -> Dict[str, LoraPair]:
        """Like generate_lora but at OVERRIDDEN head params (functional) — for the Theta+DeltaTheta
        lookahead. `params` keys match self.heads.named_parameters() names."""
        from torch.func import functional_call
        cond = clip_pooled.to(next(self.parameters()).dtype)
        out: Dict[str, LoraPair] = {}
        for name in self.layer_names:
            k = _key(name)
            sub = {pn[len(k) + 1:]: pv for pn, pv in params.items() if pn.startswith(k + ".")}
            _, x_L, x_R = functional_call(self.heads[k], sub, (cond,))
            out[name] = (x_L, x_R)
        return out

    # --------------------------------------------------------------- params
    def hyper_parameters(self) -> List[nn.Parameter]:
        return list(self.heads.parameters())


def build_hyper(
    bundle,
    rank: int = 4,
    head_hidden: int = 100,
    alpha_init: float = 1.0,
    learn_alpha: bool = True,
    target_modules: Tuple[str, ...] = DEFAULT_TARGETS,
) -> ContinualHyperManager:
    """Build per-layer heads, inject LoRA wrappers, attach to the UNet.

    Conditioning dim = CLIP pooled size (bundle.clip_hidden_size). The LoRA is
    timestep-independent (heads see only the pooled prompt).
    """
    cond_dim = int(bundle.clip_hidden_size)
    manager = ContinualHyperManager()

    wrappers = inject_lora(bundle.unet, target_modules)
    if not wrappers:
        raise RuntimeError(f"inject_lora matched 0 modules for targets {target_modules}")
    for name, wrapper in wrappers:
        head = HyperHead(
            in_dim=wrapper.in_features,
            out_dim=wrapper.out_features,
            cond_dim=cond_dim,
            rank=rank,
            hidden=head_hidden,
            alpha_init=alpha_init,
            learn_alpha=learn_alpha,
        )
        manager.add_head(name, head)
        wrapper.set_parent(bundle.unet)

    bundle.unet.hyper = manager  # CachedLoRALinear reads parent.hyper
    manager.to(bundle.device)
    manager.float()  # keep the hyper stack in fp32 even with an fp16 backbone
    return manager
