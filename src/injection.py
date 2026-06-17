"""Inject application-only LoRA wrappers into the diffusers UNet cross-attention.

`CachedLoRALinear` is intentionally thin: it owns NO weight-generation logic. It reads the
LoRA `(x_L, x_R)` the manager cached for its layer and applies UnHype's exact math
`orig + (x @ x_L) @ x_R` (with the `batch==1 -> expand` trick). All weight generation lives in
the per-layer hypernetwork heads conditioned on the CLIP-pooled prompt (see manager.py).

We replace `attn2.to_k` / `attn2.to_v` (cross-attention key/value projections, input = text
context [B,77,768]). The LoRA is applied to every context token column.
"""

from __future__ import annotations

import weakref
from typing import List, Tuple

import torch
import torch.nn as nn

DEFAULT_TARGETS: Tuple[str, ...] = ("attn2.to_k", "attn2.to_v")


class CachedLoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, layer_name: str = ""):
        super().__init__()
        self.original = original_linear
        self.layer_name = layer_name
        self._parent_ref = None  # weakref to the module holding `.hyper` (the UNet)

    def set_parent(self, parent: nn.Module) -> None:
        self._parent_ref = weakref.ref(parent)

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        parent = self._parent_ref() if self._parent_ref is not None else None
        hyper = getattr(parent, "hyper", None) if parent is not None else None
        if hyper is None or not hyper.lora_enabled:
            return out
        cached = hyper.get_cached_lora(self.layer_name)
        if cached is None:
            return out

        x_L, x_R = cached  # x_L: [Bc, in, rank], x_R: [Bc, rank, out]
        if x_L.shape[0] == 1 and x.shape[0] > 1:
            x_L = x_L.expand(x.shape[0], -1, -1)
            x_R = x_R.expand(x.shape[0], -1, -1)
        lora = (x.float() @ x_L.float()) @ x_R.float()
        return out + lora.to(out.dtype)


def inject_lora(
    module: nn.Module,
    target_modules: Tuple[str, ...] = DEFAULT_TARGETS,
    name: str = "",
) -> List[Tuple[str, CachedLoRALinear]]:
    """Recursively swap matching `nn.Linear` children for `CachedLoRALinear` wrappers.

    Returns a list of (dotted_full_name, wrapper); `layer_name` is the dotted path, used as
    the manager's cache key.
    """
    wrapped: List[Tuple[str, CachedLoRALinear]] = []
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear) and any(full_name.endswith(t) for t in target_modules):
            wrapper = CachedLoRALinear(child, layer_name=full_name).to(next(child.parameters()).device)
            setattr(module, child_name, wrapper)
            wrapped.append((full_name, wrapper))
        else:
            wrapped.extend(inject_lora(child, target_modules, full_name))
    return wrapped
