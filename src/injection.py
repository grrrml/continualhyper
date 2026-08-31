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


def _match(full_name: str, target: str) -> bool:
    """Suffix match ("attn2.to_k"), optionally block-scoped via "*": "up_blocks*attn2.to_k"
    requires the name to START with the prefix and END with the suffix."""
    if "*" in target:
        prefix, suffix = target.split("*", 1)
        return full_name.startswith(prefix) and full_name.endswith(suffix)
    return full_name.endswith(target)


_SM_CACHE = {}


def _spatial_mask(box, n_tokens: int, device, dtype) -> torch.Tensor:
    """[n_tokens] region mask at this layer's resolution. `box` is either a normalised bbox
    (x0,y0,x1,y1) or a DENSE mask [s,s] derived from cross-attention -- the latter follows the
    object's actual silhouette, so a subject spilling past its box keeps its own identity."""
    if torch.is_tensor(box):
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:
            return torch.ones(n_tokens, device=device, dtype=dtype)
        m = torch.nn.functional.interpolate(box[None, None].float(), size=(side, side),
                                            mode="nearest")[0, 0]
        return m.reshape(-1).to(device=device, dtype=dtype)
    key = (box, n_tokens, str(device), str(dtype))
    hit = _SM_CACHE.get(key)
    if hit is not None:
        return hit
    side = int(round(n_tokens ** 0.5))
    if side * side != n_tokens:                    # nie-kwadratowa mapa: bez maskowania
        m = torch.ones(n_tokens, device=device, dtype=dtype)
    else:
        x0, y0, x1, y1 = box
        m = torch.zeros(side, side, device=device, dtype=dtype)
        c0, c1 = int(x0 * side), max(int(x0 * side) + 1, int(round(x1 * side)))
        r0, r1 = int(y0 * side), max(int(y0 * side) + 1, int(round(y1 * side)))
        m[r0:r1, c0:c1] = 1.0
        m = m.reshape(-1)
    _SM_CACHE[key] = m
    return m


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
        multi = hyper.get_multi(self.layer_name) if hasattr(hyper, "get_multi") else None
        if multi is not None:
            # composition: sum of per-concept deltas, each confined to its own span on the
            # 77-token axis; image-side layers (no 77 axis) receive the plain sum
            # Composition, ONE UNet pass. Each concept's delta is confined to its own region:
            # text-side layers (seq == 77) by a TOKEN mask, image-side layers by a SPATIAL mask
            # built from the concept's bbox at this layer's resolution. CIDM instead runs U+1
            # UNet passes and merges noise -- same capability, cost linear in U.
            for x_L, x_R, m, box in multi:
                if x_L.shape[0] == 1 and x.shape[0] > 1:
                    x_L = x_L.expand(x.shape[0], -1, -1); x_R = x_R.expand(x.shape[0], -1, -1)
                lora = (x.float() @ x_L.float()) @ x_R.float()
                if x.dim() == 3:
                    n = x.shape[1]
                    if m is not None and m.shape[-1] == n:               # strona tekstowa
                        mm = m.to(device=lora.device, dtype=lora.dtype)
                        if mm.shape[0] == 1 and x.shape[0] > 1:
                            mm = mm.expand(x.shape[0], -1)
                        lora = lora * mm[:, :, None]
                    elif box is not None and n != 77:                    # strona obrazowa
                        sm = _spatial_mask(box, n, lora.device, lora.dtype)
                        lora = lora * sm[None, :, None]
                out = out + lora.to(out.dtype)
            return out
        cached = hyper.get_cached_lora(self.layer_name)
        if cached is None:
            return out

        x_L, x_R = cached  # x_L: [Bc, in, rank], x_R: [Bc, rank, out]
        if x_L.shape[0] == 1 and x.shape[0] > 1:
            x_L = x_L.expand(x.shape[0], -1, -1)
            x_R = x_R.expand(x.shape[0], -1, -1)
        lora = (x.float() @ x_L.float()) @ x_R.float()
        mask = hyper.get_token_mask() if hasattr(hyper, "get_token_mask") else None
        if mask is not None and x.dim() == 3 and mask.shape[-1] == x.shape[1]:
            m = mask.to(device=lora.device, dtype=lora.dtype)
            if m.shape[0] == 1 and x.shape[0] > 1:
                m = m.expand(x.shape[0], -1)
            lora = lora * m[:, :, None]   # delta only at the concept's token positions
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
        if isinstance(child, nn.Linear) and any(_match(full_name, t) for t in target_modules):
            wrapper = CachedLoRALinear(child, layer_name=full_name).to(next(child.parameters()).device)
            setattr(module, child_name, wrapper)
            wrapped.append((full_name, wrapper))
        else:
            wrapped.extend(inject_lora(child, target_modules, full_name))
    return wrapped
