"""SVDiff backbone (Han et al. 2023) -- the parameterization L2DM is built on.

Every weight matrix is decomposed once, `W = U diag(sigma) V^T`, and only a per-singular-value
**spectral shift** `delta` is trained:

    W' = U diag(ReLU(sigma + scale * delta)) V^T ,   delta initialised to 0

Convolutions are reshaped `[c_out, c_in, kh, kw] -> [c_out, c_in*kh*kw]` before the SVD and back
afterwards. This follows the reference implementation (mkshing/svdiff-pytorch) exactly, including
the ReLU on the shifted spectrum and the zero init.

Only the deltas are trainable (~1.7M parameters for SD-1.5's UNet), which is what L2DM reports.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SVDBase(nn.Module):
    """Holds the frozen factors and the trainable spectral shift."""

    def __init__(self, weight: torch.Tensor, bias):
        super().__init__()
        self.orig_shape = tuple(weight.shape)
        w2d = weight.reshape(weight.shape[0], -1).float()
        u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
        self.register_buffer("U", u)
        self.register_buffer("S", s)
        self.register_buffer("Vh", vh)
        self.delta = nn.Parameter(torch.zeros_like(s))
        self.bias = None if bias is None else nn.Parameter(bias.clone(), requires_grad=False)
        self.scale = 1.0

    def weight_updated(self, dtype: torch.dtype) -> torch.Tensor:
        s = F.relu(self.S + self.scale * self.delta)
        w = (self.U * s.unsqueeze(0)) @ self.Vh          # U diag(s) Vh
        return w.reshape(self.orig_shape).to(dtype)


class SVDLinear(_SVDBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_updated(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class SVDConv2d(_SVDBase):
    def __init__(self, conv: nn.Conv2d):
        super().__init__(conv.weight.data, None if conv.bias is None else conv.bias.data)
        self.stride, self.padding, self.dilation, self.groups = (
            conv.stride, conv.padding, conv.dilation, conv.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight_updated(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype),
                        self.stride, self.padding, self.dilation, self.groups)


@torch.no_grad()
def inject_svdiff(module: nn.Module, min_dim: int = 16, name: str = "") -> List[Tuple[str, nn.Module]]:
    """Replace Linear/Conv2d children by their SVD-parameterized versions (in place).

    `min_dim` skips tiny projections (e.g. time-embedding biases) where an SVD is pointless.
    Returns the list of (name, module) that were replaced."""
    swapped: List[Tuple[str, nn.Module]] = []
    for child_name, child in list(module.named_children()):
        full = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_dim:
            new = SVDLinear(child.weight.data, None if child.bias is None else child.bias.data)
            setattr(module, child_name, new.to(child.weight.device))
            swapped.append((full, new))
        elif isinstance(child, nn.Conv2d) and min(child.in_channels, child.out_channels) >= min_dim:
            new = SVDConv2d(child)
            setattr(module, child_name, new.to(child.weight.device))
            swapped.append((full, new))
        else:
            swapped.extend(inject_svdiff(child, min_dim, full))
    return swapped


def spectral_shifts(swapped: List[Tuple[str, nn.Module]]) -> List[nn.Parameter]:
    return [m.delta for _, m in swapped]


def shift_state(swapped: List[Tuple[str, nn.Module]]) -> dict:
    """Only the deltas -- the whole point of SVDiff is that this is the checkpoint."""
    return {n: m.delta.detach().cpu().clone() for n, m in swapped}


def load_shift_state(swapped: List[Tuple[str, nn.Module]], state: dict) -> None:
    by_name = dict(swapped)
    with torch.no_grad():
        for n, v in state.items():
            if n in by_name:
                by_name[n].delta.copy_(v.to(by_name[n].delta.device))


def set_scale(swapped: List[Tuple[str, nn.Module]], scale: float) -> None:
    """Inference-time strength of the spectral shift (analogous to our LoRA scale)."""
    for _, m in swapped:
        m.scale = scale
