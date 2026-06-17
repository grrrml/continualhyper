"""Per-layer hypernetwork head: c* -> a single low-rank LoRA for one attn2 projection.

For a target linear with `in_dim` inputs and `out_dim` outputs we produce
`(alpha, x_L[B,in,rank], x_R[B,rank,out])`. The cached-application math downstream is exactly
UnHype's `orig + (x @ x_L) @ x_R`, so we fold the scalar `alpha` into `x_L` here and keep the
cache application alpha-free.

Deliberately removed vs. UnHype's `HyperLora`: the `xL_const + t/train_steps * head` field
ramp, the constants-trajectory buffers, and the convergence-state time axis. This head is a
plain MLP regressing the LoRA directly. The right branch's last layer is zero-initialised so
every LoRA starts at exactly zero (identity backbone) at the beginning of training.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HyperHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        rank: int = 4,
        hidden: int = 128,
        alpha_init: float = 1.0,
        learn_alpha: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        self.left = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim * rank),
        )
        self.right = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim * rank),
        )
        # Zero-init the right branch -> LoRA delta == 0 at init (backbone unchanged).
        nn.init.zeros_(self.right[-1].weight)
        nn.init.zeros_(self.right[-1].bias)

        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """c: [B, cond_dim] -> (alpha[B], x_L[B,in,rank], x_R[B,rank,out]).

        `alpha` is folded into `x_L`, so downstream application is plain `(x @ x_L) @ x_R`.
        """
        b = c.shape[0]
        x_L = self.left(c).view(b, self.in_dim, self.rank)
        x_R = self.right(c).view(b, self.rank, self.out_dim)
        alpha = self.alpha.to(c.dtype)
        x_L = alpha * x_L
        return alpha.expand(b).clone(), x_L, x_R
