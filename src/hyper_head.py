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
        basis_q: int = 0,
        learn_basis: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.q = int(basis_q)

        # Factorised output. Measured on the baked adapters: across T tasks the columns of x_L
        # span only ~T*r dimensions of R^in (39 of 40 at T=10, in up to 1280). The dense output
        # layer therefore spends h * in * r parameters on freedom the head never uses. With
        # `basis_q` the head predicts q*r coefficients against a task-shared basis U (resp. V)
        # instead, which is the same low-rank trick as LoRA applied one level up.
        l_out = (self.q * rank) if self.q else (in_dim * rank)
        r_out = (self.q * rank) if self.q else (out_dim * rank)
        self.left = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, l_out))
        self.right = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, r_out))
        if self.q:
            # 1/sqrt(q) keeps Var[(U z)_i] == Var[z_k], so the factored branch starts at the same
            # scale as the dense one and the zero-init below still gives an exactly-zero delta.
            u = torch.randn(in_dim, self.q) / (self.q ** 0.5)
            v = torch.randn(self.q, out_dim) / (self.q ** 0.5)
            if learn_basis:
                self.U, self.V = nn.Parameter(u), nn.Parameter(v)
            else:
                self.register_buffer("U", u)
                self.register_buffer("V", v)
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
        if self.q:
            x_L = torch.einsum("iq,bqr->bir", self.U.to(c.dtype),
                               self.left(c).view(b, self.q, self.rank))
            x_R = torch.einsum("brq,qo->bro", self.right(c).view(b, self.rank, self.q),
                               self.V.to(c.dtype))
        else:
            x_L = self.left(c).view(b, self.in_dim, self.rank)
            x_R = self.right(c).view(b, self.rank, self.out_dim)
        alpha = self.alpha.to(c.dtype)
        x_L = alpha * x_L
        return alpha.expand(b).clone(), x_L, x_R
