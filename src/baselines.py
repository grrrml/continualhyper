"""Continual-learning baselines: a plain (non-hypernetwork) LoRA trained sequentially.

The comparison in the paper isolates the CL mechanism, so every baseline shares one backbone
recipe -- LoRA on `attn2.{to_q,to_k,to_v,to_out.0}` plus one learned identifier token per concept,
frozen UNet/VAE/CLIP, the hyperparameters CIDM reports (r=4, lr 1e-4 LoRA / 1e-3 embeddings) --
and differs only in the penalty added to the diffusion loss:

* `finetune` : nothing (lower reference).
* `ewc`      : Kirkpatrick et al. 2017 -- (lambda/2) * sum_k sum_i F^k_i (theta_i - theta*^k_i)^2
               with F^k the diagonal empirical Fisher of task k (squared per-batch gradients).
* `lwf`      : Li & Hoiem 2016, response distillation -- MSE between the current model's and the
               previous task's model epsilon-prediction, conditioned on OLD concepts' prompts.
* `clora`    : Smith et al. 2023 (arXiv:2304.06027) -- one LoRA pair per task, previous ones
               frozen and summed into the applied delta, with the self-regularization
               || |sum_{t'<t} A_t' B_t'| (*) (A_t B_t) ||_F^2  (elementwise product).
* `lora_m`   : independent LoRA per task (no CL mechanism at all), MERGED at inference:
               W = W_0 + sum_t A_t B_t. Trains exactly like `clora` with lam=0.
* `lora_c`   : the same independent per-task LoRAs, but COMPOSED at inference by averaging the
               epsilon-predictions of the individual adapters (one UNet pass per adapter) --
               CIDM's `--method composer` path. Identical weights to `lora_m`; the difference
               lives in sampling (`StaticLoRABank.compose_mode`).

`StaticLoRABank` mimics the manager interface (`get_cached_lora`, `lora_enabled`,
`get_token_mask`, `lora_scale`) so the existing injection, sampling and evaluation code work
unchanged.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LoraPair = Tuple[torch.Tensor, torch.Tensor]


class StaticLoRABank(nn.Module):
    """Plain LoRA weights per layer. With `per_task=True` (C-LoRA) each task adds its own pair
    and the applied delta is the sum over tasks -- implemented by concatenating along the rank
    axis, since [A_1 ... A_T] @ [B_1; ...; B_T] == sum_t A_t B_t."""

    def __init__(self, wrappers, rank: int = 4, per_task: bool = False, n_tasks: int = 1):
        super().__init__()
        self.solo_task = None      # None -> apply all tasks' deltas; int -> only that task
                                   # (LoRA-C composition runs one adapter at a time)
        self.layer_names: List[str] = [name for name, _ in wrappers]
        self.shapes = {name: (w.in_features, w.out_features) for name, w in wrappers}
        self.rank = rank
        self.per_task = per_task
        self.lora_enabled = True
        self.lora_scale = 1.0
        self.lora_scale_map = None
        self._active = 1 if not per_task else 0        # how many task-pairs are in use
        n = n_tasks if per_task else 1
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for name, (i_dim, o_dim) in self.shapes.items():
            key = _key(name)
            self.A[key] = nn.Parameter(torch.stack([_kaiming(i_dim, rank) for _ in range(n)]))
            self.B[key] = nn.Parameter(torch.zeros(n, rank, o_dim))

    # ----------------------------------------------------------------- task lifecycle
    def start_task(self, task_idx: int) -> List[nn.Parameter]:
        """Parameters to optimize for this task; C-LoRA trains only the new pair."""
        if self.per_task:
            self._active = task_idx + 1
            return list(self.A.parameters()) + list(self.B.parameters())
        self._active = 1
        return list(self.parameters())

    def task_slice(self, task_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Per-layer (A_t, B_t) of one task (C-LoRA penalty operates on these)."""
        t = task_idx if self.per_task else 0
        return ({k: v[t] for k, v in self.A.items()}, {k: v[t] for k, v in self.B.items()})

    # ----------------------------------------------------------------- manager interface
    def get_cached_lora(self, layer_name: str) -> Optional[LoraPair]:
        key = _key(layer_name)
        if key not in self.A:
            return None
        if self.solo_task is not None:                                    # LoRA-C: single adapter
            a = self.A[key][self.solo_task:self.solo_task + 1]
            b = self.B[key][self.solo_task:self.solo_task + 1]
        else:
            a, b = self.A[key][:self._active], self.B[key][:self._active]  # [T,in,r], [T,r,out]
        x_L = a.permute(1, 0, 2).reshape(a.shape[1], -1)                  # [in, T*r]
        x_R = b.reshape(-1, b.shape[2])                                   # [T*r, out]
        if self.lora_scale != 1.0:
            x_L = x_L * self.lora_scale
        return x_L.unsqueeze(0), x_R.unsqueeze(0)

    def get_token_mask(self):
        return None                      # baselines apply the delta to the whole context

    def enable_lora(self):
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False

    def set_context(self, *a, **kw):     # no conditioning: weights are static
        return None

    @contextmanager
    def no_lora(self):
        """Unconditional branch of CFG: the sampler runs it with the adapter switched off."""
        prev = self.lora_enabled
        self.lora_enabled = False
        try:
            yield
        finally:
            self.lora_enabled = prev

    def compute_and_cache_loras(self, *a, **kw):
        return None

    def eval_only_grads(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def _key(name: str) -> str:
    return name.replace(".", "__")


def _kaiming(i_dim: int, rank: int) -> torch.Tensor:
    t = torch.empty(i_dim, rank)
    nn.init.kaiming_uniform_(t, a=5 ** 0.5)
    return t


# --------------------------------------------------------------------------- EWC
@torch.enable_grad()
def diagonal_fisher(loss_fn, params: List[nn.Parameter], n_batches: int) -> List[torch.Tensor]:
    """E[(d L / d theta)^2] over `n_batches` draws of the task's own loss (Kirkpatrick et al.)."""
    fisher = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for f, g in zip(fisher, grads):
            if g is not None:
                f += g.detach() ** 2
    return [f / max(n_batches, 1) for f in fisher]


def ewc_penalty(params: List[nn.Parameter], anchors: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]
                ) -> torch.Tensor:
    """(1/2) * sum over stored (Fisher, theta*) pairs of F_i (theta_i - theta*_i)^2."""
    if not anchors:
        return torch.zeros((), device=params[0].device)
    total = torch.zeros((), device=params[0].device)
    for fisher, star in anchors:
        for p, f, s in zip(params, fisher, star):
            total = total + (f * (p - s) ** 2).sum()
    return 0.5 * total


# --------------------------------------------------------------------------- LwF
def lwf_distill(eps_now: torch.Tensor, eps_prev: torch.Tensor) -> torch.Tensor:
    """Response distillation: keep the previous model's prediction for old concepts."""
    return F.mse_loss(eps_now.float(), eps_prev.detach().float())


# --------------------------------------------------------------------------- C-LoRA
def clora_penalty(bank: StaticLoRABank, task_idx: int) -> torch.Tensor:
    """|| |sum_{t'<t} A_t' B_t'| (*) (A_t B_t) ||_F^2 summed over layers (Smith et al. 2023)."""
    if task_idx == 0:
        return torch.zeros((), device=next(bank.parameters()).device)
    total = torch.zeros((), device=next(bank.parameters()).device)
    for key in bank.A:
        a_new, b_new = bank.A[key][task_idx], bank.B[key][task_idx]
        a_old, b_old = bank.A[key][:task_idx].detach(), bank.B[key][:task_idx].detach()
        delta_old = torch.einsum("tir,tro->io", a_old, b_old).abs()
        delta_new = a_new @ b_new
        total = total + (delta_old * delta_new).pow(2).sum()
    return total


def snapshot(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_bank(bank: StaticLoRABank) -> StaticLoRABank:
    """Frozen copy of the bank (LwF teacher)."""
    twin = copy.deepcopy(bank)
    for p in twin.parameters():
        p.requires_grad_(False)
    twin.eval()
    return twin
