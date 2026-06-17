"""Diffusion reconstruction loss for the LoRA-hypernetwork.

The training objective is plain epsilon-prediction MSE `MSE(eps_pred, eps)`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(eps_pred: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Epsilon-prediction MSE."""
    return F.mse_loss(eps_pred, eps)
