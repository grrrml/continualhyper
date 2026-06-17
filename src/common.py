"""Small shared helpers: config loading, seeding, and hypernetwork checkpoint I/O."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Allow a single top-level wrapper key (UnHype style) or a flat dict.
    if isinstance(cfg, dict) and len(cfg) == 1:
        (only,) = cfg.values()
        if isinstance(only, dict):
            return only
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_hyper(manager, path: str) -> None:
    """Persist the hypernetwork (per-layer heads)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"manager": manager.state_dict(), "layer_names": list(manager.layer_names)}, path)


def load_hyper(manager, path: str, map_location: str = "cpu") -> None:
    blob = torch.load(path, map_location=map_location)
    manager.load_state_dict(blob["manager"])
