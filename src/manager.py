"""ContinualHyperManager: the LoRA-hypernetwork.

Conditioning is the prompt's CLIP **pooler_output** only; per-layer heads map `clip_pooled ->
(x_L, x_R)` for one cross-attn projection. The LoRA is **timestep-independent** — computed once
per prompt and applied unchanged at every denoising step — and is applied to every context token.

`compute_and_cache_loras()` runs all heads once and fills a per-layer cache that the injected
`CachedLoRALinear` reads via `unet.hyper`. This module just maps a pooled prompt -> LoRA.

Optional task conditioning (`task_cond.enabled`): a learnable per-task vector `V_t` modulates
the pooled prompt (`h_t = V_t * pooled`, Hadamard) and the result is Gram-Schmidt-projected
against the frozen basis of previous tasks' conditionings — so same-class concepts (whose raw
pooled embeddings are nearly parallel, cos ~ 0.79) get structurally orthogonal hyper inputs.
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
    def __init__(self, n_tasks: int = 0, cond_dim: int = 768, task_cond: Optional[dict] = None):
        super().__init__()
        self.heads = nn.ModuleDict()           # sanitized layer name -> HyperHead
        self.layer_names: List[str] = []        # dotted names, cache-key order
        self._cache: Dict[str, LoraPair] = {}
        self._token_mask: Optional[torch.Tensor] = None   # [B, 77]; None -> apply to all tokens
        self.lora_scale = 1.0                    # inference-time LoRA strength (fidelity<->editability)
        self.lora_scale_map = None               # optional [(pattern, scale)]: per-layer-group
                                                 # strengths (patterns as in target_modules);
                                                 # first match wins, fallback = lora_scale
        self._ctx: Dict[str, object] = {"pooled": None, "task_idx": None, "token_mask": None}
        self.lora_enabled = True

        tc = task_cond or {}
        self.task_cond_enabled = bool(tc.get("enabled", False)) and n_tasks > 0
        self.preserve_norm = bool(tc.get("preserve_norm", True))
        self.learn_v = bool(tc.get("learn_v", True))    # False -> V_t stays at ones (pure GS)
        if self.task_cond_enabled:
            # V_t init = ones -> h_t = pooled at task start (no-op, no regression at init).
            self.task_emb = nn.ParameterList(
                [nn.Parameter(torch.ones(cond_dim)) for _ in range(n_tasks)])
            self.register_buffer("ortho_basis", torch.zeros(n_tasks, cond_dim))   # unit rows z_i
            self.register_buffer("basis_count", torch.zeros((), dtype=torch.long))
            # canonical prompt's pooled embedding per task -- the SOURCE of the task conditioning
            self.register_buffer("canon_pooled", torch.zeros(n_tasks, cond_dim))

    # --------------------------------------------------------------- setup
    def add_head(self, layer_name: str, head: HyperHead) -> None:
        self.heads[_key(layer_name)] = head
        self.layer_names.append(layer_name)

    # --------------------------------------------------------------- task conditioning
    @torch.no_grad()
    def set_canonical(self, task_idx: int, pooled_canonical: torch.Tensor) -> None:
        """Store the canonical prompt's pooled embedding for task t (start of the task)."""
        if self.task_cond_enabled:
            self.canon_pooled[task_idx] = pooled_canonical.reshape(-1).to(self.canon_pooled.dtype)

    def condition(self, pooled: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        """pooled [B, D] -> hyper conditioning.

        Without task conditioning (or task_idx=None): identity (raw pooled, as before).
        With it the conditioning is CONSTANT per task and IGNORES `pooled` (faithful to the
        sketch: one h_t per task): h = V_t * canon_pooled_t, Gram-Schmidt against the frozen
        basis of previous tasks, `preserve_norm` rescales back to ||h||. Training and sampling
        therefore see the SAME vector -- per-prompt GS residuals proved unstable for same-class
        concepts (the shared semantic component is exactly what the projection removes).
        Returns [1, D]; CachedLoRALinear broadcasts batch-1 LoRAs over the input batch.
        """
        cond = pooled.to(next(self.parameters()).dtype)
        if task_idx is None or not self.task_cond_enabled:
            return cond
        canon = self.canon_pooled[task_idx]
        if not torch.any(canon != 0):
            raise RuntimeError(f"canonical conditioning for task {task_idx} not set "
                               "(set_canonical at task start / load a ckpt that carries it)")
        h = (self.task_emb[task_idx] * canon.to(cond.dtype)).unsqueeze(0)  # [1, D]
        n_prev = min(int(self.basis_count.item()), int(task_idx))
        if n_prev == 0:
            return h
        basis = self.ortho_basis[:n_prev].to(h.dtype)                     # [n, D]
        z = h - (h @ basis.t()) @ basis
        if self.preserve_norm:
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8) * h.norm(dim=-1, keepdim=True)
        return z

    @torch.no_grad()
    def freeze_task_basis(self, task_idx: int) -> None:
        """After task t finishes: freeze its conditioning direction z_t into the ortho basis."""
        if not self.task_cond_enabled:
            return
        h = self.condition(self.canon_pooled[task_idx:task_idx + 1], task_idx)[0]
        z = h / h.norm().clamp_min(1e-8)
        self.ortho_basis[task_idx] = z
        self.basis_count.fill_(max(int(self.basis_count.item()), task_idx + 1))

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
    def set_context(self, clip_pooled: torch.Tensor, task_idx: Optional[int] = None,
                    token_mask: Optional[torch.Tensor] = None) -> None:
        """clip_pooled: [B, clip_size] CLIP pooler_output; task_idx: task-conditioning index;
        token_mask: [B, 77] LoRA application mask (concept-token positions), None -> all."""
        self._ctx = {"pooled": clip_pooled, "task_idx": task_idx, "token_mask": token_mask}

    def compute_and_cache_loras(self, clip_pooled=None, task_idx: Optional[int] = None,
                                token_mask: Optional[torch.Tensor] = None) -> None:
        """Run all heads once: clip_pooled -> per-layer (x_L, x_R). Timestep-independent."""
        if clip_pooled is None:
            clip_pooled = self._ctx["pooled"]
            task_idx = self._ctx.get("task_idx")
            token_mask = self._ctx.get("token_mask")
        if clip_pooled is None:
            raise RuntimeError("compute_and_cache_loras called without context")
        cond = self.condition(clip_pooled, task_idx)            # [B, clip_size]
        cache: Dict[str, LoraPair] = {}
        for name in self.layer_names:
            _, x_L, x_R = self.heads[_key(name)](cond)          # [B,in,r], [B,r,out]
            sc = self._scale_for(name)
            if sc != 1.0:
                x_L = x_L * sc
            cache[name] = (x_L, x_R)
        self._cache = cache
        self._token_mask = token_mask

    def get_token_mask(self) -> Optional[torch.Tensor]:
        return self._token_mask

    def _scale_for(self, layer_name: str) -> float:
        if self.lora_scale_map:
            from .injection import _match
            for pattern, sc in self.lora_scale_map:
                if _match(layer_name, pattern):
                    return float(sc)
        return self.lora_scale

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

    def task_parameters(self, task_idx: int) -> List[nn.Parameter]:
        """The current task's learnable embedding (empty when task conditioning is off or
        `task_cond.learn_v` is false -- then V_t stays at ones and the conditioning is pure
        Gram-Schmidt of the canonical pooled, with no learned modulation)."""
        return [self.task_emb[task_idx]] if (self.task_cond_enabled and self.learn_v) else []


def build_hyper(
    bundle,
    rank: int = 4,
    head_hidden: int = 100,
    alpha_init: float = 1.0,
    learn_alpha: bool = True,
    target_modules: Tuple[str, ...] = DEFAULT_TARGETS,
    n_tasks: int = 0,
    task_cond: Optional[dict] = None,
) -> ContinualHyperManager:
    """Build per-layer heads, inject LoRA wrappers, attach to the UNet.

    Conditioning dim = CLIP pooled size (bundle.clip_hidden_size). The LoRA is
    timestep-independent (heads see only the pooled prompt). `task_cond` (config section)
    turns on the learnable per-task embeddings + Gram-Schmidt orthogonalization.
    """
    cond_dim = int(bundle.clip_hidden_size)
    manager = ContinualHyperManager(n_tasks=n_tasks, cond_dim=cond_dim, task_cond=task_cond)

    wrappers = inject_lora(bundle.unet, target_modules)
    if not wrappers:
        raise RuntimeError(f"inject_lora matched 0 modules for targets {target_modules}")
    print(f"[hyper] {len(wrappers)} LoRA layers for targets {list(target_modules)}", flush=True)
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
