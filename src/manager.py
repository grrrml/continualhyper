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
    def __init__(self, n_tasks: int = 0, cond_dim: int = 768, task_cond: Optional[dict] = None,
                 clip_dim: int = 768):
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
        # `key_dim` replaces the CLIP-pooled key by a random vector of that size. The content of
        # the key is irrelevant (ablated); what matters is that the keys end up orthogonal. Random
        # draws are near-orthogonal already, so Gram-Schmidt removes far less than it does from
        # CLIP keys (mean pairwise cos 0.77), and the conditioning stops depending on the encoder.
        self.random_keys = bool(tc.get("key_dim"))
        # Prompt-dependent modulation of the task key. A static LoRA has to pick ONE
        # identity/editability compromise per concept; conditioning the head on the prompt as
        # well lets the adapter be strong on identity-heavy prompts and yield on edit-heavy
        # ones. Zero-init on the last layer => at init the conditioning is EXACTLY the old
        # constant key, so this cannot regress the trained behaviour.
        # `key_split`: dims [0, key_split) carry task identity, dims [key_split, cond_dim)
        # carry the prompt modulation. The two channels are then orthogonal BY CONSTRUCTION,
        # so a task's modulation cone can never intersect another task's -- Gram-Schmidt only
        # ever touches the identity block (the frozen basis lives entirely inside it).
        self.key_split = int(tc.get("key_split", 0))
        # `sem_dim`: dims [0, sem_dim) carry a FROZEN semantic key (mean CLIP image embedding
        # of the concept's reference photos), dims [sem_dim, cond_dim) carry the orthogonal
        # instance key. Gram-Schmidt runs ONLY on the instance block -- projecting the semantic
        # block would remove exactly the shared component we are trying to introduce, which is
        # what destroyed semantics when keys came from CLIP text (pairwise cos 0.77).
        self.sem_dim = int(tc.get("sem_dim", 0))
        # `scale_cond`: the inference LoRA scale becomes an INPUT instead of a multiplier.
        # Today we train at s=1 and evaluate at s in [0.5,1.0], i.e. we multiply the delta by
        # a number it was never trained under. Conditioning on s (with the s-weighted loss in
        # train_cl) lets the head produce the adapter that belongs at that operating point,
        # so it can drop the parts that hurt editability rather than shrink everything.
        self.scale_cond = bool(tc.get("scale_cond", False))
        # `time_cond`: the adapter stops being timestep-independent. Early denoising steps
        # fix layout, late ones fix texture and identity; a static LoRA must pick one
        # compromise for all 50 steps, a hypernetwork does not have to. Supervision is the
        # densest we have -- every training step samples a fresh t.
        self.time_cond = bool(tc.get("time_cond", False))
        # `latent_cond`: condition on the STATE of the generation (per-channel mean/std of the
        # partial latent) instead of on the prompt. Closed loop rather than open. Note it may
        # simply rediscover the timestep, since latent statistics track the noise level --
        # that is why it is run as a separate variant, not combined with time_cond.
        self.latent_cond = bool(tc.get("latent_cond", False))
        self.cond_latent = None          # [B, 2*C] stats of z_t
        self.cond_t = 0.0                 # normalised timestep in [0,1]
        self.cond_scale_val = 1.0        # current s; set via set_context / lora_scale
        self.mod_scale = float(tc.get("prompt_mod_scale", 0.0))
        # `prompt_gate`: the minimal version -- a single bounded scalar that scales the task
        # key, i.e. a per-prompt adapter strength. 769 parameters instead of 57k, which is
        # what 52 distinct training captions can actually support. tanh bounds it to
        # [1-s, 1+s] and the zero-init makes the gate exactly 1 at the start.
        # `box_cond`: placement as an input the static bank cannot take. (cx,cy,w,h) -> Fourier
        # features -> MLP added to the task key. Zero-init => at start the adapter is exactly
        # placement-agnostic; supervision comes from paste-augmentation with a box-masked loss.
        self.box_cond = bool(tc.get("box_cond", False))
        self.cond_box = None                 # (cx,cy,w,h) in [0,1], or None -> full frame
        self.box_emb = None
        if self.box_cond:
            self.box_emb = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, cond_dim))
            nn.init.zeros_(self.box_emb[-1].weight); nn.init.zeros_(self.box_emb[-1].bias)
        # `ground_cond`: GLIGEN-style grounding, hypernet-generated. The box probe proved a
        # weight delta CANNOT carry placement (48.8% = chance), so the box moves to attention:
        # the hypernet emits a grounding vector e = G(key, fourier(box)); attn2 layers add a
        # zero-init-gated, position-dependent contribution tanh(g_l)*sigmoid(<q_i,Ke>)*Ve.
        self.ground_cond = bool(tc.get("ground_cond", False))
        self.ground_head = None
        self.ground_gates = None
        self._ground_vec = None              # [1, clip_dim] current concept+box embedding
        if self.ground_cond:
            gin = (int(tc.get("key_dim") or clip_dim)) + 64
            _m = int(tc.get("ground_gsa_tokens", 4)) if bool(tc.get("ground_gsa", False)) else 1
            self.ground_head = nn.Sequential(nn.Linear(gin, 256), nn.SiLU(),
                                             nn.Linear(256, clip_dim * _m))
            # tylko gate'y startuja z zera (styl GLIGEN); zero-init takze heada
            # daje martwy punkt: grad(gate) ~ v(e)=0 i grad(head) ~ tanh(gate)=0
            with torch.no_grad():
                self.ground_head[-1].weight.mul_(0.1); nn.init.zeros_(self.ground_head[-1].bias)
        # `ground_geo`: jawny czlon geometryczny w logicie bramki (LayoutDiffusion-style).
        # Diagnoza A/B: <q_i,Ke> nie niesie pozycji (uwaga SD bez kodowania pozycji), wiec
        # "gdzie" liczymy z geometrii: logit += <P*fourier(pos_i), R*fourier(box)>.
        self.ground_geo = bool(tc.get("ground_geo", False)) and self.ground_cond
        # `ground_geo_analytic`: maska ramki Z KONSTRUKCJI zamiast uczonych projekcji.
        # Diagnoza v2 (pos_proj na inicie po 400 krokach pelnej straty): gradient do uczonej
        # geometrii jest dlawiony przez tanh(gate)~0, a gate nie rosnie, bo losowy wzor
        # przestrzenny nie pomaga stracie - petla startowa. Analityczne inside(pos,box)
        # usuwa OSTATNI uczony element adresowania: zostaje pytanie "czy trafiony zastrzyk
        # oplaca sie stracie" i gradient gate'a odpowiada na nie wprost.
        self.ground_geo_analytic = bool(tc.get("ground_geo_analytic", False)) and self.ground_cond
        # `ground_gsa`: pelna (waska) uwaga czytajaca a la GLIGEN zamiast skalarnej bramki.
        # Diagnoza z 6 negatywow: adres dziala (maska analityczna), tresc plynie (|L-OFF|),
        # ale skalar*wektor to za ubogi zastrzyk. Tu kazda pozycja obrazu CZYTA z 4 tokenow
        # groundingu (hipersiec: klucz+ramka) przez wspolne waskie projekcje; maska pozycyjna
        # pozostaje analityczna (jedyny adres, ktory kiedykolwiek zadzialal); klucz taska
        # moduluje FiLM-em wartosci - hipersiec podaje tresc I dostraja mechanizm czytania.
        self.ground_gsa = bool(tc.get("ground_gsa", False)) and self.ground_cond
        self.ground_gsa_tokens = int(tc.get("ground_gsa_tokens", 4))
        self.ground_gsa_mods = None
        self.ground_film = None
        self._ground_film_gb = None
        if self.ground_gsa:
            self.ground_film = nn.Linear(int(tc.get("key_dim") or clip_dim), 128)
            nn.init.zeros_(self.ground_film.weight); nn.init.zeros_(self.ground_film.bias)
        self.ground_geo_a = None
        self.ground_geo_b = None
        self._ground_box = None
        if self.ground_geo_analytic:
            self.ground_geo_a = nn.Parameter(torch.tensor(5.0))
            self.ground_geo_b = nn.Parameter(torch.tensor(-2.5))
        self.ground_pos_proj = None
        self.ground_box_proj = None
        self._ground_boxvec = None
        self._geo_grid_cache = {}
        if self.ground_geo:
            self.ground_pos_proj = nn.Linear(32, 64)
            self.ground_box_proj = nn.Linear(64, 64)
        self.latent_emb = None
        if self.latent_cond:
            self.latent_emb = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, cond_dim))
            nn.init.zeros_(self.latent_emb[-1].weight); nn.init.zeros_(self.latent_emb[-1].bias)
        self.time_emb = None
        if self.time_cond:
            self.time_emb = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, cond_dim))
            nn.init.zeros_(self.time_emb[-1].weight); nn.init.zeros_(self.time_emb[-1].bias)
        self.scale_emb = None
        if self.scale_cond:
            self.scale_emb = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, cond_dim))
            nn.init.zeros_(self.scale_emb[-1].weight); nn.init.zeros_(self.scale_emb[-1].bias)
        # Separate knob on purpose: sharing `prompt_mod_scale` with the vector modulation
        # silently enabled BOTH mechanisms at once and made the comparison meaningless.
        self.gate_scale = float(tc.get("prompt_gate_scale", 0.0))
        self.prompt_gate = None
        if tc.get("prompt_gate") and self.gate_scale > 0:
            self.prompt_gate = nn.Linear(clip_dim, 1)
            nn.init.zeros_(self.prompt_gate.weight); nn.init.zeros_(self.prompt_gate.bias)
        self.prompt_mod = None
        if self.mod_scale > 0:
            hid = int(tc.get("prompt_mod_hidden", 64))
            self.prompt_mod = nn.Sequential(nn.Linear(clip_dim, hid), nn.SiLU(),
                                            nn.Linear(hid, cond_dim))
            nn.init.zeros_(self.prompt_mod[-1].weight); nn.init.zeros_(self.prompt_mod[-1].bias)
            # Registered ONLY when it actually masks something: it is a config-derived
            # constant, and registering an all-ones buffer would break loading every
            # checkpoint trained before this option existed (strict state_dict).
            if self.key_split:
                m = torch.ones(1, cond_dim); m[:, :self.key_split] = 0.0
                self.register_buffer("_mod_mask", m)
        if self.task_cond_enabled:
            # V_t init = ones -> h_t = pooled at task start (no-op, no regression at init).
            self.task_emb = nn.ParameterList(
                [nn.Parameter(torch.ones(cond_dim)) for _ in range(n_tasks)])
            self.register_buffer("ortho_basis", torch.zeros(n_tasks, cond_dim))   # unit rows z_i
            self.register_buffer("basis_count", torch.zeros((), dtype=torch.long))
            # canonical prompt's pooled embedding per task -- the SOURCE of the task conditioning
            self.register_buffer("canon_pooled", torch.zeros(n_tasks, cond_dim))

    # --------------------------------------------------------------- setup
    def add_head(self, layer_name: str, head: HyperHead, shared: bool = False) -> None:
        """`shared=True` allows the same HyperHead object under several layer names (the
        ModuleDict then holds one reference per name but only one set of parameters)."""
        self.heads[_key(layer_name)] = head
        self.layer_names.append(layer_name)

    # --------------------------------------------------------------- task conditioning
    @torch.no_grad()
    def set_canonical(self, task_idx: int, pooled_canonical: torch.Tensor,
                      sem_vec: Optional[torch.Tensor] = None) -> None:
        """Store the canonical prompt's pooled embedding for task t (start of the task).

        With `key_dim` set the argument is ignored and a deterministic random key is drawn instead.
        It is left unnormalised: a randn vector has per-component RMS 1, which is what CLIP pooled
        also has (norm 28.5 over 768 dims), so the head sees inputs of the same scale either way."""
        if not self.task_cond_enabled:
            return
        if self.random_keys:
            g = torch.Generator().manual_seed(1234 + int(task_idx))
            d = self.canon_pooled.shape[1]
            key = torch.zeros(d)
            if self.sem_dim:
                # block A: fixed random projection of the concept's mean image embedding.
                # Projection is seeded and frozen -> no continual-learning problem in it.
                v = sem_vec.reshape(-1).float()
                gp = torch.Generator().manual_seed(777)
                P = torch.randn(v.numel(), self.sem_dim, generator=gp) / (self.sem_dim ** 0.5)
                key[:self.sem_dim] = v @ P
                key[self.sem_dim:] = torch.randn(d - self.sem_dim, generator=g)
            else:
                n_id = self.key_split or d          # identity lives in the first block only
                key[:n_id] = torch.randn(n_id, generator=g)
        else:
            key = pooled_canonical.reshape(-1)
        self.canon_pooled[task_idx] = key.to(self.canon_pooled.dtype)

    def set_ground(self, task_idx: Optional[int], box=None) -> None:
        """Compute the grounding vector for (concept, box); None disables grounding."""
        if self.ground_head is None or task_idx is None:
            self._ground_vec = None
            return
        key = self.canon_pooled[task_idx:task_idx + 1].float()
        fb = self._fourier_box(box if box is not None else (0.5, 0.5, 1.0, 1.0))
        dev = key.device
        gv = self.ground_head(torch.cat([key, fb.to(dev)], dim=-1))
        self._ground_vec = gv.reshape(1, -1, gv.shape[-1] // max(1, gv.shape[-1] // 768)) \
            if False else gv
        if self.ground_gsa:
            self._ground_vec = gv.reshape(1, self.ground_gsa_tokens, -1)   # [1, M, 768]
            self._ground_film_gb = self.ground_film(key)                    # [1, 128] -> (gamma|beta)
        if self.ground_geo_analytic or self.ground_gsa:
            self._ground_box = tuple(box) if box is not None else (0.5, 0.5, 1.0, 1.0)
        if self.ground_geo:
            self._ground_boxvec = self.ground_box_proj(fb.to(dev))    # [1, 64]

    def init_ground_gsa(self, layer_dims: dict) -> None:
        """layer_dims: nazwa to_q -> in_features. Wspolne waskie projekcje per warstwa."""
        if not self.ground_gsa or self.ground_gsa_mods is not None:
            return
        mods = {}
        for n, d in layer_dims.items():
            mods[_key(n)] = nn.ModuleDict({
                "q": nn.Linear(d, 64, bias=False),
                "k": nn.Linear(768, 64, bias=False),
                "v": nn.Linear(768, 64, bias=False),
                "o": nn.Linear(64, d, bias=False),
            })
        self.ground_gsa_mods = nn.ModuleDict(mods)

    def geo_inside(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """[n,1] analityczna maska inside(pos, ramka) w [0,1] (adres dla GSA)."""
        key = ("xy", h, w)
        grid = self._geo_grid_cache.get(key)
        if grid is None:
            ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
            xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            self._geo_grid_cache[key] = grid
        g = grid.to(device=device, dtype=torch.float32)
        cx, cy, bw, bh = self._ground_box or (0.5, 0.5, 1.0, 1.0)
        sh = 40.0
        inside = (torch.sigmoid(sh * (g[:, 0] - (cx - bw / 2)))
                  * torch.sigmoid(sh * ((cx + bw / 2) - g[:, 0]))
                  * torch.sigmoid(sh * (g[:, 1] - (cy - bh / 2)))
                  * torch.sigmoid(sh * ((cy + bh / 2) - g[:, 1])))
        return inside.to(dtype).unsqueeze(1)

    def gsa_read(self, attn2_name: str, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Uwaga czytajaca: x [B,n,d] -> wklad [B,n,d] (bez gate'a i maski - dodaje procesor)."""
        if not self.ground_gsa or self._ground_vec is None:
            return None
        k = _key(attn2_name + ".to_q")
        if k not in self.ground_gsa_mods:
            return None
        m = self.ground_gsa_mods[k]
        e = self._ground_vec.to(x.device)                              # [1, M, 768]
        gb = self._ground_film_gb.to(x.device)                         # [1, 128]
        gamma, beta = gb[:, :64], gb[:, 64:]
        xf = x.float()
        q = m["q"](xf)                                                 # [B, n, 64]
        kk = m["k"](e.float())                                         # [1, M, 64]
        v = m["v"](e.float()) * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        att = torch.softmax(q @ kk.transpose(-1, -2) / 8.0, dim=-1)    # [B, n, M]
        return m["o"](att @ v).to(x.dtype)                             # [B, n, d]

    def init_ground_gates(self, layer_names) -> None:
        if self.ground_head is None or self.ground_gates is not None:
            return
        self.ground_gates = nn.ParameterDict(
            {_key(n): nn.Parameter(torch.zeros(1)) for n in layer_names if n.endswith("to_q")})

    def get_ground(self, attn2_name: str):
        """(e_ground, gate) for this attn2 module, or None. Keyed by the to_q layer name."""
        if self._ground_vec is None or self.ground_gates is None:
            return None
        k = _key(attn2_name + ".to_q")
        if k not in self.ground_gates:
            return None
        geo = self._ground_boxvec if self.ground_geo else (
            self._ground_box if self.ground_geo_analytic else None)
        return self._ground_vec, self.ground_gates[k], geo

    def geo_logit(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """[n,1] geometryczny skladnik logitu bramki dla siatki h x w (pozycje z konstrukcji)."""
        if self.ground_geo_analytic:
            key = ("xy", h, w)
            grid = self._geo_grid_cache.get(key)
            if grid is None:
                ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
                xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)   # [n, 2]
                self._geo_grid_cache[key] = grid
            g = grid.to(device=device, dtype=torch.float32)
            cx, cy, bw, bh = self._ground_box or (0.5, 0.5, 1.0, 1.0)
            sh = 40.0                                             # ostrosc krawedzi (~2px @64)
            inside = (torch.sigmoid(sh * (g[:, 0] - (cx - bw / 2)))
                      * torch.sigmoid(sh * ((cx + bw / 2) - g[:, 0]))
                      * torch.sigmoid(sh * (g[:, 1] - (cy - bh / 2)))
                      * torch.sigmoid(sh * ((cy + bh / 2) - g[:, 1])))
            return (self.ground_geo_a * inside + self.ground_geo_b).to(dtype).unsqueeze(1)
        key = (h, w)
        grid = self._geo_grid_cache.get(key)
        if grid is None:
            ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
            xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            v = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)      # [n, 2]
            k = 2.0 ** torch.arange(8) * torch.pi
            ang = v[:, :, None] * k[None, None, :]                        # [n, 2, 8]
            grid = torch.cat([ang.sin(), ang.cos()], dim=2).reshape(-1, 32)
            self._geo_grid_cache[key] = grid
        grid = grid.to(device=device, dtype=torch.float32)
        pos = self.ground_pos_proj(grid)                                  # [n, 64]
        return (pos @ self._ground_boxvec.float().t()).to(dtype)          # [n, 1]

    @staticmethod
    def _fourier_box(box) -> torch.Tensor:
        """(cx,cy,w,h) in [0,1] -> 64-d Fourier features (8 frequencies x sin/cos x 4 coords)."""
        v = torch.as_tensor(box, dtype=torch.float32)
        k = 2.0 ** torch.arange(8) * torch.pi
        ang = v[:, None] * k[None, :]                       # [4, 8]
        return torch.cat([ang.sin(), ang.cos()], dim=1).reshape(1, 64)

    @staticmethod
    def latent_stats(z: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B,2C]: per-channel mean and std. Cheap, permutation-free summary
        of the generation state; 8 numbers for SD-1.5's 4-channel latent."""
        return torch.cat([z.mean(dim=(2, 3)), z.std(dim=(2, 3))], dim=-1)

    def condition(self, pooled: torch.Tensor, task_idx: Optional[int] = None,
                  use_prompt_mod: bool = True) -> torch.Tensor:
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
        if self.latent_emb is not None and self.cond_latent is not None:
            h = h + self.latent_emb(self.cond_latent.to(h.dtype))
        if self.time_emb is not None:
            tt = torch.as_tensor(self.cond_t, device=h.device, dtype=h.dtype).reshape(-1, 1)
            f = torch.arange(16, device=h.device, dtype=h.dtype)
            ang = tt * (2.0 ** f) * 3.14159265
            h = h + self.time_emb(torch.cat([ang.sin(), ang.cos()], dim=-1))
        if self.scale_emb is not None:
            sv = torch.full((1, 1), float(self.cond_scale_val), device=h.device, dtype=h.dtype)
            h = h + self.scale_emb(sv)
        if use_prompt_mod and self.prompt_gate is not None:
            h = h * (1.0 + self.gate_scale * torch.tanh(self.prompt_gate(cond).mean()))
        if use_prompt_mod and self.prompt_mod is not None:
            # added BEFORE Gram-Schmidt so the modulated key stays orthogonal to old tasks
            mod = self.prompt_mod(cond).mean(0, keepdim=True)
            if self.key_split:
                mod = mod * self._mod_mask.to(cond.dtype)
            h = h + self.mod_scale * mod
        n_prev = min(int(self.basis_count.item()), int(task_idx))
        if n_prev > 0:
            basis = self.ortho_basis[:n_prev].to(h.dtype)                 # [n, D]
            h = h - (h @ basis.t()) @ basis
            if self.preserve_norm:
                h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8) * h.norm(dim=-1, keepdim=True)
        # placement AFTER the projection: identity stays orthogonal across tasks (that is what
        # GS is for), while the box code is DELIBERATELY shared -- same box, same modulation.
        # Adding it earlier would let GS project the box component out, more so for later tasks.
        if self.box_emb is not None:
            box = self.cond_box if self.cond_box is not None else (0.5, 0.5, 1.0, 1.0)
            h = h + self.box_emb(self._fourier_box(box).to(h.device, h.dtype))
        return h

    @torch.no_grad()
    def freeze_task_basis(self, task_idx: int) -> None:
        """After task t finishes: freeze its conditioning direction z_t into the ortho basis."""
        if not self.task_cond_enabled:
            return
        _saved_box = self.cond_box
        self.cond_box = None                 # basis/anchors pinned at the canonical full frame
        h = self.condition(self.canon_pooled[task_idx:task_idx + 1], task_idx, use_prompt_mod=False)[0]
        self.cond_box = _saved_box
        if self.sem_dim:
            h = h.clone(); h[:self.sem_dim] = 0.0    # basis has no semantic component, so the
                                                     # projection below can only touch the tail
        z = h / h.norm().clamp_min(1e-8)
        self.ortho_basis[task_idx] = z
        self.basis_count.fill_(max(int(self.basis_count.item()), task_idx + 1))

    # --------------------------------------------------------------- composition (multi-adapter)
    @torch.no_grad()
    def set_multi_context(self, entries) -> None:
        """Composition inference: one LoRA per concept, each masked to its own token span.
        entries: [{'task_idx': int, 'token_mask': [1,77] or None, 'box': (x0,y0,x1,y1) or None}]
        Routing happens at the INPUT level (which task, which span) -- the probes showed
        contextual routing is causally contaminated, so selection stays structural."""
        device = next(self.parameters()).device
        ref = torch.zeros(1, 768, device=device)      # condition() ignores content for task keys
        multi = []
        for e in entries:
            cond = self.condition(ref, e["task_idx"])
            cache = {}
            for name in self.layer_names:
                _, x_L, x_R = self.heads[_key(name)](self._head_input(cond, name))
                sc = self._scale_for(name)
                if sc != 1.0:
                    x_L = x_L * sc
                cache[name] = (x_L, x_R)
            multi.append((cache, e.get("token_mask"), e.get("box")))
        self._multi = multi

    def clear_multi(self) -> None:
        self._multi = None

    def get_multi(self, name):
        m = getattr(self, "_multi", None)
        if not m:
            return None
        return [(c[name][0], c[name][1], msk, box) for c, msk, box in m]

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
            _, x_L, x_R = self.heads[_key(name)](self._head_input(cond, name))
            sc = self._scale_for(name)
            if sc != 1.0:
                x_L = x_L * sc
            cache[name] = (x_L, x_R)
        self._cache = cache
        self._token_mask = token_mask

    def get_token_mask(self) -> Optional[torch.Tensor]:
        return self._token_mask

    def _head_input(self, cond: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Conditioning seen by one layer's head. With shared heads a small learned per-layer
        code is appended so layers of the same shape are told apart."""
        codes = getattr(self, "layer_codes", None)
        if codes is None:
            return cond
        code = codes[_key(layer_name)].to(cond.dtype).unsqueeze(0).expand(cond.shape[0], -1)
        return torch.cat([cond, code], dim=-1)

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
        return {name: self.heads[_key(name)](self._head_input(cond, name))[1:]
                for name in self.layer_names}

    def lora_from_params(self, clip_pooled: torch.Tensor, params: Dict[str, torch.Tensor]) -> Dict[str, LoraPair]:
        """Like generate_lora but at OVERRIDDEN head params (functional) — for the Theta+DeltaTheta
        lookahead. `params` keys match self.heads.named_parameters() names."""
        from torch.func import functional_call
        cond = clip_pooled.to(next(self.parameters()).dtype)
        out: Dict[str, LoraPair] = {}
        for name in self.layer_names:
            k = _key(name)
            sub = {pn[len(k) + 1:]: pv for pn, pv in params.items() if pn.startswith(k + ".")}
            _, x_L, x_R = functional_call(self.heads[k], sub, (self._head_input(cond, name),))
            out[name] = (x_L, x_R)
        return out

    # --------------------------------------------------------------- params
    def hyper_parameters(self) -> List[nn.Parameter]:
        # nn.Module.parameters() already skips duplicates, so shared heads are counted once
        params = list(self.heads.parameters())
        codes = getattr(self, "layer_codes", None)
        params = params + (list(codes.parameters()) if codes is not None else [])
        params = params + (list(self.prompt_mod.parameters()) if self.prompt_mod is not None else [])
        params = params + (list(self.prompt_gate.parameters()) if self.prompt_gate is not None else [])
        params = params + (list(self.scale_emb.parameters()) if self.scale_emb is not None else [])
        params = params + (list(self.time_emb.parameters()) if self.time_emb is not None else [])
        params = params + (list(self.latent_emb.parameters()) if self.latent_emb is not None else [])
        params = params + (list(self.box_emb.parameters()) if self.box_emb is not None else [])
        params = params + (list(self.ground_head.parameters()) if self.ground_head is not None else [])
        if self.ground_pos_proj is not None:
            params = params + list(self.ground_pos_proj.parameters()) + list(self.ground_box_proj.parameters())
        if self.ground_geo_a is not None:
            params = params + [self.ground_geo_a, self.ground_geo_b]
        if self.ground_gsa_mods is not None:
            params = params + list(self.ground_gsa_mods.parameters()) + list(self.ground_film.parameters())
        return params + (list(self.ground_gates.parameters()) if self.ground_gates is not None else [])

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
    share_heads: bool = False,
    code_dim: int = 16,
    share_by_role: bool = False,
    basis_q: int = 0,
    learn_basis: bool = True,
) -> ContinualHyperManager:
    """Build per-layer heads, inject LoRA wrappers, attach to the UNet.

    Conditioning dim = CLIP pooled size (bundle.clip_hidden_size). The LoRA is
    timestep-independent (heads see only the pooled prompt). `task_cond` (config section)
    turns on the learnable per-task embeddings + Gram-Schmidt orthogonalization.
    """
    # `task_cond.key_dim` decouples the conditioning width from the text encoder (see set_canonical)
    cond_dim = int((task_cond or {}).get("key_dim") or bundle.clip_hidden_size)
    manager = ContinualHyperManager(n_tasks=n_tasks, cond_dim=cond_dim, task_cond=task_cond,
                                    clip_dim=int(bundle.clip_hidden_size))

    wrappers = inject_lora(bundle.unet, target_modules)
    if not wrappers:
        raise RuntimeError(f"inject_lora matched 0 modules for targets {target_modules}")
    print(f"[hyper] {len(wrappers)} LoRA layers for targets {list(target_modules)}", flush=True)

    # Optional: one head per (in_dim, out_dim) SHAPE instead of per layer. SD-1.5's attn2 has only
    # a handful of distinct shapes, so this cuts the head parameters by ~30x. Layers sharing a head
    # are told apart by a small learned per-layer code appended to the task key.
    # `share_by_role` additionally splits the buckets by projection role, so `to_q` and `to_out.0`
    # (identical shapes, opposite jobs) stop being forced through one head.
    shared: Dict[Tuple, HyperHead] = {}
    code_dim = int(code_dim) if share_heads else 0
    if share_heads:
        manager.layer_codes = nn.ParameterDict()

    for name, wrapper in wrappers:
        shape: Tuple = (wrapper.in_features, wrapper.out_features)
        if share_by_role:
            shape = shape + (name.rsplit(".", 1)[-1] if not name.endswith(".0")
                             else name.rsplit(".", 2)[-2],)
        if share_heads and shape in shared:
            head = shared[shape]
        else:
            head = HyperHead(
                in_dim=wrapper.in_features,
                out_dim=wrapper.out_features,
                cond_dim=cond_dim + code_dim,
                rank=rank,
                hidden=head_hidden,
                alpha_init=alpha_init,
                learn_alpha=learn_alpha,
                basis_q=basis_q,
                learn_basis=learn_basis,
            )
            if share_heads:
                shared[shape] = head
        if share_heads:
            manager.layer_codes[_key(name)] = nn.Parameter(torch.randn(code_dim) * 0.02)
        manager.add_head(name, head, shared=share_heads)
        wrapper.set_parent(bundle.unet)
    if share_heads:
        print(f"[hyper] shared heads: {len(shared)} buckets for {len(wrappers)} layers "
              f"(by_role={share_by_role}, code_dim={code_dim})", flush=True)

    if getattr(manager, "ground_cond", False):
        manager.init_ground_gates(manager.layer_names)
        if getattr(manager, "ground_gsa", False):
            dims = {}
            for n, mod in bundle.unet.named_modules():
                if n.endswith("attn2.to_q") and hasattr(mod, "in_features"):
                    dims[n] = int(mod.in_features)
            manager.init_ground_gsa(dims)
    bundle.unet.hyper = manager  # CachedLoRALinear reads parent.hyper
    manager.to(bundle.device)
    manager.float()  # keep the hyper stack in fp32 even with an fp16 backbone
    return manager
