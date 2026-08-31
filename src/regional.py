"""Regional cross-attention for SINGLE-PASS multi-concept composition.

CIDM composes by running U+1 UNet passes (one per region plus a global one) and merging the
noise predictions with bbox masks -- cost linear in the number of concepts. Here the layout is
imposed inside attention instead: image positions inside region u attend to region u's tokens
plus the shared ones, and are discouraged from attending to the other concepts' tokens. One
pass, cost independent of U.

`strength=None` -> hard mask (-inf); a float -> soft logit penalty (DenseDiffusion-style),
which avoids the boundary artefacts hard masking can produce.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


ATTN_ACC: dict = {}          # idx konceptu -> zakumulowana mapa uwagi [S,S]
ACC_SIDE = 32                # wspolna rozdzielczosc akumulacji


def _region_vec(spec, n: int, device) -> Optional[torch.Tensor]:
    """spec = bbox (x0,y0,x1,y1) albo gesta maska [s,s] -> wektor [n] w rozdzielczosci warstwy."""
    side = int(round(n ** 0.5))
    if side * side != n:
        return None
    if torch.is_tensor(spec):
        m = F.interpolate(spec[None, None].float(), size=(side, side), mode="nearest")[0, 0]
        return m.reshape(-1).to(device)
    x0, y0, x1, y1 = spec
    m = torch.zeros(side, side, device=device)
    c0, c1 = int(x0 * side), max(int(x0 * side) + 1, int(round(x1 * side)))
    r0, r1 = int(y0 * side), max(int(y0 * side) + 1, int(round(y1 * side)))
    m[r0:r1, c0:c1] = 1.0
    return m.reshape(-1)


def reset_attn_acc() -> None:
    ATTN_ACC.clear()


@torch.no_grad()
def derive_masks(n_concepts: int, gate=None, min_frac: float = 0.02, tau: float = 0.25):
    """Cross-attention maps -> DISJOINT per-concept masks by argmax over concepts.
    `gate` optionally restricts each concept to a dilated box (a safety rail early on); `tau` is
    the floor below which a pixel is background and gets NO adapter at all.
    Returns [n_concepts] masks [S,S], or None if attention has not been collected yet."""
    if len(ATTN_ACC) < n_concepts:
        return None
    A = torch.stack([ATTN_ACC[i] for i in range(n_concepts)])          # [C,S,S]
    A = A / A.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)           # per-concept normalisation
    if gate is not None:
        A = A * torch.stack(gate).to(A.device)
    win = A.argmax(0)
    strong = A.amax(0) > tau              # background belongs to NOBODY: no adapter, no penalty
    masks = []
    for c in range(n_concepts):
        m = ((win == c) & strong).float()
        if m.mean() < min_frac:           # concept claimed almost nothing -> keep its own peak
            m = (A[c] > A[c].amax() * 0.5).float()
        masks.append(m)
    return masks


class RegionalAttnProcessor:
    def __init__(self, regions, strength: Optional[float] = None, collect: bool = False,
                 confine: bool = False):
        """`confine=True`: a single region per pass -- positions OUTSIDE the box are stopped from
        attending to that concept's tokens, so the subject forms INSIDE the box. This is what makes
        CIDM's per-region pass (eq. 4 conditions on [c_u, s_u]) place its subject in the region
        instead of centring it; without it, merging two centred subjects yields one hybrid."""
        """regions: [(box, token_mask[77], is_global)]. A GLOBAL entry (a style) sits above the
        partition: it is never confined to a region and its tokens are never suppressed anywhere.
        Only the regional entries (objects) partition the image between themselves -- otherwise
        two entries both covering the whole frame would suppress each other's tokens everywhere,
        leaving the layout to the adapters alone."""
        norm = []
        for r in regions:
            box, tm = r[0], r[1]
            is_global = bool(r[2]) if len(r) > 2 else False
            norm.append((box, tm, is_global))
        self.regions = norm
        self.strength = strength
        self.collect = collect
        self.confine = confine
        self._cache = {}

    @torch.no_grad()
    def _accumulate(self, probs, B: int, n_img: int) -> None:
        """probs [B*heads, n_img, 77] -> per-concept attention map, averaged over heads,
        conditional half only, upsampled to a common resolution."""
        side = int(round(n_img ** 0.5))
        if side * side != n_img or side < 8:            # skip the coarsest maps
            return
        heads = probs.shape[0] // B
        p = probs.view(B, heads, n_img, -1)[B // 2:].mean(dim=(0, 1))     # [n_img, 77]
        for i, (_, tm, _) in enumerate(self.regions):
            sel = tm.reshape(-1)[: p.shape[1]].to(p.device) > 0
            if not bool(sel.any()):
                continue
            m = p[:, sel].sum(-1).reshape(side, side).float()
            m = F.interpolate(m[None, None], size=(ACC_SIDE, ACC_SIDE), mode="bilinear",
                              align_corners=False)[0, 0]
            ATTN_ACC[i] = ATTN_ACC.get(i, torch.zeros_like(m)) + m

    def _bias(self, n_img: int, n_tok: int, device, dtype) -> torch.Tensor:
        key = (n_img, n_tok, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        side = int(round(n_img ** 0.5))
        bias = torch.zeros(n_img, n_tok, device=device, dtype=torch.float32)
        regional = [(b, m) for b, m, g in self.regions if not g]
        if side * side == n_img and self.confine:
            for box, tm in regional:
                inside = _region_vec(box, n_img, device)
                if inside is None:
                    continue
                mine = tm.reshape(-1)[:n_tok].to(device).float()
                pen = torch.outer(1.0 - inside, mine)     # outside my box -> not my tokens
                bias = bias - pen * (1e4 if self.strength is None else float(self.strength))
            bias = bias.to(dtype); self._cache[key] = bias; return bias
        if side * side == n_img and len(regional) >= 2:
            # only REGIONAL entries partition the frame; global entries (styles) are exempt
            owned = torch.zeros(n_tok, device=device)
            for _, tm in regional:
                owned = torch.maximum(owned, tm.reshape(-1)[:n_tok].to(device).float())
            for box, tm in regional:
                inside = _region_vec(box, n_img, device)
                if inside is None:
                    continue
                mine = tm.reshape(-1)[:n_tok].to(device).float()
                foreign = (owned - mine).clamp_min(0.0)          # tokens owned by OTHER regions
                pen = torch.outer(inside, foreign)
                bias = bias - pen * (1e4 if self.strength is None else float(self.strength))
        bias = bias.to(dtype)
        self._cache[key] = bias
        return bias

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            ctx = attn.norm_encoder_hidden_states(ctx)

        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        B, n_img, _ = q.shape
        q = attn.head_to_batch_dim(q); k = attn.head_to_batch_dim(k); v = attn.head_to_batch_dim(v)

        bias = None
        if encoder_hidden_states is not None and self.regions:
            bias = self._bias(n_img, k.shape[1], q.device, q.dtype)     # [n_img, n_tok]
            heads = q.shape[0] // B
            # CFG: batch is [uncond, cond]; the layout applies to the conditional half only
            full = torch.zeros(B, n_img, k.shape[1], device=q.device, dtype=q.dtype)
            full[B // 2:] = bias if B > 1 else bias
            bias = full.repeat_interleave(heads, dim=0)

        scores = torch.baddbmm(
            torch.zeros(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=attn.scale)
        if bias is not None:
            scores = scores + bias
        probs = scores.softmax(dim=-1).to(v.dtype)
        if encoder_hidden_states is not None and self.collect:
            self._accumulate(probs, B, n_img)
        hidden_states = torch.bmm(probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class RegionalSelfAttnProcessor:
    """Self-attention confined to regions. Cross-attention masking controls WHERE identity comes
    from; attn1 controls how it SPREADS. Without this, positions in one region attend to the
    other's features at every layer and step, and two same-class concepts converge to a blend.

    `leak` in [0,1] keeps a fraction of cross-region attention so global coherence (lighting,
    perspective) survives; 0 = full isolation, which is what U+1-pass methods get for free.
    """

    def __init__(self, boxes, leak: float = 0.0, strength: Optional[float] = None,
                 manager=None):
        """`manager` (opcjonalny): gdy podany, ograniczenie zyje tylko dopoki
        `manager.ground_gain > 0`, czyli dzieli harmonogram z kappa. Bez tego twarda izolacja
        w poznych krokach zjada teksture i spojnosc oswietlenia."""
        self.boxes = [b for b in boxes if b is not None]
        self.leak = float(leak)
        self.strength = strength
        self.manager = manager
        self._cache = {}

    def _bias(self, n: int, device, dtype) -> Optional[torch.Tensor]:
        key = (n, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        side = int(round(n ** 0.5))
        bias = None
        # >= 1, nie >= 2: dla JEDNEJ ramki ta sama logika daje dokladnie to, czego trzeba --
        # `same` to wnetrze-wnetrze, `free` to tlo-tlo, a wnetrze<->tlo jest karane. Wartownik
        # ">= 2" byl pisany pod kompozycje i blokowal przypadek jednoobiektowy bez powodu.
        if side * side == n and len(self.boxes) >= 1:
            occ = [v for v in (_region_vec(b, n, device) for b in self.boxes) if v is not None]
            if len(occ) < 1:
                self._cache[key] = None
                return None
            same = torch.zeros(n, n, device=device)
            for m in occ:
                same = torch.maximum(same, torch.outer(m, m))       # ta sama strefa
            covered = torch.stack(occ).amax(0)                       # piksele nalezace do stref
            free = 1.0 - torch.maximum(covered[:, None], covered[None, :]).clamp(0, 1)
            allow = torch.maximum(same, free)                        # wolne tlo laczy wszystko
            pen = (1.0 - allow) * (1.0 - self.leak)
            bias = (-pen * (1e4 if self.strength is None else float(self.strength))).to(dtype)
        self._cache[key] = bias
        return bias

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        if encoder_hidden_states is not None:        # nie nasza sprawa
            encoder_hidden_states = encoder_hidden_states
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        B, n, _ = q.shape
        q = attn.head_to_batch_dim(q); k = attn.head_to_batch_dim(k); v = attn.head_to_batch_dim(v)
        scores = torch.baddbmm(
            torch.zeros(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=attn.scale)
        active = (self.manager is None
                  or float(getattr(self.manager, "ground_gain", 1.0)) > 0)
        bias = self._bias(n, q.device, q.dtype)             if (encoder_hidden_states is None and active) else None
        if bias is not None:
            scores = scores + bias[None]
        probs = scores.softmax(dim=-1).to(v.dtype)
        hidden_states = attn.batch_to_head_dim(torch.bmm(probs, v))
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def set_regional(unet, regions, strength=None, collect: bool = False,
                 confine: bool = False) -> int:
    """Install the regional processor on every attn2; `regions=None` restores defaults."""
    from diffusers.models.attention_processor import AttnProcessor
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "set_processor"):
            mod.set_processor(AttnProcessor() if not regions
                              else RegionalAttnProcessor(regions, strength, collect, confine))
            n += 1
    return n


def set_regional_self(unet, boxes, leak: float = 0.0, strength=None, manager=None) -> int:
    """Install regional SELF-attention on attn1; `boxes=None` restores defaults."""
    from diffusers.models.attention_processor import AttnProcessor
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn1") and hasattr(mod, "set_processor"):
            mod.set_processor(AttnProcessor() if not boxes
                              else RegionalSelfAttnProcessor(boxes, leak, strength, manager))
            n += 1
    return n


class RegionKVAttnProcessor:
    """attn2 with per-region K/V REPLACEMENT (Mix-of-Show region_rewrite, done properly).

    Positions inside region u attend to that region's OWN prompt sequence (with region u's
    adapter applied to its K/V); positions outside any region attend to the global prompt.
    The concept simply does not exist in the conditioning outside its box -- which is the only
    thing that works, because CLIP's causal encoder smears each concept into every subsequent
    token, so masking the concept span can never remove it (measured: audit 2885915).

    One UNet pass; U+1 attention computes in the 16 attn2 layers only.

    regions: [{'hidden': [1,77,768] encoder states of the region prompt,
               'task_idx': int, 'box': bbox or dense mask, 'token_mask': [1,77] or None}]
    manager: hypernet manager -- adapters are swapped per K/V compute.
    """

    def __init__(self, regions, manager):
        self.regions = regions
        self.manager = manager

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        is_cross = encoder_hidden_states is not None
        ctx_g = hidden_states if not is_cross else encoder_hidden_states
        B, n_img, _ = hidden_states.shape

        m = self.manager
        with m.no_lora():                          # ONE global query, as in the reference
            q_g = attn.head_to_batch_dim(attn.to_q(hidden_states))

        def _attend(ctx, task_idx=None, token_mask=None):
            # region identity enters through K/V only; q stays global (reference behaviour)
            if is_cross and task_idx is not None:
                m.set_context(m.canon_pooled[task_idx:task_idx + 1], task_idx=task_idx,
                              token_mask=token_mask)
                m.compute_and_cache_loras()
                k_ = attn.to_k(ctx); v_ = attn.to_v(ctx)
            else:
                with m.no_lora():
                    k_ = attn.to_k(ctx); v_ = attn.to_v(ctx)
            k_ = attn.head_to_batch_dim(k_); v_ = attn.head_to_batch_dim(v_)
            s_ = torch.baddbmm(torch.zeros(q_g.shape[0], q_g.shape[1], k_.shape[1],
                                           device=q_g.device, dtype=q_g.dtype),
                               q_g, k_.transpose(-1, -2), beta=0, alpha=attn.scale)
            return torch.bmm(s_.softmax(dim=-1).to(v_.dtype), v_)

        out = _attend(ctx_g)                       # global: no adapter

        if is_cross and self.regions:
            side = int(round(n_img ** 0.5))
            if side * side == n_img:
                for r in self.regions:
                    vec = _region_vec(r["box"], n_img, hidden_states.device)
                    if vec is None or float(vec.sum()) == 0:
                        continue
                    ctx_r = r["hidden"].to(device=hidden_states.device,
                                           dtype=hidden_states.dtype)
                    if ctx_r.shape[0] == 1 and B > 1:
                        ctx_r = ctx_r.expand(B, -1, -1)
                    o_r = _attend(ctx_r, r["task_idx"], r.get("token_mask"))
                    pm = vec.to(dtype=out.dtype)[None, :, None]
                    out = out * (1 - pm) + o_r * pm       # paste region output inside its box
        out = attn.batch_to_head_dim(out)
        with self.manager.no_lora():               # keep the last region's cache out of to_out
            out = attn.to_out[1](attn.to_out[0](out))
        if ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def set_region_kv(unet, regions, manager) -> int:
    """Install K/V-replacement processors on attn2; `regions=None` restores defaults."""
    from diffusers.models.attention_processor import AttnProcessor
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "set_processor"):
            mod.set_processor(AttnProcessor() if not regions
                              else RegionKVAttnProcessor(regions, manager))
            n += 1
    return n


class GroundedAttnProcessor:
    """attn2 + hypernet-generated grounding (post-box-probe redesign).

    Standard cross-attention runs untouched; on top, each image position i receives
        tanh(gate_l) * sigmoid(<q_i, K e> / sqrt(d)) * V e
    where e = hypernet(concept key, fourier(box)) and K/V are the layer's OWN frozen
    projections. Zero-init gates -> bit-exact F_base at start; position-dependence enters
    through q_i (UNet features carry location), which is the mechanism GLIGEN relies on.
    """

    def __init__(self, attn2_name: str, manager):
        self.name = attn2_name
        self.manager = manager

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            ctx = attn.norm_encoder_hidden_states(ctx)
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(ctx))
        v = attn.head_to_batch_dim(attn.to_v(ctx))
        s = torch.baddbmm(torch.zeros(q.shape[0], q.shape[1], k.shape[1],
                                      device=q.device, dtype=q.dtype),
                          q, k.transpose(-1, -2), beta=0, alpha=attn.scale)
        # `ground_confine`: kara w logicie dla pozycji POZA ramka na tokenach konceptu.
        # Metryka IoU pokazala, ze wstrzyk GSA steruje POLOZENIEM najwyrazistszej czesci,
        # ale nie ROZCIAGLOSCIA obiektu (wypelnienie 1.9-2.4 ramki): dodaje tresc w ramce,
        # a nic nie tlumi konceptu poza nia. Adres jest ten sam analityczny inside(), a
        # harmonogram wspolny z kappa (kara zyje tylko dopoki ground_gain > 0), bo twarde
        # tlumienie w poznych krokach zjada teksture. Pelny kadr => inside=1 => kara = 0,
        # czyli protokol bazowy jest nietkniety.
        conf = float(getattr(self.manager, "ground_confine", 0.0) or 0.0)
        if (conf > 0 and encoder_hidden_states is not None
                and getattr(self.manager, "lora_enabled", True)
                and float(getattr(self.manager, "ground_gain", 1.0)) > 0
                and getattr(self.manager, "_ground_box", None) is not None):
            n_img = s.shape[1]
            ch_, cw_ = (h, w) if ndim == 4 else (int(n_img ** 0.5),) * 2
            if ch_ * cw_ == n_img:
                outside = 1.0 - self.manager.geo_inside(ch_, cw_, s.device, s.dtype)  # [n,1]
                tmask = self.manager.get_token_mask()
                mine = (tmask.reshape(-1)[:s.shape[2]].to(device=s.device, dtype=s.dtype)
                        if tmask is not None
                        else torch.ones(s.shape[2], device=s.device, dtype=s.dtype))
                if getattr(self.manager, "ground_confine_tail", False):
                    # CLIP jest PRZYCZYNOWY: koncept wycieka do kazdego kolejnego tokenu,
                    # wiec kara tylko na jego wlasnych pozycjach zostawia EOS i padding,
                    # ktore niosa cale zdanie (zmierzone w tym repo: audit 2885915).
                    # cummax => kara od pierwszego tokenu konceptu do konca sekwencji.
                    mine = torch.cummax(mine, dim=0).values
                s = s - conf * (outside * mine[None, :]).unsqueeze(0)
        out = torch.bmm(s.softmax(dim=-1).to(v.dtype), v)

        if getattr(self.manager, "ground_gsa", False) and encoder_hidden_states is not None:
            # cond-only: odczyt GSA pomijany w galezi uncond (liczonej pod no_lora()),
            # ale tryb GSA NIGDY nie spada do starej sciezki skalarnej (e ma tu 4 tokeny)
            g = self.manager.get_ground(self.name) \
                if getattr(self.manager, "lora_enabled", True) else None
            if g is not None:
                _, gate, _ = g
                read = self.manager.gsa_read(self.name, hidden_states)
                if read is not None:
                    n_img = hidden_states.shape[1]
                    if ndim == 4:
                        gh, gw = h, w
                    else:
                        gh = gw = int(n_img ** 0.5)
                    inside = self.manager.geo_inside(gh, gw, out.device, out.dtype)  # [n,1]
                    heads = attn.heads
                    read_h = attn.head_to_batch_dim(read)                            # [B*hd, n, d/hd]
                    ins = inside.unsqueeze(0)                                        # [1, n, 1]
                    gain = float(getattr(self.manager, "ground_gain", 1.0))
                    # kappa per ROZDZIELCZOSC mapy: uklad rozstrzyga sie na mapach 8/16,
                    # kolor i tekstura na 32/64. Wstrzyk jednakowy na wszystkich 16 attn2
                    # przy kappa>1 maluje kolorem tam, gdzie mial tylko wskazac miejsce.
                    # None -> mnoznik 1.0 wszedzie, czyli bitowo obecne zachowanie.
                    gres = getattr(self.manager, "ground_gain_res", None)
                    if gres:
                        gain *= float(gres.get(gh, 1.0))
                    out = out + gain * torch.tanh(gate).to(out.dtype) * ins * read_h
            g = None
        else:
            g = self.manager.get_ground(self.name) if encoder_hidden_states is not None else None
        if g is not None:
            e, gate, geo = g
            e = e.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if e.ndim == 2:                                  # [1, dim] -> [1, 1, dim]
                e = e.unsqueeze(1)
            with self.manager.no_lora():                     # frozen projections for grounding
                kg = attn.head_to_batch_dim(attn.to_k(e))    # [heads, 1, d]
                vg = attn.head_to_batch_dim(attn.to_v(e))
            if kg.shape[0] != q.shape[0]:                    # broadcast batch
                r = q.shape[0] // kg.shape[0]
                kg = kg.repeat(r, 1, 1); vg = vg.repeat(r, 1, 1)
            logit = (q * kg).sum(-1, keepdim=True) * attn.scale          # [heads, n_img, 1]
            if geo is not None:                                           # jawna geometria:
                n_img = logit.shape[1]                                    # pozycja z siatki,
                if ndim == 4:
                    gh, gw = h, w
                else:
                    gh = gw = int(n_img ** 0.5)                           # SD: kwadratowe mapy
                gl = self.manager.geo_logit(gh, gw, logit.device, logit.dtype)  # [n, 1]
                logit = logit + gl.unsqueeze(0)                           # broadcast po batch*heads
            out = out + torch.tanh(gate).to(out.dtype) * torch.sigmoid(logit) * vg
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[1](attn.to_out[0](out))
        if ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def set_grounded(unet, manager, enable: bool = True) -> int:
    """Install grounded processors on every attn2; enable=False restores defaults."""
    from diffusers.models.attention_processor import AttnProcessor
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "set_processor"):
            mod.set_processor(GroundedAttnProcessor(name, manager) if enable else AttnProcessor())
            n += 1
    return n
