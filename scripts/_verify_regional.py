"""Audyt RegionalAttnProcessor na CPU, bez SD i bez wag.

Sprawdza jedna rzecz, ktora przez dlugi czas byla po cichu zla: kara ukladu ma dzialac
WYLACZNIE w przebiegu warunkowym. Nasze samplery licza cond i uncond oddzielnymi wywolaniami
UNetu, wiec dawny podzial `full[B//2:]` nigdy ich nie rozroznial -- przy B=1 kara ladowala
takze w predykcji bezwarunkowej, a CFG mnozy jej blad przez (1 - guidance).

Uruchomienie: python scripts/_verify_regional.py
"""
import contextlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from src.regional import (RegionalAttnProcessor, RegionKVAttnProcessor,
                          box_to_cxcywh, reset_attn_acc, ATTN_ACC)


class StubAttn(nn.Module):
    """Minimalny odpowiednik diffusers.Attention: tylko to, czego dotyka procesor."""

    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        for name in ("to_q", "to_k", "to_v"):
            setattr(self, name, nn.Linear(dim, dim, bias=False))
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])

    def head_to_batch_dim(self, x):
        b, n, d = x.shape
        return x.reshape(b, n, self.heads, d // self.heads).permute(0, 2, 1, 3) \
                .reshape(b * self.heads, n, d // self.heads)

    def batch_to_head_dim(self, x):
        bh, n, d = x.shape
        b = bh // self.heads
        return x.reshape(b, self.heads, n, d).permute(0, 2, 1, 3).reshape(b, n, self.heads * d)


class StubManager:
    def __init__(self, lora_enabled=True):
        self.lora_enabled = lora_enabled
        self.ground_gsa = False
        self.active = None          # ktory adapter jest "w cache" -- do kontroli 9

    @contextlib.contextmanager
    def no_lora(self):
        prev, prev_a = self.lora_enabled, self.active
        self.lora_enabled, self.active = False, None
        try:
            yield
        finally:
            self.lora_enabled, self.active = prev, prev_a

    def restore_lora(self, snap):
        self.active = snap[0]


def run(regions, manager, **kw):
    torch.manual_seed(0)
    attn = StubAttn()
    proc = RegionalAttnProcessor(regions, manager=manager, **kw)
    torch.manual_seed(1)
    hs = torch.randn(1, 256, 8)                  # 16x16 pozycji obrazu
    ehs = torch.randn(1, 8, 8)                   # 8 tokenow tekstu
    with torch.no_grad():
        return proc(attn, hs, encoder_hidden_states=ehs)


def main():
    tm1 = torch.zeros(1, 8); tm1[0, 1:3] = 1.0           # tokeny konceptu 1
    tm2 = torch.zeros(1, 8); tm2[0, 4:6] = 1.0           # tokeny konceptu 2
    L, R = (0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)
    regions = [(L, tm1, False), (R, tm2, False)]

    plain = run([], None)                                # bez regionow = zwykla uwaga
    cond = run(regions, StubManager(lora_enabled=True))
    uncond = run(regions, StubManager(lora_enabled=False))
    nomgr = run(regions, None)

    d_cond = float((cond - plain).abs().max())
    d_uncond = float((uncond - plain).abs().max())
    d_nomgr = float((nomgr - plain).abs().max())
    print(f"1) kara w przebiegu warunkowym:    max|d| = {d_cond:.6f}  (ma byc > 0)")
    print(f"2) kara w przebiegu bezwarunkowym: max|d| = {d_uncond:.6f}  (ma byc == 0)")
    print(f"3) bez managera (stare zachowanie):max|d| = {d_nomgr:.6f}  (kara wszedzie)")
    ok = d_cond > 1e-6 and d_uncond == 0.0 and d_nomgr > 1e-6

    # 4) zbieranie map uwagi tez jest warunkowe -- inaczej srednia miesza obie galezie
    reset_attn_acc()
    run(regions, StubManager(lora_enabled=False), collect=True)
    empty = len(ATTN_ACC)
    reset_attn_acc()
    run(regions, StubManager(lora_enabled=True), collect=True)
    filled = len(ATTN_ACC)
    print(f"4) akumulacja uwagi: uncond {empty} map (ma byc 0), cond {filled} map (ma byc 2)")
    ok = ok and empty == 0 and filled == 2

    # 5) confine: kara ma spadac na pozycje POZA ramka konceptu, wiec sama geometria
    #    musi zalezec od ramki -- dwie rozne ramki daja rozne wyjscia
    a = run([(L, tm1, False)], StubManager(), confine=True)
    b = run([(R, tm1, False)], StubManager(), confine=True)
    d_geo = float((a - b).abs().max())
    print(f"5) confine zalezy od ramki:        max|d| = {d_geo:.6f}  (ma byc > 0)")
    ok = ok and d_geo > 1e-6

    # 6-7) RegionKVAttnProcessor: ten sam kontrakt cond/uncond, ale w torze JEDNOPRZEBIEGOWYM.
    #      Bez bramki wolajacy musial odinstalowywac procesor wokol przebiegu uncond recznie.
    def run_kv(regions, manager):
        torch.manual_seed(0)
        attn = StubAttn()
        proc = RegionKVAttnProcessor(regions, manager, "mid.attn2", False)
        torch.manual_seed(1)
        hs = torch.randn(1, 256, 8)
        ehs = torch.randn(1, 8, 8)
        with torch.no_grad():
            return proc(attn, hs, encoder_hidden_states=ehs)

    torch.manual_seed(7)
    kv_regions = [{"task_idx": 0, "hidden": torch.randn(1, 8, 8), "box": L,
                   "lora": ("snap", None)}]

    kv_plain = run_kv([], StubManager())
    kv_cond = run_kv(kv_regions, StubManager(lora_enabled=True))
    kv_unc = run_kv(kv_regions, StubManager(lora_enabled=False))
    d_kc = float((kv_cond - kv_plain).abs().max())
    d_ku = float((kv_unc - kv_plain).abs().max())
    print(f"6) region_kv, przebieg warunkowy:  max|d| = {d_kc:.6f}  (ma byc > 0)")
    print(f"7) region_kv, przebieg uncond:     max|d| = {d_ku:.6f}  (ma byc == 0)")
    ok = ok and d_kc > 1e-6 and d_ku == 0.0

    # 8) konwersja ramki zyje w jednym miejscu i liczy to, co trzeba
    got = box_to_cxcywh((0.2, 0.4, 0.6, 1.0))
    exp = (0.4, 0.7, 0.4, 0.6)
    same = all(abs(a - b) < 1e-9 for a, b in zip(got, exp))
    print(f"8) box_to_cxcywh: {tuple(round(x, 4) for x in got)} (ma byc {exp}); "
          f"gesta maska -> {box_to_cxcywh(torch.zeros(4, 4))}")
    ok = ok and same and box_to_cxcywh(torch.zeros(4, 4)) is None
    # 9) KTORE projekcje widza adapter regionu. To jest kontrola, ktora zlapalaby dziure
    #    sprzed poprawki: `to_q` liczylo sie raz globalnie, a `to_out` raz na scalonym
    #    wyjsciu, oba pod no_lora(), wiec adapter regionu dotykal tylko `to_k`/`to_v`.
    attn = StubAttn()
    mgr = StubManager(lora_enabled=True)
    log = []
    for nm in ("to_q", "to_k", "to_v"):
        getattr(attn, nm).register_forward_hook(
            lambda mod, i, o, nm=nm: log.append((nm, mgr.active)))
    attn.to_out[0].register_forward_hook(lambda mod, i, o: log.append(("to_out", mgr.active)))
    proc = RegionKVAttnProcessor([{"task_idx": 0, "hidden": torch.randn(1, 8, 8), "box": L,
                                   "lora": ("snap", None)}], mgr, "mid.attn2", False)
    with torch.no_grad():
        proc(attn, torch.randn(1, 256, 8), encoder_hidden_states=torch.randn(1, 8, 8))
    tlo = {nm for nm, a in log if a is None}
    reg = {nm for nm, a in log if a == "snap"}
    czte = {"to_q", "to_k", "to_v", "to_out"}
    print(f"9) galaz tla    widzi: {sorted(tlo)}  (ma byc bez adaptera, wszystkie 4)")
    print(f"   galaz regionu widzi: {sorted(reg)}  (ma byc {sorted(czte)})")
    ok = ok and tlo == czte and reg == czte
    print("\n" + ("OK" if ok else "BLAD"))
    return 0 if ok else 1


sys.exit(main())
