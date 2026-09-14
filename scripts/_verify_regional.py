"""Audyt RegionalAttnProcessor na CPU, bez SD i bez wag.

Sprawdza jedna rzecz, ktora przez dlugi czas byla po cichu zla: kara ukladu ma dzialac
WYLACZNIE w przebiegu warunkowym. Nasze samplery licza cond i uncond oddzielnymi wywolaniami
UNetu, wiec dawny podzial `full[B//2:]` nigdy ich nie rozroznial -- przy B=1 kara ladowala
takze w predykcji bezwarunkowej, a CFG mnozy jej blad przez (1 - guidance).

Uruchomienie: python scripts/_verify_regional.py
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from src.regional import RegionalAttnProcessor, reset_attn_acc, ATTN_ACC


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

    print("\n" + ("OK" if ok else "BLAD"))
    return 0 if ok else 1


sys.exit(main())
