import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("assets/hookpoint_grid_points.json"))
SCALES = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

MAP2D = list(data.pop("_map2d_a2f").values())
AUG = list(data.pop("_aug_a2f").values())
HEADLINE = data.pop("_headline")["final"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GREYS = ["#8f8e88", "#a9a8a1", "#c1c0b9"]

panelA = [("noid", "attn2 k,v (baza)", BLUE),
          ("noidv", "attn2: tylko v", GREYS[0]),
          ("up", "attn2: tylko up-bloki", GREYS[1]),
          ("dm", "attn2: tylko down+mid", GREYS[2])]
panelB = [("noid", "attn2 k,v (baza)", BLUE),
          ("a2f", "attn2 q,k,v,out", ORANGE),
          ("a1", "attn1 k,v", AQUA),
          ("a12", "attn1+attn2 k,v", YELLOW),
          ("a12f", "attn1+attn2 q,k,v,out", MAGENTA),
          ("ff", "attn2 k,v + FFN", GREEN)]
refs = [("CIDM", 0.743, 0.780)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, series, title in [(axes[0], panelA, "cięcia kanału tekstowego (K/V cross-attn) — nic nie zyskują"),
                          (axes[1], panelB, "dodany kanał obrazowy (q/out, self-attn, FFN) — krzywe wyżej")]:
    ax.set_facecolor(SURF)
    cx, cy = 0.743, 0.780
    ax.axvline(cx, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.axhline(cy, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.fill_between([cx, 0.815], cy, 0.86, color=BLUE, alpha=0.05, zorder=0)
    for key, label, color in series:
        scales = (["0.4"] + SCALES) if (key == "a2f" and ax is axes[1]) else SCALES
        pts = [(data[key][s][0], data[key][s][1]) for s in scales]
        xs, ys = zip(*pts)
        lw = 2.2 if key != "noid" or ax is axes[0] else 1.8
        ax.plot(xs, ys, "-o", color=color, lw=lw, ms=6, mec=SURF, mew=1.2, zorder=3, label=label)
        ax.annotate(f"s={scales[0]}", pts[0], xytext=(5, -9), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate("s=1.0", pts[-1], xytext=(-30, 4), textcoords="offset points", fontsize=8, color=INK2)
    if ax is axes[1]:
        # siatka 2D (s_text x s_img na ckpt a2f): laduje NA krzywej a2f -> galka tekstowa martwa
        ax.plot([p[0] for p in MAP2D], [p[1] for p in MAP2D], "x", color="#8f5a1f", ms=7, mew=1.6,
                ls="", zorder=4, label="a2f: siatka 2D (s_text×s_img)")
        # a2f + augmentacja (osobny trening) -- curve-neutral
        ax.plot([p[0] for p in AUG], [p[1] for p in AUG], "--s", color=ORANGE, alpha=0.55, lw=1.6,
                ms=5, mec=SURF, zorder=3, label="a2f + augmentacja (trening)")
        ax.annotate("headline: 0.762/0.794\nFgt +0.004", (HEADLINE[0], HEADLINE[1]),
                    xytext=(10, 14), textcoords="offset points", fontsize=8.5, color=INK)
    for name, x, y in refs:
        ax.plot([x], [y], "D", ms=9, color="#3a3935", mec=SURF, mew=1.2, zorder=4)
        ax.annotate(name, (x, y), xytext=(6, -12), textcoords="offset points",
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("Text Alignment (TA / CLIP-T)  →", fontsize=10.5, color=INK)
    leg = ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURF,
                    edgecolor=GRID, framealpha=0.95)
axes[0].set_ylabel("Image Alignment (IA / CLIP-I)  →", fontsize=10.5, color=INK)
axes[0].set_xlim(0.655, 0.815); axes[0].set_ylim(0.72, 0.86)
fig.suptitle("Trade-off TA–IA wg punktu wstrzyknięcia LoRA (CIFC, SD-1.5); punkty = skala LoRA przy inferencji",
             fontsize=12.5, color=INK, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("assets/hookpoints_ta_ia.png", bbox_inches="tight")
print("saved assets/hookpoints_ta_ia.png")
