import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- data ---------------------------------------------------------------
noid = [  # (TA, IA, label, label-offset)
    (0.790, 0.760, "s=0.5", (6, -4)),
    (0.777, 0.776, "s=0.6", (6, -4)),
    (0.758, 0.792, "s=0.7", (7, 2)),
    (0.694, 0.822, "s=1.0", (6, -10)),
]
okey = [
    (0.753, 0.792, "s=0.5", (-38, -2)),
    (0.717, 0.820, "s=0.7", (-38, -2)),
    (0.695, 0.836, "s=0.85", (-46, -2)),
    (0.677, 0.841, "s=1.0", (-40, -2)),
]
refs = [  # CIDM paper, SD-1.5
    ("CIDM", 0.743, 0.780, (6, 4)),
    ("L2DM", 0.750, 0.761, (6, -3)),
    ("CLoRA", 0.736, 0.769, (6, -3)),
    ("EWC", 0.727, 0.759, (6, -3)),
    ("LWF", 0.734, 0.741, (6, -3)),
    ("Finetuning", 0.700, 0.737, (6, -3)),
]
ours_reg = (0.755, 0.772)  # UnHype + von Oswald (bez ortho)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

# dominance region relative to CIDM (both metrics better)
cx, cy = 0.743, 0.780
ax.axvline(cx, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.axhline(cy, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.fill_between([cx, 0.805], cy, 0.86, color=BLUE, alpha=0.05, zorder=0)
ax.annotate("lepiej niż CIDM\nna obu metrykach", xy=(0.7975, 0.799), ha="right", va="top",
            fontsize=9.5, color=INK2, style="italic")

# curves
for pts, color, name in [(noid, BLUE, "nasza — prompty naturalne (V⟨k⟩ tylko routuje), LoRA na tokenach konceptu"),
                         (okey, ORANGE, "nasza — V⟨k⟩ w prompcie, LoRA na całym kontekście")]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=color, lw=2, zorder=3, label=name)
    ax.plot(xs, ys, "o", color=color, ms=8, mec="#fcfcfb", mew=1.5, zorder=4)
    for x, y, lab, (dx, dy) in pts:
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=INK2)

# reference methods (their paper)
for name, x, y, (dx, dy) in refs:
    emph = name == "CIDM"
    ax.plot([x], [y], "D" if emph else "o", ms=9 if emph else 7,
            color="#3a3935" if emph else "#a3a29b", mec="#fcfcfb", mew=1.2, zorder=4)
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=10 if emph else 9, color=INK if emph else INK2,
                fontweight="bold" if emph else "normal")

# our reg-only point (context)
ax.plot([ours_reg[0]], [ours_reg[1]], "s", ms=7, color="#a3a29b", mec="#fcfcfb", mew=1.2, zorder=4)
ax.annotate("UnHype + reg (bez ortho)", ours_reg, xytext=(6, -3), textcoords="offset points",
            fontsize=9, color=INK2)

ax.set_xlabel("Text Alignment (TA / CLIP-T)  →", fontsize=11, color=INK)
ax.set_ylabel("Image Alignment (IA / CLIP-I)  →", fontsize=11, color=INK)
ax.set_title("Trade-off TA–IA na CIFC (SD-1.5); punkty na krzywych = skala LoRA przy inferencji",
             fontsize=12, color=INK, pad=12)
ax.set_xlim(0.665, 0.805); ax.set_ylim(0.730, 0.855)
ax.grid(True, color=GRID, lw=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
leg = ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="#fcfcfb",
                edgecolor=GRID, framealpha=0.95)
fig.tight_layout()
fig.savefig("assets/tradeoff_ta_ia.png", bbox_inches="tight")
print("saved assets/tradeoff_ta_ia.png")
