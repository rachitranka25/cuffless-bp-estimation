"""Generates supplementary figures for the journal paper that don't already
exist elsewhere in the project (the deck's own figures are reused directly).

Run: python3 scripts/make_paper_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import viz
from paths import ROOT

viz.use_style()

OUT = ROOT / "results" / "overview" / "paper_protocol_schematic.png"


def rounded_box(ax, x, y, w, h, color, label, sublabel="", fontsize=9, text_color="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + (0.05 if sublabel else 0), label, ha="center",
            va="center", fontsize=fontsize, color=text_color, weight="bold", zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.09, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, zorder=3)


fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
rows = [
    ("leaky", "Random clips,\nany patient", viz.ORANGE,
     "Train pool", "Test = random\nclips (any patient)", True),
    ("calbased", "Official held-out\nclips, same patients", viz.YELLOW,
     "Train pool", "Test = held-out clips\n(same 1,164 patients)", True),
    ("calfree", "144 unseen\npatients", viz.BLUE,
     "Train pool\n(1,164 patients)", "Test = 144 new\npatients", False),
    ("aami", "116 unseen patients,\nfull BP range", viz.AQUA,
     "Train pool\n(1,164 patients)", "Test = 116 new patients\n(AAMI-range selection)", False),
]

for ax, (key, title, color, train_lbl, test_lbl, overlap) in zip(axes, rows):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=viz.INK, pad=10)
    rounded_box(ax, 0.08, 0.55, 0.84, 0.32, viz.INK_2, train_lbl, fontsize=8.5)
    if overlap:
        rounded_box(ax, 0.08, 0.12, 0.84, 0.32, color, test_lbl, fontsize=8.5)
        arrow = FancyArrowPatch((0.5, 0.55), (0.5, 0.44), arrowstyle="-|>",
                                 mutation_scale=14, color=viz.MUTED, linewidth=1.3,
                                 zorder=4)
        ax.add_patch(arrow)
        ax.text(0.5, 0.495, "overlaps", ha="center", va="center", fontsize=7.5,
                color=viz.MUTED, style="italic", zorder=5,
                bbox=dict(facecolor=viz.SURFACE, edgecolor="none", pad=2))
    else:
        rounded_box(ax, 0.08, 0.12, 0.84, 0.32, color, test_lbl, fontsize=8.5)
        ax.plot([0.5, 0.5], [0.44, 0.55], color=viz.BAD, linewidth=1.8, zorder=4)
        ax.text(0.5, 0.495, "disjoint", ha="center", va="center", fontsize=7.5,
                color=viz.BAD, weight="bold", style="italic", zorder=5,
                bbox=dict(facecolor=viz.SURFACE, edgecolor="none", pad=2))

fig.suptitle("The four evaluation protocols differ only in which patients and "
             "clips are assigned to the test set", fontsize=11.5, weight="bold", y=1.04)
plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)

# --------------------------------------------------------------------------- #
# BP-bin weighting cost/benefit, all five models (the band-error figure
# reused elsewhere only shows ResNet1D; this summarises all five for the
# paper so the figure matches the prose, which discusses every model).
# --------------------------------------------------------------------------- #
OUT2 = ROOT / "results" / "overview" / "paper_weighting_all_models.png"

models = ["Random Forest", "Gradient\nBoosting", "1D-CNN", "ResNet1D", "Transformer"]
overall = [-0.18, 1.92, 2.30, 0.76, 2.77]
above170 = [-4.05, -13.29, -17.52, -6.68, -16.41]
colors = [viz.BLUE, viz.YELLOW, viz.ORANGE, viz.AQUA, viz.VIOLET]

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))

barsL = axL.bar(models, overall, color=colors, width=0.6)
axL.axhline(0, color=viz.INK_2, linewidth=0.8)
axL.set_ylabel("Overall SBP MAE change (mmHg)")
viz.title(axL, "Cost: overall accuracy", "weighted minus unweighted, positive = weighting costs accuracy")
for b, v in zip(barsL, overall):
    axL.text(b.get_x() + b.get_width() / 2, v + (0.08 if v >= 0 else -0.08),
              f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
viz.despine(axL)

barsR = axR.bar(models, above170, color=colors, width=0.6)
axR.axhline(0, color=viz.INK_2, linewidth=0.8)
axR.set_ylabel("Above-170 mmHg SBP MAE change (mmHg)")
viz.title(axR, "Benefit: high-BP coverage", "weighted minus unweighted, negative = weighting reduces error")
for b, v in zip(barsR, above170):
    axR.text(b.get_x() + b.get_width() / 2, v - 0.4, f"{v:.2f}", ha="center",
              va="top", fontsize=8.5)
viz.despine(axR)

fig2.suptitle("BP-bin weighting's cost and benefit, all five models", fontsize=12,
              weight="bold", y=1.03)
plt.tight_layout()
fig2.savefig(OUT2, bbox_inches="tight")
print("saved", OUT2)

# --------------------------------------------------------------------------- #
# Bland-Altman summary, all five models (the agreement plots reused elsewhere
# only cover ResNet1D's raw scatter clouds; this summarises bias and 95% limits
# of agreement for every model so the agreement analysis isn't single-model).
# --------------------------------------------------------------------------- #
import json
import numpy as np
from paths import MODEL_RESULTS

MODEL_ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
MODEL_LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
               "resnet": "ResNet1D", "transformer": "Transformer"}
PROTO_ORDER = ["leaky", "calbased", "calfree", "aami"]


def _predictions(model, protocol):
    p = MODEL_RESULTS / model / f"{model}_{protocol}_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["y_true"], z["y_pred"]


OUT3 = ROOT / "results" / "overview" / "paper_bland_altman_all_models.png"

import matplotlib as mpl

fig3, axes3 = plt.subplots(1, 2, figsize=(11.5, 5.2))
cmap = mpl.colormaps["RdBu_r"]
vmax = 12.0

for ti, (target, ax) in enumerate(zip(("sbp", "dbp"), axes3)):
    k = 0 if target == "sbp" else 1
    bias_grid = np.full((len(MODEL_ORDER), len(PROTO_ORDER)), np.nan)
    limit_grid = np.full((len(MODEL_ORDER), len(PROTO_ORDER)), np.nan)
    for mi, m in enumerate(MODEL_ORDER):
        for pi, p in enumerate(PROTO_ORDER):
            pred = _predictions(m, p)
            if pred is None:
                continue
            y, yp = pred
            diff = yp[:, k] - y[:, k]
            bias_grid[mi, pi] = diff.mean()
            limit_grid[mi, pi] = 1.96 * diff.std(ddof=1)

    im = ax.imshow(bias_grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for mi in range(len(MODEL_ORDER)):
        for pi in range(len(PROTO_ORDER)):
            b, l = bias_grid[mi, pi], limit_grid[mi, pi]
            text_color = "white" if abs(b) > vmax * 0.55 else viz.INK
            ax.text(pi, mi, f"{b:+.1f}\n±{l:.0f}", ha="center", va="center",
                    fontsize=9, color=text_color, linespacing=1.6)
    ax.set_xticks(range(len(PROTO_ORDER)))
    ax.set_xticklabels(PROTO_ORDER)
    ax.set_yticks(range(len(MODEL_ORDER)))
    if ti == 0:
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, len(PROTO_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    viz.title(ax, f"{target.upper()} agreement",
              "cell = bias, ± 95% limits of agreement (mmHg)")

fig3.subplots_adjust(wspace=0.08)
cbar = fig3.colorbar(im, ax=axes3, orientation="horizontal", fraction=0.045, pad=0.1,
                      shrink=0.5)
cbar.set_label("bias, predicted minus true (mmHg)  —  blue = reads low, red = reads high")
fig3.suptitle("Bland-Altman bias and 95% limits of agreement, all five models, all four protocols",
              fontsize=12, weight="bold", y=1.02)
fig3.savefig(OUT3, bbox_inches="tight")
print("saved", OUT3)

# --------------------------------------------------------------------------- #
# CNN + ResNet1D training curves together (the schedule-confound discussion
# covers both architectures but the single-model figure reused elsewhere only
# showed the CNN; this combines both so the figure matches the prose).
# --------------------------------------------------------------------------- #
OUT4 = ROOT / "results" / "overview" / "paper_training_curves_cnn_resnet.png"


def _history(model, protocol):
    p = MODEL_RESULTS / model / f"{model}_{protocol}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("info", {}).get("history")


fig4, axes4 = plt.subplots(2, len(PROTO_ORDER), figsize=(3.5 * len(PROTO_ORDER), 6.8),
                            sharex="col")
for row, m in enumerate(("cnn", "resnet")):
    for col, proto in enumerate(PROTO_ORDER):
        ax = axes4[row][col]
        h = _history(m, proto)
        if not h:
            ax.axis("off")
            continue
        ep = [pt["epoch"] for pt in h]
        tr = [pt["train_loss"] for pt in h]
        va = [pt["val_loss"] for pt in h]
        best = min(h, key=lambda pt: pt["val_loss"])
        ax.plot(ep, tr, color=viz.MUTED, lw=1.6, label="train")
        ax.plot(ep, va, color=viz.BLUE, lw=1.9, label="validation")
        ax.axvline(best["epoch"], ls=":", lw=1.2, color=viz.BAD)
        if row == 0:
            ax.set_title(proto, fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{MODEL_LABEL[m]}\nloss", fontsize=9.5)
            if row == 0:
                ax.legend(loc="upper right", fontsize=7.5)
        if row == 1:
            ax.set_xlabel("epoch")
        viz.despine(ax)

fig4.suptitle("1D-CNN (top) and ResNet1D (bottom) training/validation loss, all four "
              "protocols. Validation plateaus within the first 5-15 epochs for both "
              "architectures.", fontsize=11.5, weight="bold", y=1.02)
plt.tight_layout()
fig4.savefig(OUT4, bbox_inches="tight")
print("saved", OUT4)
