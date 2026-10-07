"""Diagrams for the model-training plan.

Two figures:
  1. which data feeds which model, and where each trained model gets scored
  2. the four evaluation rules side by side, since the difference between them
     is the project's actual subject matter

Run:  python3 scripts/plot_training_plan.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import FIG_PLAN, ROOT     # noqa: E402
import viz                           # noqa: E402

viz.use_style()
OUT = FIG_PLAN
OUT.mkdir(parents=True, exist_ok=True)

DATA = "#dfe3e8"      # a file on disk
MODEL = "#dce9f9"     # a model being trained
SEEN = "#fde4d6"      # scored on patients it has already met
NEW = "#d8f0e6"       # scored on patients it has never met


def save(fig, name):
    fig.savefig(OUT / name)
    print("saved", (OUT / name).relative_to(ROOT))
    plt.close(fig)


def box(ax, x, y, w, h, title, body, fill, fs=9.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fill, ec="#b9c0c8", lw=.9))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
            fontsize=fs + .9, weight="bold", color=viz.INK)
    if body:
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top",
                fontsize=fs, color=viz.INK_2, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#8a8a85", dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.3, color=color,
                                 linestyle=(0, (5, 3)) if dashed else "solid"))


# --------------------------------------------------------------------------- #

def fig_which_data_which_model():
    fig, ax = plt.subplots(figsize=(15.6, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1, 99, "Which data goes into which model, and where each one is scored",
            fontsize=17, weight="bold", va="top", color=viz.INK)
    ax.text(1, 95.3, "Grey = a file on disk.   Blue = a model.   Orange = scored on patients it "
                     "has already met.   Green = scored on patients it never met.",
            fontsize=11, va="top", color=viz.INK_2)

    # ── two representations of the same clips ────────────────────────
    ax.text(45.5, 91.2, "TWO WAYS OF LOOKING AT THE SAME 465,480 CLIPS",
            ha="center", fontsize=11.5, weight="bold", color=viz.INK)

    box(ax, 3, 79.5, 40, 9.5, "data/features/train.parquet",
        "60 numbers per clip — pulse-transit time,\npulse width, area, stiffness index …", DATA)
    box(ax, 48, 79.5, 40, 9.5, "data/processed/pulsedb/train/signals.npy",
        "2 signals x 1250 numbers per clip —\nthe raw ECG and PPG waveform", DATA)

    arrow(ax, 23, 79.3, 23, 75.2)
    arrow(ax, 68, 79.3, 68, 75.2)

    box(ax, 3, 61.5, 40, 13.5, "Model 1  —  classical",
        "Random Forest\nGradient Boosting\n\nreads the 60 numbers,\nnever the waveform", MODEL)
    box(ax, 48, 61.5, 40, 13.5, "Models 2, 3, 4  —  deep",
        "1D-CNN    ·    ResNet1D    ·    Transformer\n\nread the waveform and learn their own\n"
        "features. The Transformer also takes\nage, sex and BMI", MODEL)

    arrow(ax, 23, 61.3, 40, 56.5)
    arrow(ax, 68, 61.3, 51, 56.5)

    box(ax, 27, 47.0, 37, 9.0, "Trained on the SAME clips",
        "train  —  465,480 clips, 1,293 patients\nvalidation held out by patient, for early stopping", MODEL)

    # ── the four ways of scoring ─────────────────────────────────────
    ax.text(45.5, 43.4,
            "THEN SCORED FOUR WAYS — the only thing that changes is which patients are in the test set",
            ha="center", fontsize=11.5, weight="bold", color=viz.INK)

    cols = [(3, 21.5, SEEN, "leaky split",
             "test clips taken at random\nfrom the training pool\n\nSAME patients both sides"),
            (27, 21.5, SEEN, "calbased_test",
             "51,720 clips\n1,293 patients\n\nthe same people as training"),
            (51, 21.5, NEW, "calfree_test",
             "57,600 clips\n144 patients\n\nnever seen — THE HEADLINE"),
            (75, 22, NEW, "aami_cal → aami_test",
             "116 patients, never seen.\nGive the model 1, 3 or 5 of\ntheir readings, then score")]
    for x, w, fill, t, b_ in cols:
        arrow(ax, 45.5, 46.8, x + w / 2, 40.2)
        box(ax, x, 26.0, w, 14.0, t, b_, fill, fs=9.0)

    ax.text(3, 22.6, "what each one is for", fontsize=10.5, weight="bold", color=viz.INK)
    ax.text(3, 19.6,
            "leaky        the number the field usually reports — kept only to measure how much it inflates\n"
            "calbased     an upper bound. Real, but it assumes the device already knows you\n"
            "calfree      the honest number, and the one that goes in the paper\n"
            "aami         the proposed method: a few cuff readings from a new person, then score",
            fontsize=10.2, color=viz.INK_2, va="top", linespacing=1.55)

    # ── external test ────────────────────────────────────────────────
    box(ax, 60, 1.5, 37, 8.0, "Finally  —  PPG-DaLiA",
        "64,682 clips, 15 volunteers, a real watch.\nNo BP labels, so no score — a sanity check", NEW)
    arrow(ax, 61.5, 25.8, 78.5, 9.7, color=viz.AQUA, dashed=True)
    ax.text(3, 9.4,
            "Every model above is trained once, then scored on all four.\n"
            "Same weights, same clips — only the test patients change, so any\n"
            "difference between the four numbers comes from that choice alone.",
            fontsize=10.6, color=viz.INK, va="top", linespacing=1.5)

    plt.tight_layout()
    save(fig, "01_which_data_which_model.png")


def fig_evaluation_rules():
    fig, ax = plt.subplots(figsize=(15.0, 6.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1, 97, "The four ways of scoring, drawn",
            fontsize=17, weight="bold", va="top", color=viz.INK)
    ax.text(1, 89, "Each box is the pool of patients. Blue = used for training. "
                   "Orange = used for testing.",
            fontsize=11.5, va="top", color=viz.INK_2)

    panels = [
        ("leaky", "clips of the same people\nland on both sides", True, False),
        ("calbased_test", "same people, but clips\nthe model never saw", True, False),
        ("calfree_test", "completely different\npeople", False, False),
        ("aami_cal → aami_test", "different people, but the\nmodel is given a few of\ntheir readings first",
         False, True),
    ]
    W, GAP = 21, 3.5
    for i, (name, note, overlap, calib) in enumerate(panels):
        x = 2 + i * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, 40), W, 30, boxstyle="round,pad=0.5,rounding_size=1",
                                    fc="#fbfbfa", ec="#c9ced4", lw=1.0))
        ax.text(x + W / 2, 73.5, name, ha="center", fontsize=11.5, weight="bold", color=viz.INK)

        # training people
        for r in range(3):
            for c in range(6):
                ax.add_patch(plt.Circle((x + 3.2 + c * 2.6, 63 - r * 4.6), 1.0,
                                        fc=viz.BLUE, ec="none", alpha=.85))
        # testing people
        for c in range(6):
            fc = viz.BLUE if overlap else viz.ORANGE
            ec = viz.ORANGE if overlap else "none"
            ax.add_patch(plt.Circle((x + 3.2 + c * 2.6, 46), 1.0, fc=fc, ec=ec,
                                    lw=1.8 if overlap else 0))
        ax.text(x + W / 2, 41.8, "tested on this row", ha="center", fontsize=8.6,
                color=viz.MUTED)
        # the calibration note sits below the panel, not inside it
        if calib:
            ax.text(x + W / 2, 36.0, "+ a few labelled clips from\neach of them, given first",
                    ha="center", fontsize=9.0, color=viz.AQUA, linespacing=1.3, va="top")
        ax.text(x + W / 2, 28.0 if calib else 36.0, note, ha="center", fontsize=9.6,
                color=viz.INK_2, va="top", linespacing=1.35)

    ax.text(2, 13,
            "In the first two the test people are already in the training rows — the model has met them. "
            "In the last two it has not.\n"
            "The gap between the first and the third is what this project is built to measure. The fourth is "
            "the fix it proposes.",
            fontsize=11.2, color=viz.INK, va="top", linespacing=1.6)
    plt.tight_layout()
    save(fig, "02_evaluation_rules.png")


if __name__ == "__main__":
    fig_which_data_which_model()
    fig_evaluation_rules()
