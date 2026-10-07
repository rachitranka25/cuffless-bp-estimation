"""Figures made from the processed arrays themselves, not from the source .mat.

The point is to see what actually landed in data/processed/pulsedb/*/signals.npy —
if a preprocessing bug ever corrupts a split, it shows up here rather than silently
in a training run.

Run:  python3 scripts/visualize_processed.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import (PROC_PULSEDB, PROC_DALIA, PULSEDB_SUBSETS, DATASET_FIG,  # noqa: E402
                   FIG_PULSEDB, FIG_DALIA, FIG_COMPARE, ROOT)
from fiducials import FS, detect_ppg_fiducials                             # noqa: E402
import viz                                                                 # noqa: E402

viz.use_style()
FIG = DATASET_FIG
for _d in (FIG_PULSEDB, FIG_DALIA, FIG_COMPARE):
    _d.mkdir(parents=True, exist_ok=True)
SPLITS = list(PULSEDB_SUBSETS)
T = np.arange(1250) / FS
RNG = np.random.default_rng(7)


def load(split):
    p = PROC_PULSEDB / split
    return (np.load(p / "signals.npy", mmap_mode="r"),
            np.load(p / "labels.npy"),
            np.load(p / "subjects.npy"),
            pd.read_parquet(p / "meta.parquet"))


def save(fig, name):
    fig.savefig(FIG / name)
    print("saved", (FIG / name).relative_to(ROOT))
    plt.close(fig)


# --------------------------------------------------------------------------- #

def fig_array_heatmap():
    """The array as an image: every row is one segment, colour is amplitude."""
    sig, lab, subj, _ = load("calfree_test")
    order = np.argsort(lab[:, 0])          # sort by SBP so structure is visible
    take = order[np.linspace(0, len(order) - 1, 400).astype(int)]
    take.sort()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4),
                             gridspec_kw={"width_ratios": [1, 1, .5]})
    for ax, ch, name in zip(axes[:2], (0, 1), ("ECG", "PPG")):
        block = np.asarray(sig[take, ch], dtype=np.float32)
        im = ax.imshow(block, aspect="auto", cmap="magma", vmin=0, vmax=1,
                       extent=[0, 10, len(take), 0], interpolation="nearest")
        ax.set_xlabel("seconds")
        ax.set_ylabel("segment (sorted by SBP)" if ch == 0 else "")
        ax.grid(False)
        viz.title(ax, f"{name} channel", "each row is one 10-second segment")
        fig.colorbar(im, ax=ax, pad=.01, fraction=.045).set_label(
            "normalised amplitude", fontsize=8)

    ax = axes[2]
    ax.plot(lab[take, 0], np.arange(len(take)), color=viz.BLUE, lw=1.6, label="SBP")
    ax.plot(lab[take, 1], np.arange(len(take)), color=viz.ORANGE, lw=1.6, label="DBP")
    ax.set_ylim(len(take), 0)
    ax.set_xlabel("mmHg")
    ax.legend(loc="lower right")
    viz.despine(ax)
    viz.title(ax, "Labels", "the target for each row")

    fig.suptitle("Inside calfree_test/signals.npy — the stored array, drawn directly",
                 fontsize=11, x=.02, ha="left", weight="bold")
    plt.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "pulsedb/09_array_as_image.png")


def fig_samples_per_split():
    """One random segment from every split, read out of the .npy."""
    fig, axes = plt.subplots(len(SPLITS), 1, figsize=(12, 9), sharex=True)
    for ax, split in zip(axes, SPLITS):
        sig, lab, subj, _ = load(split)
        i = int(RNG.integers(len(sig)))
        seg = np.asarray(sig[i], dtype=np.float32)
        ax.plot(T, seg[0], color=viz.CH["ecg"], lw=.9, label="ECG")
        ax.plot(T, seg[1], color=viz.CH["ppg"], lw=1.3, label="PPG")
        ax.set_ylabel(split, fontsize=9)
        ax.text(.995, .08, f"{subj[i]} · SBP {lab[i,0]:.0f} / DBP {lab[i,1]:.0f}",
                transform=ax.transAxes, ha="right", fontsize=8, color=viz.INK_2)
        viz.despine(ax)
    axes[0].legend(loc="upper right", ncol=2)
    viz.title(axes[0], "One segment from each split",
              "read from data/processed/pulsedb/<split>/signals.npy")
    axes[-1].set_xlabel("seconds")
    plt.tight_layout()
    save(fig, "pulsedb/08_samples_from_arrays.png")


def fig_mean_beat():
    """Beat-averaged PPG per split — the shape a model actually sees."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    win = int(1.2 * FS)
    for split, col in zip(SPLITS, [viz.BLUE, viz.ORANGE, viz.AQUA, viz.YELLOW, viz.VIOLET]):
        sig, lab, _, _ = load(split)
        idx = RNG.choice(len(sig), min(600, len(sig)), replace=False)
        beats = []
        for i in idx:
            ppg = np.asarray(sig[i, 1], dtype=np.float64)
            fid = detect_ppg_fiducials(ppg)
            for f0 in fid["foot"][:3]:
                if f0 + win <= ppg.size:
                    b = ppg[f0:f0 + win]
                    beats.append((b - b.min()) / (np.ptp(b) + 1e-9))
        if not beats:
            continue
        B = np.array(beats)
        m = B.mean(0)
        axes[0].plot(np.arange(win) / FS, m, color=col, lw=1.8, label=split)
        axes[0].fill_between(np.arange(win) / FS, m - B.std(0), m + B.std(0),
                             color=col, alpha=.10, lw=0)
        axes[1].hist(lab[:, 0], bins=70, range=(50, 220), density=True,
                     histtype="step", lw=1.6, color=col, label=split)
    axes[0].set_xlabel("seconds from pulse onset")
    axes[0].set_ylabel("normalised PPG")
    axes[0].legend(loc="upper right")
    viz.despine(axes[0])
    viz.title(axes[0], "Average pulse shape per split",
              "band is ±1 sd; the splits are morphologically comparable")

    axes[1].set_xlabel("SBP (mmHg)")
    axes[1].set_ylabel("density")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="x", visible=False)
    viz.despine(axes[1])
    viz.title(axes[1], "SBP label distribution",
              "aami_test spans the full range on purpose")
    plt.tight_layout()
    save(fig, "pulsedb/07_pulse_shape_per_split.png")


def fig_split_composition():
    """Subjects, segments and quality per split, from the stored metadata."""
    rows = []
    for split in SPLITS:
        sig, lab, subj, meta = load(split)
        rows.append(dict(split=split, segments=len(sig),
                         subjects=meta.subject.nunique(),
                         gb=(PROC_PULSEDB / split / "signals.npy").stat().st_size / 1e9,
                         q=meta.get("qual_hr_agreement", pd.Series([np.nan])).median(),
                         age=meta.age.median()))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    o = df.sort_values("subjects")
    bars = ax.barh(o.split, o.subjects, color=viz.BLUE, height=.62)
    for b, v in zip(bars, o.subjects):
        ax.text(b.get_width() * 1.03, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=8, color=viz.INK_2)
    ax.set_xlabel("subjects")
    ax.set_xlim(right=o.subjects.max() * 1.25)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, "Subjects per split", "1,553 unique people in total")

    ax = axes[1]
    o = df.sort_values("segments")
    bars = ax.barh(o.split, o.segments, color=viz.ORANGE, height=.62)
    for b, v in zip(bars, o.segments):
        ax.text(b.get_width() * 1.06, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=8, color=viz.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("segments (log)")
    ax.set_xlim(right=o.segments.max() * 4)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, "Segments per split", "10 seconds each")

    ax = axes[2]
    for split, col in zip(SPLITS, [viz.BLUE, viz.ORANGE, viz.AQUA, viz.YELLOW, viz.VIOLET]):
        _, _, _, meta = load(split)
        if "qual_hr_agreement" in meta:
            ax.hist(meta.qual_hr_agreement, bins=40, range=(0, 1), density=True,
                    histtype="step", lw=1.6, color=col, label=split)
    ax.set_yscale("log")
    ax.set_xlabel("ECG/PPG heart-rate agreement")
    ax.set_ylabel("density (log)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    viz.despine(ax)
    viz.title(ax, "Stored signal quality", "kept as a column, never used to drop rows silently")
    plt.tight_layout()
    save(fig, "pulsedb/04_split_composition.png")
    print("\n" + df.round(3).to_string(index=False))


TPL_LEN = 128


def _pulse_template(ppg):
    """One onset-aligned, duration-normalised, amplitude-normalised pulse.

    Removes beat phase, heart rate and gain, so what remains is pulse *shape* —
    the thing being compared. Returns None if no clean beats were found.
    """
    fid = detect_ppg_fiducials(ppg)
    foot, nfoot = fid["foot"], fid["next_foot"]
    if foot.size < 2 or fid["quality"] < 0.8:
        return None
    beats = []
    grid = np.linspace(0, 1, TPL_LEN)
    for f0, f1 in zip(foot, nfoot):
        b = ppg[f0:f1 + 1]
        if b.size < 20 or np.ptp(b) <= 0:
            continue
        beats.append(np.interp(grid, np.linspace(0, 1, b.size),
                               (b - b.min()) / np.ptp(b)))
    if len(beats) < 2:
        return None
    return np.mean(beats, axis=0)


def fig_subject_variation():
    """Why subject-disjoint splitting matters: one subject's PPG is highly consistent."""
    sig, lab, subj, meta = load("calfree_test")
    uniq = pd.unique(subj)
    pick = RNG.choice(uniq, 4, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    cols = [viz.BLUE, viz.ORANGE, viz.AQUA, viz.VIOLET]
    for k, (s, c) in enumerate(zip(pick, cols)):
        idx = np.flatnonzero(subj == s)[:400]
        take = RNG.choice(idx, 6, replace=False)
        for j, i in enumerate(take):
            seg = np.asarray(sig[i, 1], dtype=np.float32)
            axes[0].plot(T[:400], seg[:400] + k * 1.15, color=c, lw=1.0,
                         alpha=.75, label=s if j == 0 else None)
    axes[0].set_yticks([])
    axes[0].set_xlabel("seconds")
    axes[0].legend(loc="upper right", ncol=2)
    viz.despine(axes[0], keep=("bottom",))
    viz.title(axes[0], "Six random segments from each of four subjects",
              "PPG shape is close to a fingerprint — hours apart, still recognisable")

    # Quantify, comparing like with like. A raw trace carries an arbitrary beat
    # phase, so distances between raw traces mostly measure phase, not shape —
    # and averaging unaligned traces flattens the very shape being compared.
    # Each segment is therefore reduced to one onset-aligned, duration-normalised
    # pulse template, and the same template distance is used on both sides.
    templates = {}
    for s in uniq[:80]:
        idx = np.flatnonzero(subj == s)
        take = np.sort(RNG.choice(idx, min(20, idx.size), replace=False))
        tpl = [t for i in take
               if (t := _pulse_template(np.asarray(sig[i, 1], dtype=np.float64))) is not None]
        if len(tpl) >= 4:
            templates[s] = np.array(tpl)

    keys = list(templates)
    within, between = [], []
    for s in keys:
        M = templates[s]
        for a in range(len(M)):
            for b in range(a + 1, len(M)):
                within.append(np.linalg.norm(M[a] - M[b]) / np.sqrt(M.shape[1]))
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            A, B = templates[keys[a]], templates[keys[b]]
            for i in range(min(3, len(A))):
                for j in range(min(3, len(B))):
                    between.append(np.linalg.norm(A[i] - B[j]) / np.sqrt(A.shape[1]))

    ax = axes[1]
    ax.hist(within, bins=50, density=True, color=viz.AQUA, alpha=.85,
            label=f"within subject (median {np.median(within):.3f})")
    ax.hist(between, bins=50, density=True, color=viz.RED, alpha=.85,
            label=f"between subjects (median {np.median(between):.3f})")
    ax.set_xlabel("RMS distance between onset-aligned pulse templates")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    viz.despine(ax)
    viz.title(ax, "Same person vs different people",
              "the separation is why a random split leaks and a subject split does not")
    plt.tight_layout()
    save(fig, "pulsedb/10_subject_fingerprint.png")
    print(f"\nwithin-subject median {np.median(within):.4f} | "
          f"between-subject median {np.median(between):.4f} | "
          f"ratio {np.median(between)/np.median(within):.2f}x")


if __name__ == "__main__":
    fig_array_heatmap()
    fig_samples_per_split()
    fig_mean_beat()
    fig_split_composition()
    fig_subject_variation()
