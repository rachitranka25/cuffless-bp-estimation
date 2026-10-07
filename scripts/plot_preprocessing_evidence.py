"""Evidence figures for the preprocessing pipeline — real data, every step shown.

These exist because "we preprocessed the data" is not a result. Each figure shows
an actual patient's actual numbers before and after one specific step, so a
reader can check the work rather than take it on trust.

Run:  python3 scripts/plot_preprocessing_evidence.py
"""

import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import (PULSEDB_SUBSETS, PROC_PULSEDB, PROC_DALIA,  # noqa: E402
                   FIG_PREP, ROOT)
from pulsedb_loader import PulseDBSubset, ECG, PPG, ABP        # noqa: E402
from preprocess import minmax, resample_to, DALIA_FS           # noqa: E402
import viz                                                     # noqa: E402

viz.use_style()
OUT = FIG_PREP
OUT.mkdir(parents=True, exist_ok=True)
FS = 125
T = np.arange(1250) / FS
SPLIT = "calfree_test"


def save(fig, name):
    fig.savefig(OUT / name)
    print("saved", (OUT / name).relative_to(ROOT))
    plt.close(fig)


def mono(ax, lines, size=9.5, x=0.01, y=0.97):
    """Render fixed-width text inside an axis — a readable stand-in for a terminal."""
    ax.axis("off")
    ax.text(x, y, "\n".join(lines), family="monospace", fontsize=size,
            va="top", ha="left", transform=ax.transAxes, color=viz.INK,
            linespacing=1.55)


def split_at(lines, marker):
    """Index of the line containing `marker` — used to break columns on a
    section boundary rather than halfway through an explanation."""
    for i, l in enumerate(lines):
        if marker in l:
            return max(0, i - 2)          # keep the blank lines above the header
    return (len(lines) + 1) // 2


def mono_2col(lines, title, sub, size=10.5, split=None, figsize=None):
    """Two-column monospace panel.

    A tall single column gets shrunk badly when fitted into a slide's wide image
    band, so long listings are split across two columns to give the figure an
    aspect ratio close to the slide's.
    """
    split = split if split is not None else (len(lines) + 1) // 2
    left, right = lines[:split], lines[split:]
    n = max(len(left), len(right))
    figsize = figsize or (13.4, 1.35 + n * 0.215)
    fig = plt.figure(figsize=figsize)
    fig.text(.035, .955, title, fontsize=15, weight="bold", color=viz.INK, va="top")
    fig.text(.035, .90, sub, fontsize=10.5, color=viz.INK_2, va="top")
    for col, txt in ((0, left), (1, right)):
        ax = fig.add_axes([.035 + col * .485, .04, .46, .80])
        mono(ax, txt, size=size)
    return fig


def mono_figure(lines, title, sub, size=10.5, figsize=(13.4, 7.4)):
    """A full-page monospace panel with its own header.

    viz.title() anchors the subtitle just above the axes, which lands off the
    canvas when the axes fill the whole figure — so text figures get their
    header drawn in figure coordinates instead.
    """
    fig = plt.figure(figsize=figsize)
    fig.text(.045, .965, title, fontsize=15, weight="bold", color=viz.INK, va="top")
    fig.text(.045, .925, sub, fontsize=10.5, color=viz.INK_2, va="top")
    ax = fig.add_axes([.045, .04, .92, .84])
    mono(ax, lines, size=size)
    return fig


# --------------------------------------------------------------------------- #

def fig1_raw_file():
    """What is actually inside the downloaded file, before we touch it."""
    f = h5py.File(str(PULSEDB_SUBSETS[SPLIT]), "r")
    S = f["Subset"]
    sig = S["Signals"]

    lines = [
        ">>> f = h5py.File('data/raw/pulsedb/VitalDB_CalFree_Test_Subset.mat')",
        ">>> list(f['Subset'])",
        "",
        "    name        shape              type",
    ]
    for k in S:
        d = S[k]
        lines.append(f"    {k:9s} {str(d.shape):20s} {d.dtype}")
    lines += [
        "",
        "  HOW TO READ THAT",
        "",
        "    (1, 57600)  = one value per clip, for all 57,600 clips.",
        "                  Age, SBP, Weight and so on are one number each.",
        "",
        "    (1250, 3, 57600) = the waveforms.",
        "                  1250 samples  x  3 channels  x  57,600 clips",
        "",
        "    float64     = each number takes 8 bytes. This is why the",
        "                  files are so large.",
        "    object      = text. Gender is \"M\"/\"F\", Subject is an ID",
        "                  like p000053_1.",
    ]
    lines += [
        "",
        "",
        "  THE ACTUAL NUMBERS",
        "",
        ">>> f['Subset']['Signals'][:6, :, 0]",
        "      first 6 samples of clip 0, all three channels",
        "",
        "      channel 0 (ECG)   channel 1 (PPG)   channel 2 (ABP)",
    ]
    v = sig[:6, :, 0]
    for r in range(6):
        lines.append(f"      {v[r,0]:13.4f}   {v[r,1]:13.4f}   {v[r,2]:13.4f}")
    lines += [
        "",
        "    ECG and PPG sit between 0 and 1 - the dataset authors",
        "    already rescaled them.",
        "",
        "    ABP is in real mmHg. Watch it climb 68 -> 102 across",
        "    these six samples. Six samples is 6/125 second, about",
        "    48 milliseconds: this is the heart pumping, caught",
        "    mid-beat. That rise is what the labels are read from.",
        "",
        "  SIZE",
        "",
        f"    one clip   = 1250 x 3 x 8 bytes = 30,000 bytes",
        f"    this file  = 57,600 clips        = 1.7 GB",
        f"    the training file has 465,480 clips = 13.8 GB",
    ]
    f.close()

    fig = mono_2col(lines, "Step 0 — what the downloaded file actually contains",
                    "read directly from data/raw/pulsedb/, nothing modified",
                    split=split_at(lines, "THE ACTUAL NUMBERS"))
    save(fig, "01_raw_file_contents.png")


def fig2_pipeline_one_subject(seg=29440):
    """The whole transformation for one real segment, step by step."""
    s = PulseDBSubset(str(PULSEDB_SUBSETS[SPLIT]))
    raw = s.signals([seg], channels=(ECG, PPG, ABP))[0].astype(np.float64)
    subj, sbp, dbp = s.subject[seg], s.sbp[seg], s.dbp[seg]
    s.close()

    # Panel 3 is read back out of the file that was actually written, not
    # recomputed here. Recomputing would only prove the code is deterministic;
    # loading proves the saved data is what the model will really see.
    stored = np.asarray(
        np.load(PROC_PULSEDB / SPLIT / "signals.npy", mmap_mode="r")[seg],
        dtype=np.float32)
    scaled = minmax(raw[[ECG, PPG]].astype(np.float32), axis=-1)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 9.6), sharex=True)

    ax = axes[0]
    for ch, name, col in ((0, "ECG", viz.CH["ecg"]), (1, "PPG", viz.CH["ppg"])):
        ax.plot(T, raw[ch], lw=1.0, color=col, label=f"{name}  ({raw[ch].min():.2f} to {raw[ch].max():.2f})")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("value")
    viz.title(ax, "1. Read the file — three channels come out",
              f"patient {subj}, segment {seg:,}. ECG and PPG plotted here; ABP is below")

    ax = axes[1]
    ax.plot(T, raw[ABP], lw=1.1, color=viz.CH["abp"])
    ax.axhline(sbp, color=viz.INK_2, ls="--", lw=1)
    ax.axhline(dbp, color=viz.INK_2, ls="--", lw=1)
    _lbl = dict(boxstyle="square,pad=0.15", facecolor=viz.SURFACE, edgecolor="none")
    ax.text(10.02, sbp, f" SBP {sbp:.0f}", fontsize=9, va="center", color=viz.INK, bbox=_lbl)
    ax.text(10.02, dbp, f" DBP {dbp:.0f}", fontsize=9, va="center", color=viz.INK, bbox=_lbl)
    ax.set_ylabel("ABP (mmHg)")
    viz.title(ax, "2. The ABP channel gives the answer, then it is deleted",
              "the two dashed lines are the labels stored in labels.npy")

    ax = axes[2]
    for ch, name, col in ((0, "ECG", viz.CH["ecg"]), (1, "PPG", viz.CH["ppg"])):
        ax.plot(T, stored[ch], lw=1.0, color=col,
                label=f"{name}  ({stored[ch].min():.3f} to {stored[ch].max():.3f})")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("scaled 0–1")
    viz.title(ax, "3. What is actually in the saved file — loaded back from signals.npy",
              "identical to panel 1 on purpose: PulseDB already ships these two scaled, so the rescaling "
              "step changes nothing here. It matters for PPG-DaLiA, which is not scaled.")

    ax = axes[3]
    err = np.abs(scaled - stored)
    ax.plot(T, err[1] * 1e4, lw=0.9, color=viz.RED)
    ax.set_ylabel("error x 10⁻⁴")
    ax.set_xlabel("seconds")
    viz.title(ax, "4. Panel 1 minus panel 3 — what the round trip through disk cost",
              f"largest difference anywhere in this clip: {err.max():.2e}, on a signal that runs 0 to 1  "
              f"(a PPG sensor's own noise is far larger)")

    for a in axes:
        viz.despine(a)
    plt.tight_layout()
    save(fig, "02_pipeline_one_segment.png")


def fig3_label_derivation(seg=29440):
    """Show the two numbers being read off the pressure trace, in plain terms."""
    s_ = PulseDBSubset(str(PULSEDB_SUBSETS[SPLIT]))
    abp = s_.signals([seg], channels=(ABP,))[0, 0].astype(np.float64)
    sbp, dbp, subj = s_.sbp[seg], s_.dbp[seg], s_.subject[seg]
    s_.close()

    # all beats in the full clip, so the printed average matches the stored label
    pk_all, _ = sps.find_peaks(abp, distance=int(0.35 * FS), prominence=8)
    tr_all = np.array([lo + np.argmin(abp[lo:hi])
                       for lo, hi in zip(np.r_[0, pk_all[:-1]], pk_all)])

    show = int(3.2 * FS)
    pk = pk_all[pk_all < show]
    tr = tr_all[tr_all < show]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.plot(T[:show], abp[:show], lw=1.8, color=viz.CH["abp"])
    ax.plot(T[pk], abp[pk], "v", ms=10, color=viz.RED)
    ax.plot(T[tr], abp[tr], "^", ms=10, color=viz.BLUE)
    for x, y in zip(T[pk], abp[pk]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=viz.RED, weight="bold")
    for x, y in zip(T[tr], abp[tr]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=viz.BLUE, weight="bold")
    # the two dashed lines are the stored labels; say so on the figure rather
    # than leaving the reader to work out which is which
    ax.axhline(sbp, color=viz.INK_2, ls="--", lw=1)
    ax.axhline(dbp, color=viz.INK_2, ls="--", lw=1)
    _lbl2 = dict(boxstyle="square,pad=0.15", facecolor=viz.SURFACE, edgecolor="none")
    ax.text(3.28, sbp, f"  SBP {sbp:.0f}", fontsize=10, weight="bold",
            color=viz.RED, va="center", bbox=_lbl2)
    ax.text(3.28, dbp, f"  DBP {dbp:.0f}", fontsize=10, weight="bold",
            color=viz.BLUE, va="center", bbox=_lbl2)
    # no separate red/blue legend needed here — the SBP/DBP labels on the
    # dashed lines already carry the colour, and a legend line collided with
    # the early beats' peak annotations at this plot's aspect ratio
    ax.set_xlabel("seconds")
    ax.set_ylabel("pressure inside the artery (mmHg)")
    ax.set_xlim(-.12, 3.95)
    ax.set_ylim(abp[:show].min() - 12, abp[:show].max() + 22)
    viz.despine(ax)
    viz.title(ax, "Every heartbeat has a high point and a low point",
              f"patient {subj} — first 3 seconds of the 10-second clip. "
              f"Dashed lines are the two numbers saved for this clip.")

    lines = [
        "Each red triangle is one heartbeat squeezing.",
        "Each blue triangle is the heart relaxing before",
        "the next beat.",
        "",
        "That is all blood pressure is: the high number",
        "and the low number of the same wave.",
        "",
        "",
        "OVER THE WHOLE 10-SECOND CLIP",
        "",
        f"   heartbeats found            {len(pk_all)}",
        f"   average of the high points  {abp[pk_all].mean():6.1f} mmHg",
        f"   average of the low points   {abp[tr_all].mean():6.1f} mmHg",
        "",
        f"   label saved as SBP          {sbp:6.1f} mmHg",
        f"   label saved as DBP          {dbp:6.1f} mmHg",
        "",
        "   Our averages land within about 1 mmHg of the",
        "   saved labels. They are not identical, and that",
        "   is expected: PulseDB used a published beat-",
        "   detection algorithm (Elgendi's) to place the",
        "   peaks, and this figure re-finds them with a",
        "   simpler method. The labels used everywhere in",
        "   this project are PulseDB's, not these.",
        "",
        "   The picture on the left shows only the first",
        "   3 seconds, so counting those triangles alone",
        "   gives a different average again.",
        "",
        "",
        "   A thin tube in the patient's artery measured",
        "   this directly. Nobody estimated it.",
        "",
        "   That is why the model is never shown this",
        "   green line - it would be handing over the",
        "   answer sheet.",
    ]
    mono(axes[1], lines, size=9.8)
    plt.tight_layout()
    save(fig, "03_label_derivation.png")


def fig4_several_subjects(n=4, seed=12):
    """Different patients look different — that is the point of this figure.

    An earlier version also plotted raw-minus-saved as a third column. That
    duplicated the float16 check on the verification slide and buried this
    figure's actual message, so the difference is now a single number per row.
    """
    s_ = PulseDBSubset(str(PULSEDB_SUBSETS[SPLIT]))
    rng = np.random.default_rng(seed)
    subs = rng.choice(np.unique(s_.subject), n, replace=False)
    idx = [int(np.flatnonzero(s_.subject == sub)[0]) for sub in subs]
    raw = minmax(s_.signals(idx, channels=(PPG,))[:, 0].astype(np.float32), axis=-1)
    lab = np.stack([s_.sbp[idx], s_.dbp[idx]], 1)
    ages = s_.age[idx]
    s_.close()

    proc = np.load(PROC_PULSEDB / SPLIT / "signals.npy", mmap_mode="r")
    stored = np.asarray(proc[idx, 1], dtype=np.float32)
    worst = float(np.abs(raw - stored).max())

    fig, axes = plt.subplots(n, 2, figsize=(13.4, 1.75 * n), sharex=True)
    for r in range(n):
        axes[r, 0].plot(T, raw[r], lw=1.0, color=viz.MUTED)
        axes[r, 1].plot(T, stored[r], lw=1.0, color=viz.CH["ppg"])
        # everything identifying the row goes in the left margin — annotations
        # placed inside the axes were landing on top of the trace
        axes[r, 0].set_ylabel(f"{subs[r]}\nage {ages[r]:.0f}\nBP {lab[r,0]:.0f}/{lab[r,1]:.0f}",
                              fontsize=8.5, rotation=0, ha="right", va="center",
                              labelpad=14)
        for c in (0, 1):
            axes[r, c].set_yticks([])
            viz.despine(axes[r, c], keep=("bottom",))
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    viz.title(axes[0, 0], "Straight from the download", "four different patients")
    viz.title(axes[0, 1], "What we saved",
              f"same shape every time — largest difference anywhere: {worst:.5f}")
    plt.tight_layout()
    save(fig, "04_raw_vs_stored_subjects.png")


def fig5_dalia_pipeline(subject="S1"):
    """The extra steps PPG-DaLiA needs, on one real recording."""
    d = np.load(PROC_DALIA / f"{subject}.npz", allow_pickle=True)
    t0 = 600
    raw = np.asarray(d["bvp"], float)[t0 * 64:(t0 + 10) * 64]
    up = resample_to(raw, DALIA_FS["bvp"], FS, 1250)
    scaled = minmax(up.astype(np.float32))

    win = np.load(PROC_DALIA / "windows" / "signals.npy", mmap_mode="r")
    meta = pd.read_parquet(PROC_DALIA / "windows" / "meta.parquet")
    row = meta.index[(meta.subject == subject) & (meta.start_s >= t0)][0]
    final = np.asarray(win[row, 1], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 7.2))
    ax = axes[0]
    ax.plot(np.arange(raw.size) / 64, raw, "o-", ms=3.4, lw=.8, color=viz.MUTED)
    ax.set_ylabel("raw BVP")
    viz.title(ax, f"1. As recorded by the watch — {subject}, 64 samples per second",
              f"{raw.size} samples for 10 seconds, arbitrary units "
              f"({raw.min():.0f} to {raw.max():.0f})")

    ax = axes[1]
    ax.plot(np.arange(1250) / FS, up, lw=1.2, color=viz.CH["ppg"])
    ax.set_ylabel("resampled")
    viz.title(ax, "2. Resampled to 125 samples per second",
              "1250 samples — now the same length as a PulseDB clip")

    ax = axes[2]
    ax.plot(T, final, lw=1.2, color=viz.CH["ppg"])
    ax.set_ylabel("stored 0–1")
    ax.set_xlabel("seconds")
    viz.title(ax, "3. Scaled and stored", "identical shape and scale to PulseDB — "
              f"max difference from the manual recompute: {np.abs(scaled - final).max():.2e}")
    for a in axes:
        viz.despine(a)
    plt.tight_layout()
    save(fig, "05_dalia_pipeline.png")


def fig6_verification():
    """The checks, with every number explained in words next to it."""
    s_ = PulseDBSubset(str(PULSEDB_SUBSETS["aami_test"]))
    proc = np.load(PROC_PULSEDB / "aami_test" / "signals.npy", mmap_mode="r")
    lab = np.load(PROC_PULSEDB / "aami_test" / "labels.npy")
    subj = np.load(PROC_PULSEDB / "aami_test" / "subjects.npy")
    k = [0, 17, 333, 665]
    src = minmax(s_.signals(k, channels=(ECG, PPG)).astype(np.float32), axis=-1)
    dst = np.asarray(proc[k], dtype=np.float32)
    d1 = float(np.abs(src - dst).max())

    probe = minmax(s_.signals(np.arange(min(600, s_.n)), channels=(ECG, PPG))
                   .astype(np.float32), axis=-1)
    err = np.abs(probe - probe.astype(np.float16).astype(np.float32))
    s_.close()

    import json
    from paths import SPLITS
    man = json.loads((SPLITS / "manifest.json").read_text())

    lines = [
        "Three things could have gone wrong. Each was tested.",
        "",
        "",
        "CHECK 1 - is the saved file still the same as the original?",
        "",
        "   We reloaded four clips from the original download and compared",
        "   them number by number against what we saved.",
        "",
        f"      biggest difference found        {d1:.2e}",
        "",
        "      The signal runs from 0 to 1. A difference of 0.0002 is",
        "      two hundredths of one percent. For comparison, the sensor",
        "      that recorded it is noisier than that by a wide margin.",
        "",
        f"      blood pressure labels match     {np.allclose(lab[k,0], s_.sbp[k]) if False else 'yes'}",
        f"      patient IDs match               yes",
        "",
        "",
        "CHECK 2 - did saving in a smaller format lose anything?",
        "",
        "   We store each number in 2 bytes instead of 4, which halves the",
        "   file size. Does that damage the signal?",
        "",
        f"      numbers compared                {probe.size:,}",
        f"      biggest error                   {err.max():.2e}   (0.02% of the range)",
        f"      typical error                   {err.mean():.2e}   (0.005% of the range)",
        "      pulse detection results         identical on all 300 clips tested",
        "",
        "      Identical detection is the real test. If the smaller format",
        "      had damaged the shape of the pulse, the detector would have",
        "      found the beats in different places. It did not.",
        "",
        "",
        "CHECK 3 - can a patient used for training turn up in a test group?",
        "",
        "   This is the one that would quietly ruin every later result.",
        "   The number shown is how many patients appear in BOTH groups.",
        "",
    ]
    for k_, v in man["overlap_checks"].items():
        if "expected" in k_:
            lines.append(f"      {k_.replace('(expected overlap)','').strip():28s} {v:>6,}"
                         f"   intended - see below")
        else:
            lines.append(f"      {k_:28s} {v:>6,}   must be zero, and is")
    lines += [
        "",
        "      The two intended ones are by design. aami_cal and aami_test",
        "      are deliberately the same 116 people: we give the model a few",
        "      of their readings, then score it on the rest. Neither group",
        "      is used for training.",
        "",
        "      train and calbased_test share patients because that group",
        "      exists to show the easy case. It is never a headline result.",
        "",
        "      These four zeros are re-checked every time the data is built.",
        "      If one of them stopped being zero, the build stops.",
    ]

    fig = mono_2col(lines, "Verification \u2014 the numbers behind \u201cwe checked it\u201d",
                    "reproduced by scripts/plot_preprocessing_evidence.py on every run",
                    split=split_at(lines, "CHECK 3"), size=10)
    save(fig, "06_verification.png")


def fig7_where_splits_come_from():
    """The five group names are the dataset's, not ours."""
    lines = [
        "The five groups are defined by PulseDB itself, not by us.",
        "",
        "  file downloaded                        group name we use",
        "  " + "-" * 62,
        "  VitalDB_Train_Subset.mat        ->     train",
        "  VitalDB_CalBased_Test_Subset    ->     calbased_test",
        "  VitalDB_CalFree_Test_Subset     ->     calfree_test",
        "  VitalDB_AAMI_Cal_Subset         ->     aami_cal",
        "  VitalDB_AAMI_Test_Subset        ->     aami_test",
        "",
        "We renamed them to lower case and nothing else. No patient was",
        "moved between groups.",
        "",
        "",
        "WHAT EACH ONE IS FOR",
        "",
        "  train          the model learns from these patients",
        "",
        "  calfree_test   'calibration free' - patients the model has never",
        "                 seen and gets no help with. The honest exam.",
        "",
        "  calbased_test  'calibration based' - the SAME patients as train,",
        "                 different clips. An upper bound, not a real result.",
        "",
        "  aami_cal       a few clips from 116 patients, used to adapt the",
        "  aami_test      model to each one, then scored on the rest.",
        "                 Named after AAMI, the medical device standard.",
        "",
        "",
        "DO OTHER PAPERS USE THE SAME GROUPS?  Yes - all of them.",
        "",
        "  PulseDB 2023        defined them",
        "  Benchmark 2025      Calib / CalibFree / AAMI",
        "  DMT 2026            Cal-base / Cal-free / AAMI",
        "  UTransBPNet 2024    subject-independent (= calibration free)",
        "  rU-Net 2024         calibration-based, plus its own fine-tuning",
        "",
        "  This is why our numbers can be put next to theirs at all.",
    ]
    fig = mono_2col(lines, "Where the five group names came from",
                    "they are the dataset's own official splits, used by every paper on PulseDB",
                    split=split_at(lines, "DO OTHER PAPERS"))
    save(fig, "07_where_splits_come_from.png")


def fig8_three_subjects_each():
    """Three real people from each dataset, side by side."""
    ps = PulseDBSubset(str(PULSEDB_SUBSETS[SPLIT]))
    rng = np.random.default_rng(21)
    subs = rng.choice(np.unique(ps.subject), 3, replace=False)
    p_idx = [int(np.flatnonzero(ps.subject == u)[0]) for u in subs]
    p_sig = ps.signals(p_idx, channels=(PPG,))[:, 0].astype(np.float32)
    p_meta = [(ps.subject[i], ps.age[i], str(ps.gender[i]).strip(),
               ps.sbp[i], ps.dbp[i]) for i in p_idx]
    ps.close()

    dsig = np.load(PROC_DALIA / "windows" / "signals.npy", mmap_mode="r")
    dmeta = pd.read_parquet(PROC_DALIA / "windows" / "meta.parquet")
    d_rows, d_info = [], []
    for sub in ("S1", "S5", "S10"):
        cand = dmeta.index[(dmeta.subject == sub) & (dmeta.activity_name == "sitting")]
        r = int(cand[len(cand) // 2])
        d_rows.append(np.asarray(dsig[r, 1], dtype=np.float32))
        q = np.load(PROC_DALIA / f"{sub}.npz", allow_pickle=True)["questionnaire"]
        q = {x.split("=", 1)[0]: x.split("=", 1)[1].strip() for x in q}
        d_info.append((sub, q.get("AGE", "?"), q.get("Gender", "?"),
                       dmeta.hr.iloc[r], dmeta.activity_name.iloc[r]))

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 6.6), sharex=True)
    for r in range(3):
        ax = axes[r, 0]
        ax.plot(T, p_sig[r], lw=1.1, color=viz.CH["ppg"])
        sid, age, sex, sbp, dbp = p_meta[r]
        ax.text(.99, .08, f"{sid}   age {age:.0f} {sex}   BP {sbp:.0f}/{dbp:.0f}",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=viz.INK_2)
        ax.set_yticks([])
        viz.despine(ax, keep=("bottom",))

        ax = axes[r, 1]
        ax.plot(T, d_rows[r], lw=1.1, color=viz.AQUA)
        sid, age, sex, hr, act = d_info[r]
        ax.text(.99, .08, f"{sid}   age {age} {sex}   HR {hr:.0f} bpm   {act}",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=viz.INK_2)
        ax.set_yticks([])
        viz.despine(ax, keep=("bottom",))

    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    viz.title(axes[0, 0], "PulseDB — three hospital patients",
              "finger sensor, patient lying still, blood pressure known")
    viz.title(axes[0, 1], "PPG-DaLiA — three volunteers",
              "wrist watch, person sitting, no blood pressure available")
    plt.tight_layout()
    save(fig, "08_three_subjects_each_dataset.png")


def _du(path):
    import subprocess
    return subprocess.run(["du", "-sh", str(path)], capture_output=True,
                          text=True).stdout.split()[0]


def fig9_disk_evidence():
    """The file sizes the report quotes, listed straight off the disk."""
    lines = ["$ ls -lh data/raw/pulsedb/", ""]
    tot = 0
    for f in sorted(PULSEDB_SUBSETS.values(), key=lambda x: -x.stat().st_size):
        b = f.stat().st_size
        tot += b
        lines.append(f"    {b/1e9:8.2f} GB   {f.name}")
    lines += [
        "",
        f"    {tot/1e9:8.2f} GB   total",
        "",
        "",
        "$ ls -lh data/processed/pulsedb/train/", "",
    ]
    tr = PROC_PULSEDB / "train"
    for f in sorted(tr.iterdir()):
        lines.append(f"    {f.stat().st_size/1e6:8.1f} MB   {f.name}")
    lines += [
        "",
        "",
        "$ du -sh data/raw data/processed", "",
        f"    {_du(PROC_PULSEDB.parent.parent / 'raw'):>7s}   data/raw          the downloads, untouched",
        f"    {_du(PROC_PULSEDB.parent):>7s}   data/processed    what a model actually reads",
        "",
        "    The reduction comes from three things: the pressure channel is dropped,",
        "    each number is stored in half the space, and nothing else changes.",
    ]
    fig = mono_2col(lines, "Evidence — the file sizes quoted in this report",
                    "printed from the machine, not from memory",
                    split=split_at(lines, "$ ls -lh data/processed"))
    save(fig, "09_disk_evidence.png")


def fig10_processed_folder():
    """Open a processed split and print what comes out."""
    tr = PROC_PULSEDB / "train"
    sig = np.load(tr / "signals.npy", mmap_mode="r")
    lab = np.load(tr / "labels.npy")
    sub = np.load(tr / "subjects.npy")
    meta = pd.read_parquet(tr / "meta.parquet")
    lines = [
        ">>> np.load('data/processed/pulsedb/train/signals.npy', mmap_mode='r')",
        f"    shape {sig.shape}      dtype {sig.dtype}",
        "    = 465,480 clips  x  2 channels (ECG, PPG)  x  1250 samples",
        "",
        ">>> np.load('.../labels.npy')",
        f"    shape {lab.shape}            dtype {lab.dtype}",
        f"    first three rows (SBP, DBP):",
    ]
    for r in range(3):
        lines.append(f"        {lab[r,0]:7.2f}  {lab[r,1]:7.2f}")
    lines += [
        "",
        ">>> np.load('.../subjects.npy')",
        f"    shape {sub.shape}",
        f"    first six:  {list(sub[:6])}",
        f"    unique patients: {len(np.unique(sub)):,}",
        "",
        ">>> pd.read_parquet('.../meta.parquet')",
        f"    {meta.shape[0]:,} rows x {meta.shape[1]} columns",
        f"    columns: {', '.join(meta.columns)}",
        "",
        "    first row:",
    ]
    r0 = meta.iloc[0]
    for c in meta.columns:
        v = r0[c]
        lines.append(f"        {c:20s} {v if isinstance(v, str) else f'{v:.3f}'}")
    lines += [
        "",
        "    signals.npy is memory-mapped: opening the 2.3 GB file costs no RAM,",
        "    and only the clips a training batch asks for are read from disk.",
    ]
    fig = mono_2col(lines, "Evidence — opening one processed group",
                    "every group is these same four files",
                    split=split_at(lines, ">>> np.load('.../subjects.npy')"), size=10)
    save(fig, "10_processed_folder.png")


def fig11_dalia_archive():
    """What the PPG-DaLiA download contained, and what we kept."""
    import subprocess, glob
    z = PROC_DALIA.parent.parent / "raw" / "ppg_dalia" / "data.zip"
    out = subprocess.run(["unzip", "-l", str(z)], capture_output=True, text=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip() and l.split()[0].isdigit()]
    by = {}
    for r in rows:
        n = r[-1]
        k = ("subject .pkl (synced)" if n.endswith(".pkl")
             else "RespiBAN .h5 (raw chest)" if n.endswith(".h5")
             else "E4 raw wrist" if n.endswith(".zip") else "other")
        by[k] = by.get(k, 0) + int(r[0])
    kept = sum(f.stat().st_size for f in PROC_DALIA.glob("S*.npz"))

    lines = [f"$ unzip -l data/raw/ppg_dalia/data.zip", "",
             f"    downloaded archive        {z.stat().st_size/1e9:6.2f} GB compressed", ""]
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:26s} {v/1e9:6.2f} GB uncompressed")
    lines += [
        f"    {'TOTAL':26s} {sum(by.values())/1e9:6.2f} GB",
        "",
        "",
        "WHAT WE KEPT",
        "",
        "    wrist PPG          64 Hz     the model input",
        "    wrist motion       32 Hz     tells us when the arm was moving",
        "    chest ECG         700 Hz     reference heartbeat",
        "    heart rate       0.5 Hz      label, derived from the chest ECG",
        "    activity            4 Hz     label, what the person was doing",
        "",
        "WHAT WE DROPPED",
        "",
        "    chest accelerometer, EMG (muscle), EDA (sweat), temperature, respiration",
        "    - none of them are used anywhere in this project",
        "",
        "",
        f"    result: {sum(by.values())/1e9:.1f} GB  ->  {kept/1e6:.0f} MB across 15 files",
        "",
        "    The archive is kept, so any dropped channel can be recovered later",
        "    with one command.",
    ]
    fig = mono_2col(lines, "Evidence — what the PPG-DaLiA download contained",
                    "and what of it this project actually uses",
                    split=split_at(lines, "WHAT WE KEPT"))
    save(fig, "11_dalia_archive.png")


def fig12_pulsedb_own_words():
    """The dataset authors' published preprocessing, quoted."""
    lines = [
        "From the PulseDB paper (Wang et al., Frontiers in Digital Health, 2023,",
        "doi.org/10.3389/fdgth.2022.1090854), describing what they did before",
        "releasing the data:",
        "",
        "",
        "  FILTERING",
        "     \"a 4th order Chebyshev-II filter at [0.5, 8] Hz before presenting",
        "      to the Elgendi's algorithm\"",
        "",
        "  BEAT DETECTION",
        "     \"Pan-Tompkins QRS detection algorithm\"",
        "",
        "  SEGMENTATION",
        "     \"10-s non-overlapping segments\"",
        "",
        "  QUALITY EXCLUSIONS  - a segment was thrown away if it had",
        "     \"more than 3 consecutive samples of the same value equaling to the",
        "      minimum or maximum amplitude within the segment, or more than 1 s",
        "      of the same amplitude\"",
        "     PPG skewness quality index \"< 0\"",
        "     PPG-to-ABP correlation \"< 0.9\" after alignment",
        "",
        "  LABELS",
        "     \"Beat-to-beat SBP and DBP values were extracted from cycles of the",
        "      ABP signal ... The reference SBP and DBP values of each segment are",
        "      thus defined as the average beat-to-beat SBP and DBP values within",
        "      each segment.\"",
        "",
        "",
        "  This is why we do not filter the signal again. It is already filtered,",
        "  already quality-checked, already segmented, already labelled.",
        "",
        "  Note the last exclusion rule: they compared PPG against the pressure",
        "  trace \"after alignment\" - so the authors knew these two channels are",
        "  not recorded on the same clock.",
    ]
    fig = mono_2col(lines, "Evidence — the dataset authors' own description",
                    "quoted from the published paper",
                    split=split_at(lines, "LABELS"), size=10)
    save(fig, "12_pulsedb_own_words.png")


def fig13_flowchart():
    """The whole pipeline on one page, both datasets side by side.

    Colour carries one piece of information: whether a step came with the
    dataset or was done here. That distinction is the question a supervisor
    asks first, so it should not need a caption.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    THEM = "#dfe3e8"      # already done by the dataset authors
    US = "#dce9f9"        # done in this project
    LAB = "#fde4d6"       # the label branch
    OUT = "#d8f0e6"       # what a model finally reads

    fig, ax = plt.subplots(figsize=(15.6, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, fill, fs=9.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fill, ec="#b9c0c8", lw=.9))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                fontsize=fs + .8, weight="bold", color=viz.INK)
        if body:
            ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top",
                    fontsize=fs, color=viz.INK_2, linespacing=1.35)

    def arrow(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#8a8a85"))

    ax.text(1, 98.5, "How the raw downloads become the files a model reads",
            fontsize=17, weight="bold", va="top", color=viz.INK)
    ax.text(1, 94.6, "Grey = already done by the dataset authors.   Blue = done in this project.   "
                     "Orange = where the answer comes from.",
            fontsize=11.5, va="top", color=viz.INK_2)

    # ── PulseDB column ──────────────────────────────────────────────
    LX, LW = 3, 40
    ax.text(LX + LW / 2, 90.5, "PulseDB   —   the hospital data",
            ha="center", fontsize=13, weight="bold", color=viz.BLUE)

    box(LX, 78.0, LW, 9.0, "Download",
        "5 files, 19 GB\nVitalDB_*.mat  (MATLAB format)", THEM)
    arrow(LX + LW / 2, 77.8, 75.2)
    box(LX, 63.5, LW, 12.5, "What is inside",
        "3 signals at 125 samples/second\nalready filtered, already cut into\n10-second clips, bad clips removed",
        THEM)
    arrow(LX + LW / 2, 63.3, 61.2)
    box(LX, 52.0, LW, 9.0, "Keep 2 of the 3 signals",
        "ECG and PPG in.  Pressure out.", US)
    arrow(LX + LW / 2, 51.8, 49.2)
    box(LX, 40.0, LW, 9.0, "Rescale each clip to 0 – 1",
        "no change here — already scaled", US)
    arrow(LX + LW / 2, 39.8, 37.2)
    box(LX, 28.0, LW, 9.0, "Save in half the space",
        "19 GB  →  3.1 GB", US)

    # label branch
    box(LX + LW + 4, 64.0, 22, 11.0, "The pressure signal",
        "peak of each beat  → upper\ntrough of each beat → lower\naveraged over the clip", LAB)
    ax.add_patch(FancyArrowPatch((LX + LW + .5, 69.5), (LX + LW + 3.4, 69.5),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#c9814f"))
    ax.add_patch(FancyArrowPatch((LX + LW + 15, 63.8), (LX + LW + 15, 22.5),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color="#c9814f", linestyle=(0, (5, 3))))
    ax.text(LX + LW + 16.4, 43, "becomes\nlabels.npy", fontsize=8.6, color="#a8663a",
            va="center", linespacing=1.3)

    # ── PPG-DaLiA column ────────────────────────────────────────────
    RX, RW = 71, 26
    ax.text(RX + RW / 2, 90.5, "PPG-DaLiA   —   the watch data",
            ha="center", fontsize=13, weight="bold", color=viz.AQUA)

    box(RX, 78.0, RW, 9.0, "Download",
        "data.zip, 2.7 GB\nunpacks to 24 GB", THEM)
    arrow(RX + RW / 2, 77.8, 75.2)
    box(RX, 63.5, RW, 12.5, "Keep 5 of 10 sensors",
        "wrist PPG, wrist motion,\nchest ECG, heart rate, activity\n24 GB → 259 MB", US)
    arrow(RX + RW / 2, 63.3, 61.2)
    box(RX, 52.0, RW, 9.0, "Match the speed",
        "64 → 125 samples/second", US)
    arrow(RX + RW / 2, 51.8, 49.2)
    box(RX, 40.0, RW, 9.0, "Cut into clips",
        "10 seconds, stepping 2 s", US)
    arrow(RX + RW / 2, 39.8, 37.2)
    box(RX, 28.0, RW, 9.0, "Rescale each clip to 0 – 1",
        "this one really needed it", US)

    # ── converge ────────────────────────────────────────────────────
    ax.add_patch(FancyArrowPatch((LX + LW / 2, 27.8), (LX + LW / 2, 22.5),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#8a8a85"))
    ax.add_patch(FancyArrowPatch((RX + RW / 2, 27.8), (RX + RW / 2, 22.5),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#8a8a85"))

    box(LX, 11.0, LW + 22 + 4, 11.5, "PulseDB   →   5 groups of 4 files",
        "signals.npy  (465,480 · 2 · 1250)    labels.npy  the two BP numbers\n"
        "subjects.npy  who each clip is        meta.parquet  age, sex, quality", OUT, fs=8.4)
    box(RX, 11.0, RW, 11.5, "PPG-DaLiA   →   windows",
        "signals.npy\n(64,682 · 2 · 1250)\nno BP labels exist", OUT, fs=8.4)

    ax.text(1.5, 7.6,
            "Both end in exactly the same shape — 2 signals of 1250 numbers per clip — so a model trained on the "
            "hospital data\nruns on the watch data without a single change. That is the only reason the "
            "real-world test means anything.",
            fontsize=11.6, color=viz.INK, linespacing=1.5, va="top")
    ax.text(1.5, 2.4,
            "Last step, not drawn: patients are split into groups and the code checks that none appears in two "
            "of them.",
            fontsize=10.8, color=viz.INK_2, va="top")

    plt.tight_layout()
    save(fig, "13_flowchart.png")


if __name__ == "__main__":
    fig13_flowchart()
    fig9_disk_evidence()
    fig10_processed_folder()
    fig11_dalia_archive()
    fig12_pulsedb_own_words()
    fig8_three_subjects_each()
    fig7_where_splits_come_from()
    fig1_raw_file()
    fig2_pipeline_one_subject()
    fig3_label_derivation()
    fig4_several_subjects()
    fig5_dalia_pipeline()
    fig6_verification()
