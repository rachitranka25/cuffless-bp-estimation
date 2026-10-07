"""Figures for PPG-DaLiA — the free-living dataset the domain-shift test runs on.

PulseDB is 10 seconds of a still patient at a time; PPG-DaLiA is two and a half
hours of someone living their day. These figures show what that difference looks
like in the signal, and quantify how far pulse detection degrades as motion rises —
which is the concrete form the ICU-to-wearable gap takes.

Run:  python3 scripts/visualize_dalia.py
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import PROC_DALIA, DATASET_FIG, FIG_PULSEDB, FIG_DALIA, FIG_COMPARE, ROOT      # noqa: E402
from fiducials import FS, detect_ppg_fiducials       # noqa: E402
from preprocess import ACTIVITY_NAMES, LOW_MOTION    # noqa: E402
import viz                                           # noqa: E402

viz.use_style()
FIG = DATASET_FIG
for _d in (FIG_PULSEDB, FIG_DALIA, FIG_COMPARE):
    _d.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(11)

FILES = sorted(glob.glob(str(PROC_DALIA / "S*.npz")), key=lambda p: int(Path(p).stem[1:]))
# Ordered easiest -> hardest for the wrist sensor.
ACT_ORDER = ["sitting", "working", "driving", "lunch", "table_soccer",
             "stairs", "walking", "cycling"]


def save(fig, name):
    fig.savefig(FIG / name)
    print("saved", (FIG / name).relative_to(ROOT))
    plt.close(fig)


def acc_mag(acc):
    """Motion magnitude, gravity removed."""
    m = np.linalg.norm(np.asarray(acc, dtype=np.float32), axis=1)
    return np.abs(m - np.median(m))


# --------------------------------------------------------------------------- #

def fig_npz_contents():
    """Every array in one .npz, each at its native sampling rate."""
    d = np.load(FILES[0], allow_pickle=True)
    t0, dur = 600.0, 20.0            # a calm 20 s window well into the session

    fig, axes = plt.subplots(5, 1, figsize=(12.5, 8.5))
    specs = [
        ("ecg", 700, "chest ECG", viz.CH["ecg"]),
        ("bvp", 64, "wrist PPG (BVP)", viz.CH["ppg"]),
        (None, 32, "wrist accelerometer", viz.AQUA),
        ("hr", 0.5, "heart-rate label", viz.VIOLET),
        ("activity", 4, "activity label", viz.YELLOW),
    ]
    for ax, (key, fs, label, col) in zip(axes, specs):
        if key == "activity":
            a = d["activity"][int(t0 * fs):int((t0 + dur) * fs)]
            ax.step(np.arange(a.size) / fs, a, where="post", color=col, lw=1.8)
            ax.set_ylim(-.5, 8.5)
            ax.set_yticks(sorted(set(a.tolist())),
                          [ACTIVITY_NAMES[int(v)] for v in sorted(set(a.tolist()))])
        elif key is None:
            a = d["acc"][int(t0 * fs):int((t0 + dur) * fs)]
            for k, axis_name in enumerate("xyz"):
                ax.plot(np.arange(a.shape[0]) / fs, a[:, k], lw=1.0,
                        color=[viz.AQUA, viz.BLUE, viz.MAGENTA][k], label=axis_name)
            ax.legend(loc="upper right", ncol=3)
        else:
            a = d[key][int(t0 * fs):int((t0 + dur) * fs)]
            ax.plot(np.arange(a.size) / fs, a, color=col, lw=1.2)
        ax.set_ylabel(f"{label}\n{fs} Hz", fontsize=8)
        viz.despine(ax)
    axes[-1].set_xlabel(f"seconds (from t = {t0:.0f} s into the session)")
    viz.title(axes[0], "Everything inside S1.npz",
              "five streams at five different sampling rates")
    plt.tight_layout()
    save(fig, "ppg_dalia/01_npz_contents.png")


def fig_session_timeline():
    """One subject's whole session: what they did, their HR, and how much they moved."""
    d = np.load(FILES[0], allow_pickle=True)
    act = d["activity"]
    hr = d["hr"]
    mot = acc_mag(d["acc"])

    t_act = np.arange(act.size) / 4 / 60
    t_hr = np.arange(hr.size) * 2 / 60
    t_mot = np.arange(mot.size) / 32 / 60
    # one-second smoothing so the motion trace is legible
    k = 32 * 5
    mot_s = np.convolve(mot, np.ones(k) / k, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [.55, 1, 1]})
    ax = axes[0]
    present = [a for a in range(9) if (act == a).any()]
    cmap = plt.get_cmap("tab10")
    for a in present:
        m = act == a
        ax.fill_between(t_act, 0, 1, where=m, step="mid",
                        color=cmap(a % 10), alpha=.85, lw=0)
    ax.set_yticks([])
    ax.set_ylabel("activity", fontsize=8)
    ax.grid(False)
    viz.despine(ax, keep=())
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(a % 10), alpha=.85)
               for a in present]
    ax.legend(handles, [ACTIVITY_NAMES[a] for a in present],
              loc="upper center", ncol=len(present), fontsize=7,
              bbox_to_anchor=(.5, 2.15))
    viz.title(ax, "A full PPG-DaLiA session — subject S1, 2.6 hours",
              "PulseDB gives 10 still seconds at a time; this is a whole day's worth of variety")

    axes[1].plot(t_hr, hr, color=viz.VIOLET, lw=1.2)
    axes[1].set_ylabel("heart rate (bpm)")
    viz.despine(axes[1])

    axes[2].plot(t_mot, mot_s, color=viz.AQUA, lw=1.0)
    axes[2].set_ylabel("wrist motion (g)")
    axes[2].set_xlabel("minutes")
    viz.despine(axes[2])
    plt.tight_layout()
    save(fig, "ppg_dalia/02_session_timeline.png")


def _windows():
    sig = np.load(PROC_DALIA / "windows" / "signals.npy", mmap_mode="r")
    meta = pd.read_parquet(PROC_DALIA / "windows" / "meta.parquet")
    return sig, meta


def fig_ppg_by_activity():
    """The same wrist sensor, ordered by how much the person was moving."""
    sig, meta = _windows()
    shown = [a for a in ACT_ORDER if (meta.activity_name == a).any()]

    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 1.15 * len(shown)),
                             sharex=True, sharey=True)
    t = np.arange(1250) / FS
    for ax, name in zip(axes, shown):
        idx = meta.index[meta.activity_name == name].to_numpy()
        pick = RNG.choice(idx, min(3, idx.size), replace=False)
        col = viz.AQUA if name in [ACTIVITY_NAMES[a] for a in LOW_MOTION] else viz.ORANGE
        for j, i in enumerate(pick):
            ax.plot(t, np.asarray(sig[i, 1], dtype=np.float32),
                    color=col, lw=1.0, alpha=[1, .55, .3][j])
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        viz.despine(ax, keep=("bottom",))
    axes[-1].set_xlabel("seconds")
    viz.title(axes[0], "Wrist PPG by activity",
              "aqua = low motion; the pulse survives sitting and dissolves under cycling")
    plt.tight_layout()
    save(fig, "ppg_dalia/06_ppg_by_activity.png")


def fig_motion_vs_quality():
    """Quantify degradation against ground truth, not against self-consistency.

    A detector's own plausibility score is the wrong yardstick here. Walking and
    cycling drive the arm at roughly 2 Hz, which lands squarely inside the
    plausible heart-rate band, so a rhythmic motion artefact looks exactly like a
    confident pulse train and scores high. PPG-DaLiA ships a heart rate derived
    from the chest ECG, so the honest measurement is whether the wrist PPG agrees
    with it — that is what a wearable would actually be getting wrong.
    """
    sig, meta = _windows()
    take = np.sort(RNG.choice(len(meta), 4000, replace=False))

    rows = []
    for i in take:
        ppg = np.asarray(sig[i, 1], dtype=np.float64)
        fid = detect_ppg_fiducials(ppg)
        hr_true = meta.hr.iloc[i]
        hr_ppg = (60.0 * FS / np.median(fid["next_foot"] - fid["foot"])
                  if fid["foot"].size >= 2 else np.nan)
        rows.append(dict(self_q=fid["quality"], hr_ppg=hr_ppg, hr_true=hr_true,
                         err=abs(hr_ppg - hr_true) if np.isfinite(hr_ppg * hr_true) else np.nan,
                         activity=meta.activity_name.iloc[i],
                         low=meta.low_motion.iloc[i]))
    q = pd.DataFrame(rows).dropna(subset=["err"])
    q["ok"] = q.err <= 5.0            # within 5 bpm of the ECG reference

    by_act = (q.groupby("activity")
                .agg(agree=("ok", "mean"), mae=("err", "median"),
                     self_q=("self_q", "mean"), n=("ok", "size"))
                .reindex([a for a in ACT_ORDER if a in set(q.activity)])
                .dropna())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
    ax = axes[0]
    y = np.arange(len(by_act))[::-1]
    ax.barh(y + .19, by_act.self_q * 100, height=.36, color=viz.MUTED,
            label="detector's own confidence")
    cols = [viz.AQUA if a in [ACTIVITY_NAMES[x] for x in LOW_MOTION] else viz.ORANGE
            for a in by_act.index]
    ax.barh(y - .19, by_act.agree * 100, height=.36, color=cols,
            label="actually within 5 bpm of chest ECG")
    for yy, v in zip(y, by_act.agree * 100):
        ax.text(v + 1.5, yy - .19, f"{v:.0f}%", va="center", fontsize=8, color=viz.INK_2)
    ax.set_yticks(y, by_act.index)
    ax.set_xlabel("% of windows")
    ax.set_xlim(0, 112)
    # below the axes: the bars reach the right edge on some rows, so an inset
    # legend would sit on top of the data
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, "Wrist PPG vs the chest-ECG reference",
              "grey shows the trap: motion artefacts look like confident pulses")

    ax = axes[1]
    for lab, m, col in [("low motion", q.low, viz.AQUA),
                        ("higher motion", ~q.low, viz.ORANGE)]:
        ax.hist(q.loc[m, "err"], bins=50, range=(0, 60), density=True,
                color=col, alpha=.8,
                label=f"{lab} — median {q.loc[m,'err'].median():.1f} bpm (n={int(m.sum()):,})")
    ax.axvline(5, color=viz.INK_2, ls="--", lw=1)
    ax.set_xlabel("|PPG heart rate − ECG heart rate| (bpm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    viz.despine(ax)
    viz.title(ax, "Heart-rate error against ground truth",
              "dashed line = 5 bpm; the spec's low-motion-first advice, justified")
    plt.tight_layout()
    save(fig, "ppg_dalia/07_motion_vs_accuracy.png")
    print("\n" + by_act.round(3).to_string())
    return q


def fig_subject_overview():
    """All 15 subjects at a glance."""
    sig, meta = _windows()
    rows = []
    for f in FILES:
        d = np.load(f, allow_pickle=True)
        s = Path(f).stem
        qn = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in d["questionnaire"]}
        sub = meta[meta.subject == s]
        rows.append(dict(subject=s, age=int(qn.get("AGE", -1)),
                         sex=qn.get("Gender", "?").strip(),
                         minutes=len(d["ecg"]) / 700 / 60,
                         hr_med=float(np.median(d["hr"])),
                         hr_max=float(np.max(d["hr"])),
                         windows=len(sub),
                         low_pct=100 * sub.low_motion.mean()))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    o = df.sort_values("age")
    ax.barh(o.subject, o.age, color=[viz.BLUE if s == "m" else viz.MAGENTA
                                     for s in o.sex], height=.62)
    for y, (a, s) in enumerate(zip(o.age, o.sex)):
        ax.text(a + .6, y, f"{a} {s}", va="center", fontsize=7.5, color=viz.INK_2)
    ax.set_xlabel("age (years)")
    ax.set_xlim(0, o.age.max() * 1.28)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, "Subjects", "blue = m, magenta = f · nobody over 55")

    ax = axes[1]
    for _, r in df.iterrows():
        d = np.load(PROC_DALIA / f"{r.subject}.npz", allow_pickle=True)
        ax.plot(np.sort(d["hr"]), np.linspace(0, 1, d["hr"].size),
                lw=1.1, color=viz.BLUE, alpha=.55)
    ax.set_xlabel("heart rate (bpm)")
    ax.set_ylabel("cumulative fraction")
    viz.despine(ax)
    viz.title(ax, "Heart-rate range per subject",
              "one line per subject; free living spans 40-190 bpm")

    ax = axes[2]
    o = df.sort_values("low_pct")
    bars = ax.barh(o.subject, o.low_pct, color=viz.AQUA, height=.62)
    for b, v in zip(bars, o.low_pct):
        ax.text(v + .8, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                va="center", fontsize=7.5, color=viz.INK_2)
    ax.set_xlabel("% of windows at low motion")
    ax.set_xlim(0, max(60, o.low_pct.max() * 1.25))
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, "Usable-looking share", "how much easy data each subject offers")
    plt.tight_layout()
    save(fig, "ppg_dalia/03_all_subjects.png")
    print("\n" + df.round(1).to_string(index=False))


if __name__ == "__main__":
    fig_npz_contents()
    fig_session_timeline()
    fig_ppg_by_activity()
    fig_motion_vs_quality()
    fig_subject_overview()
