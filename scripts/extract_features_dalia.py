"""Extract the same 60 hand-crafted features for PPG-DaLiA into features/dalia.parquet.

The PulseDB version of this script reads its segments straight out of the
authors' HDF5. DaLiA has no such file — the windows were cut here, so this reads
data/processed/ppg_dalia/windows/ instead. Everything downstream is identical:
same `extract()`, same column names, so a model fitted on train.parquet can be
scored on this without any remapping.

Demographics come from each subject's questionnaire in the per-subject .npz;
they are not in the window metadata because they do not vary per window.

There is no BP label to attach — that is the point of the dataset — so the
columns carried through instead are the ones DaLiA does have: activity, motion
flag, and the chest-ECG heart rate.

Run:  python3 scripts/extract_features_dalia.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import PROC_DALIA, FEATURES          # noqa: E402
from features import extract                    # noqa: E402

BLOCK = 4000
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def _questionnaire(subject):
    """age / gender / height / weight / bmi for one DaLiA volunteer."""
    z = np.load(PROC_DALIA / f"{subject}.npz", allow_pickle=True)
    kv = {}
    for entry in z["questionnaire"]:
        k, _, v = str(entry).partition("=")
        kv[k.strip().upper()] = v.strip()
    height = float(kv.get("HEIGHT", "nan"))
    weight = float(kv.get("WEIGHT", "nan"))
    bmi = weight / (height / 100.0) ** 2 if height and np.isfinite(height) else np.nan
    return dict(age=float(kv.get("AGE", "nan")),
                gender=kv.get("GENDER", "")[:1].upper(),
                height=height, weight=weight, bmi=bmi)


def run():
    win = PROC_DALIA / "windows"
    out = FEATURES / "dalia.parquet"

    sig = np.load(win / "signals.npy", mmap_mode="r")
    subjects = np.load(win / "subjects.npy", allow_pickle=True)
    meta = pd.read_parquet(win / "meta.parquet")
    n = sig.shape[0]
    assert len(subjects) == len(meta) == n, "windows are out of step"

    demo_by_subject = {s: _questionnaire(s) for s in np.unique(subjects)}
    print(f"dalia: {n} windows, {len(demo_by_subject)} subjects -> {out.name}")

    rows = []
    t0 = time.time()
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        blk = np.asarray(sig[start:stop], dtype=np.float64)
        demos = [demo_by_subject[subjects[i]] for i in range(start, stop)]
        rows.extend(Parallel(n_jobs=N_JOBS, batch_size=64)(
            delayed(extract)(blk[k, 0], blk[k, 1], demos[k]) for k in range(stop - start)))
        el = time.time() - t0
        print(f"\r  {stop}/{n}  {el:6.0f}s elapsed, "
              f"{el / stop * (n - stop):6.0f}s left", end="", flush=True)
    print()

    df = pd.DataFrame(rows)
    # no BP label exists here; carry what DaLiA does have instead
    df.insert(0, "subject", subjects)
    df.insert(1, "hr_ecg_ref", meta["hr"].to_numpy())
    df.insert(2, "activity_name", meta["activity_name"].to_numpy())
    df.insert(3, "low_motion", meta["low_motion"].to_numpy())
    FEATURES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out}  {df.shape[0]} rows x {df.shape[1]} cols  "
          f"{out.stat().st_size / 1e6:.0f} MB  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
