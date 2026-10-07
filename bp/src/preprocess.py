"""Turn raw PulseDB .mat and PPG-DaLiA .npz into model-ready waveform arrays.

Outputs, per split, under data/processed/:

    signals.npy    (N, C, 1250) float16, memory-mappable
    labels.npy     (N, 2)       float32, [SBP, DBP]   (PulseDB only)
    subjects.npy   (N,)         unicode
    meta.parquet                per-segment demographics and quality

Why float16: the stored channels are min-max normalised to [0, 1], where float16
carries ~1e-3 absolute precision — below a PPG sensor's own noise floor — and it
halves a 6.5 GB array to 3.2 GB. Loaders cast to float32.

Why .npy rather than reading the .mat directly: training needs random access, and
random reads from the 13.8 GB HDF5 are far slower than a memory-mapped array. A
sequential pass over the .mat takes minutes; per-batch random access would dominate
every epoch.

PPG-DaLiA is resampled from its native 64 Hz wrist PPG (and 700 Hz chest ECG) to
PulseDB's 125 Hz and cut into 10 s windows, so a model trained on PulseDB can be
applied to it unchanged — that is the whole point of the domain-shift test.
"""

import json

import numpy as np
import pandas as pd
from scipy import signal as sps

from fiducials import FS, SEG_LEN, detect_rpeaks, detect_ppg_fiducials

ECG_CH, PPG_CH, ABP_CH = 0, 1, 2
STORE_CHANNELS = (ECG_CH, PPG_CH)          # ABP is the label source, not an input

# PPG-DaLiA native rates
DALIA_FS = dict(ecg=700, bvp=64, acc=32, activity=4, hr=0.5)
DALIA_WINDOW_S = 10.0
DALIA_STRIDE_S = 2.0                        # matches the dataset's HR label cadence

ACTIVITY_NAMES = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
                  4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
LOW_MOTION = (1, 5, 8)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

def minmax(x, axis=-1, eps=1e-8):
    """Scale each trace to [0, 1]. Matches how PulseDB ships ECG_F / PPG_F."""
    lo = np.min(x, axis=axis, keepdims=True)
    hi = np.max(x, axis=axis, keepdims=True)
    return (x - lo) / (hi - lo + eps)


def zscore(x, axis=-1, eps=1e-8):
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / (s + eps)


# --------------------------------------------------------------------------- #
# PulseDB
# --------------------------------------------------------------------------- #

def pulsedb_to_arrays(subset, out_dir, block=4000, channels=STORE_CHANNELS,
                      quality=True, progress=None):
    """Stream one PulseDB subset into signals/labels/subjects/meta files.

    Returns the meta DataFrame. Signals are written incrementally through a
    np.memmap so peak memory stays at one block regardless of subset size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n, c = subset.n, len(channels)

    sig_path = out_dir / "signals.npy"
    mm = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float16,
                                   shape=(n, c, SEG_LEN))

    quals = np.zeros((n, 3), dtype=np.float32) if quality else None

    for start in range(0, n, block):
        stop = min(start + block, n)
        raw = subset._signals[:, :, start:stop]              # (1250, 3, B)
        arr = np.transpose(raw[:, list(channels), :], (2, 1, 0))   # (B, C, 1250)
        arr = minmax(arr.astype(np.float32), axis=-1)
        mm[start:stop] = arr.astype(np.float16)

        if quality:
            ecg_i = channels.index(ECG_CH) if ECG_CH in channels else None
            ppg_i = channels.index(PPG_CH) if PPG_CH in channels else None
            for k in range(stop - start):
                q = _segment_quality(arr[k], ecg_i, ppg_i)
                quals[start + k] = q
        if progress:
            progress(stop, n)

    mm.flush()
    del mm

    labels = np.stack([subset.sbp, subset.dbp], axis=1).astype(np.float32)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "subjects.npy", subset.subject.astype("U16"))

    meta = pd.DataFrame({
        "subject": subset.subject, "sbp": subset.sbp, "dbp": subset.dbp,
        "age": subset.age, "gender": subset.gender, "height": subset.height,
        "weight": subset.weight, "bmi": subset.bmi,
    })
    if quality:
        meta["qual_ecg"] = quals[:, 0]
        meta["qual_ppg"] = quals[:, 1]
        meta["qual_hr_agreement"] = quals[:, 2]
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    return meta


def _segment_quality(seg, ecg_i, ppg_i):
    """(ecg quality, ppg quality, ECG/PPG heart-rate agreement) for one segment."""
    ecg_q = ppg_q = agree = 0.0
    hr_e = hr_p = np.nan
    if ecg_i is not None:
        rp, ecg_q = detect_rpeaks(seg[ecg_i].astype(np.float64))
        if rp.size >= 3:
            hr_e = 60.0 * FS / np.median(np.diff(rp))
    if ppg_i is not None:
        fid = detect_ppg_fiducials(seg[ppg_i].astype(np.float64))
        ppg_q = fid["quality"]
        if fid["foot"].size >= 2:
            hr_p = 60.0 * FS / np.median(fid["next_foot"] - fid["foot"])
    if np.isfinite(hr_e) and np.isfinite(hr_p) and hr_p > 0:
        agree = 1.0 - min(1.0, abs(hr_e - hr_p) / hr_p)
    return np.array([ecg_q, ppg_q, agree], dtype=np.float32)


# --------------------------------------------------------------------------- #
# PPG-DaLiA
# --------------------------------------------------------------------------- #

def resample_to(x, fs_in, fs_out, n_out=None):
    """Polyphase resample; exact output length when n_out is given."""
    x = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out:
        y = x
    else:
        from math import gcd
        g = gcd(int(fs_in), int(fs_out))
        y = sps.resample_poly(x, int(fs_out) // g, int(fs_in) // g)
    if n_out is not None:
        if y.size >= n_out:
            y = y[:n_out]
        else:
            y = np.pad(y, (0, n_out - y.size), mode="edge")
    return y


def dalia_to_windows(npz, window_s=DALIA_WINDOW_S, stride_s=DALIA_STRIDE_S):
    """Cut one PPG-DaLiA subject into PulseDB-shaped windows.

    Returns (signals (W, 2, 1250) float16 as [ECG, PPG] to match PulseDB channel
    order, meta DataFrame with hr / activity / time).
    """
    ecg_r = np.asarray(npz["ecg"], dtype=np.float64)
    bvp_r = np.asarray(npz["bvp"], dtype=np.float64)
    act_r = np.asarray(npz["activity"])
    hr_r = np.asarray(npz["hr"], dtype=np.float64)

    dur = min(ecg_r.size / DALIA_FS["ecg"], bvp_r.size / DALIA_FS["bvp"])
    n_out = int(round(dur * FS))
    ecg = resample_to(ecg_r, DALIA_FS["ecg"], FS, n_out)
    ppg = resample_to(bvp_r, DALIA_FS["bvp"], FS, n_out)

    win, stride = int(window_s * FS), int(stride_s * FS)
    starts = np.arange(0, max(0, n_out - win + 1), stride)

    sigs = np.empty((starts.size, 2, win), dtype=np.float16)
    rows = []
    for i, s in enumerate(starts):
        seg = np.stack([ecg[s:s + win], ppg[s:s + win]])
        sigs[i] = minmax(seg, axis=-1).astype(np.float16)

        t0 = s / FS
        # activity label: the mode over the window
        a0, a1 = int(t0 * DALIA_FS["activity"]), int((t0 + window_s) * DALIA_FS["activity"])
        a = act_r[a0:a1]
        act = int(np.bincount(a[a >= 0].astype(int)).argmax()) if a.size else -1
        # HR label nearest the window centre (labels are 8 s windows, 2 s shift)
        hi = int(round((t0 + window_s / 2 - 4.0) / 2.0))
        hr = float(hr_r[hi]) if 0 <= hi < hr_r.size else np.nan

        rows.append(dict(start_s=t0, activity=act,
                         activity_name=ACTIVITY_NAMES.get(act, "?"),
                         low_motion=act in LOW_MOTION, hr=hr))
    return sigs, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# splits manifest
# --------------------------------------------------------------------------- #

def write_splits_manifest(subject_map, path):
    """Persist which subjects are in which split, and assert the intended disjointness.

    The manifest is the paper's central claim in machine-readable form, so it lives
    in its own file rather than being implicit in the waveform arrays.
    """
    manifest = {k: sorted(map(str, v)) for k, v in subject_map.items()}

    must_be_disjoint = [("train", "calfree_test"), ("train", "aami_cal"),
                        ("train", "aami_test"), ("calfree_test", "calbased_test")]
    checks = {}
    for a, b in must_be_disjoint:
        if a in manifest and b in manifest:
            shared = set(manifest[a]) & set(manifest[b])
            checks[f"{a}|{b}"] = len(shared)
            if shared:
                raise AssertionError(
                    f"LEAKAGE: {len(shared)} subjects shared between {a} and {b}")

    # These two are meant to overlap; record it so it is never mistaken for a bug.
    for a, b in [("aami_cal", "aami_test"), ("train", "calbased_test")]:
        if a in manifest and b in manifest:
            checks[f"{a}&{b}(expected overlap)"] = len(
                set(manifest[a]) & set(manifest[b]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"n_subjects": {k: len(v) for k, v in manifest.items()},
         "overlap_checks": checks, "subjects": manifest}, indent=2))
    return checks
