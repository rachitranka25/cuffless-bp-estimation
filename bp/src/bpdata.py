"""Torch datasets and protocol splits over the processed PulseDB arrays.

Signals stay memory-mapped; only the sampled batch is read and cast to float32.
A 2.3 GB array therefore costs no RAM to open and the workers share the page
cache instead of each holding a copy.

The three protocols mirror the classical baselines exactly, so the deep numbers
are directly comparable:

    leaky    random segment split inside the training subset — the same patient
             on both sides
    honest   same subset, split so no subject crosses
    calfree  train on the training subset, test on the official
             calibration-free subset (144 unseen subjects)

Validation splits are always subject-disjoint from training, including under the
leaky protocol. Early stopping on a leaky validation set would leak a second
time, through model selection rather than through the test score.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from paths import PROC_PULSEDB


class SegmentDataset(Dataset):
    """Yields (signal, label), (signal, demographics, label), or — when both
    `demo` and `morph` are given — (signal, demographics, morph_target, label).

    `channels` selects which rows of the stored (2, 1250) array to return, so a
    PPG-only model can be trained on the same files as a PPG+ECG model.

    `zscore`, when set, re-scales each returned channel to zero mean / unit
    variance per clip. The stored arrays are already min-max normalised to
    [0, 1] and saved as float16 (see preprocess.py) — the original unnormalised
    waveform isn't kept — so this is z-scoring the min-max'd signal, not the raw
    ADC trace DMT describes. Close enough to match the *effect* (well-scaled
    inputs, zero-centred), not a bit-exact reproduction.

    `morph` is a full-length (N, k) array of auxiliary regression targets
    (indexed the same way as `signals`/`labels`/`demo`, i.e. by the row's
    position in the underlying split, not by position in `indices`), used only
    to train DMT's auxiliary morphology head — never needed for validation,
    test, or inference, so callers that don't pass it get the old tuple shape
    back unchanged.
    """

    def __init__(self, signals, labels, indices, y_mean=None, y_std=None,
                 demo=None, channels=None, zscore=False, morph=None):
        self.sig = signals
        self.lab = labels
        self.idx = np.asarray(indices)
        self.y_mean = y_mean
        self.y_std = y_std
        self.demo = demo
        self.channels = list(channels) if channels is not None else None
        self.zscore = zscore
        self.morph = morph

    def __len__(self):
        return self.idx.size

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.sig[j], dtype=np.float32)
        if self.channels is not None:
            x = x[self.channels]
        if self.zscore:
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mu) / sd
        x = torch.from_numpy(np.ascontiguousarray(x))
        y = torch.from_numpy(self.lab[j].astype(np.float32))
        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std
        if self.demo is None:
            return x, y
        d = torch.from_numpy(self.demo[j].astype(np.float32))
        if self.morph is None:
            return x, d, y
        # np.asarray, not .astype directly — self.morph[j] is a bare numpy
        # scalar when morph is 1-D (e.g. integer class labels), and a scalar
        # has no .astype method the way an array slice does
        m = torch.from_numpy(np.asarray(self.morph[j], dtype=np.float32))
        return x, d, m, y


DEMO_COLS = ("age", "gender", "bmi")


def demographics(split, stats=None):
    """(N, 3) array of standardised age, sex and BMI for one split.

    DMT conditions on exactly these three. Sex becomes 1 for male, 0 otherwise;
    age and BMI are standardised. Missing values become 0, i.e. the training
    mean, which is the least-informative guess.

    Pass `stats` from the training split so validation and test are standardised
    with the training statistics rather than their own.
    """
    # Prefer the plain .npy — some machines cannot load parquet at all (Windows
    # Application Control blocks the compiled pyarrow and fastparquet DLLs), and
    # the transformer only needs three columns out of that file anyway.
    npy = PROC_PULSEDB / split / "demographics.npy"
    if npy.exists():
        d = np.load(npy).astype(np.float64)
        age, sex, bmi = d[:, 0], d[:, 1], d[:, 2]
    else:
        import pandas as pd
        meta = pd.read_parquet(PROC_PULSEDB / split / "meta.parquet")
        age = meta["age"].to_numpy(dtype=np.float64)
        bmi = meta["bmi"].to_numpy(dtype=np.float64)
        sex = np.array([1.0 if str(g).upper().startswith("M") else 0.0
                        for g in meta["gender"]], dtype=np.float64)
    if stats is None:
        stats = {}
        for name, v in (("age", age), ("bmi", bmi)):
            good = v[np.isfinite(v)]
            stats[name] = (float(good.mean()), float(good.std() + 1e-8))
    out = np.stack([
        np.nan_to_num((age - stats["age"][0]) / stats["age"][1], nan=0.0),
        sex,
        np.nan_to_num((bmi - stats["bmi"][0]) / stats["bmi"][1], nan=0.0),
    ], axis=1).astype(np.float32)
    return out, stats


def load_split(name):
    p = PROC_PULSEDB / name
    return (np.load(p / "signals.npy", mmap_mode="r"),
            np.load(p / "labels.npy"),
            np.load(p / "subjects.npy"))


def cap_indices(subjects, cap, seed=0):
    """At most `cap` segments per subject, sampled without replacement."""
    if cap is None:
        return np.arange(subjects.size)
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out.append(idx if idx.size <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(out))


def _subject_split(subjects, indices, frac, seed):
    subs = np.unique(subjects[indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = max(1, int(len(subs) * frac))
    held = set(subs[:n].tolist())
    mask = np.array([s in held for s in subjects[indices]])
    return indices[~mask], indices[mask]


def make_protocol(protocol, cap=100, val_frac=0.1, test_frac=0.2, seed=0):
    """Return (train_ds_parts, val_parts, test_parts) as index/array bundles."""
    sig, lab, subj = load_split("train")
    pool = cap_indices(subj, cap, seed)

    if protocol == "leaky":
        rng = np.random.default_rng(seed)
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * test_frac)
        test_idx, rest = shuffled[:n_test], shuffled[n_test:]
        # validation is still subject-disjoint from train — see module docstring
        tr_idx, val_idx = _subject_split(subj, np.sort(rest), val_frac, seed + 1)
        test_sig, test_lab, test_subj = sig, lab, subj
        test_split = "train"

    elif protocol == "honest":
        rest, test_idx = _subject_split(subj, pool, test_frac, seed)
        tr_idx, val_idx = _subject_split(subj, rest, val_frac, seed + 1)
        test_sig, test_lab, test_subj = sig, lab, subj
        test_split = "train"

    elif protocol == "calfree":
        tr_idx, val_idx = _subject_split(subj, pool, val_frac, seed + 1)
        test_sig, test_lab, test_subj = load_split("calfree_test")
        test_idx = cap_indices(test_subj, cap, seed)
        test_split = "calfree_test"

    else:
        raise ValueError(protocol)

    return dict(
        signals=sig, labels=lab, subjects=subj,
        train_idx=tr_idx, val_idx=val_idx,
        test_signals=test_sig, test_labels=test_lab,
        test_subjects=test_subj, test_idx=test_idx,
        train_split="train", test_split=test_split,
    )


def target_stats(labels, idx):
    y = labels[idx]
    return (torch.tensor(y.mean(0), dtype=torch.float32),
            torch.tensor(y.std(0), dtype=torch.float32))


def build_datasets(bundle, channels=None, use_demo=False):
    m, s = target_stats(bundle["labels"], bundle["train_idx"])
    d_tr = d_te = None
    if use_demo:
        d_tr, stats = demographics(bundle["train_split"])
        d_te = (d_tr if bundle["test_split"] == bundle["train_split"]
                else demographics(bundle["test_split"], stats)[0])
    kw = dict(channels=channels)
    tr = SegmentDataset(bundle["signals"], bundle["labels"], bundle["train_idx"],
                        m, s, demo=d_tr, **kw)
    va = SegmentDataset(bundle["signals"], bundle["labels"], bundle["val_idx"],
                        m, s, demo=d_tr, **kw)
    te = SegmentDataset(bundle["test_signals"], bundle["test_labels"],
                        bundle["test_idx"], m, s, demo=d_te, **kw)
    return tr, va, te, m, s


def assert_no_subject_overlap(bundle, protocol):
    """Guard the honest protocols; leaky is expected to overlap by construction."""
    tr = set(bundle["subjects"][bundle["train_idx"]].tolist())
    te = set(bundle["test_subjects"][bundle["test_idx"]].tolist())
    va = set(bundle["subjects"][bundle["val_idx"]].tolist())
    shared_val = tr & va
    if shared_val:
        raise AssertionError(f"{len(shared_val)} subjects shared between train and val")
    shared = tr & te
    if protocol == "leaky":
        return len(shared)                      # expected — this is the point
    if shared:
        raise AssertionError(
            f"LEAKAGE in '{protocol}': {len(shared)} subjects in both train and test")
    return 0
