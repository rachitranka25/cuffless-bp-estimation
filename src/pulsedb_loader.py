"""Minimal loader for PulseDB Subset .mat files (MATLAB v7.3 = HDF5).

Channel order comes from the authors' Generate_Subsets.m line 102:
    Subset.Signals(pos,:,:) = [Segment.ECG_F, Segment.PPG_F, Segment.ABP_Raw]'
so channel 0 = ECG (min-max normalised), 1 = PPG (min-max normalised),
2 = ABP in mmHg (raw). Segments are 10 s at 125 Hz -> 1250 samples.
"""

import h5py
import numpy as np

FS = 125
SEG_LEN = 1250
ECG, PPG, ABP = 0, 1, 2


def _decode_str_array(f, dataset):
    """MATLAB cellstr comes through as an array of object references."""
    return np.array([
        "".join(chr(c[0]) for c in f[ref][:]) for ref in dataset[0]
    ])


class PulseDBSubset:
    """Lazy handle on one Subset file. Signals stay on disk until sliced."""

    def __init__(self, path):
        self.path = path
        self._f = h5py.File(path, "r")
        s = self._f["Subset"]
        self._signals = s["Signals"]          # h5py order: (1250, 3, n_segments)
        self.n = self._signals.shape[2]
        self.sbp = s["SBP"][0].astype(np.float32)
        self.dbp = s["DBP"][0].astype(np.float32)
        self.age = s["Age"][0].astype(np.float32)
        self.bmi = s["BMI"][0].astype(np.float32)
        self.height = s["Height"][0].astype(np.float32)
        self.weight = s["Weight"][0].astype(np.float32)
        self.subject = _decode_str_array(self._f, s["Subject"])
        self.gender = _decode_str_array(self._f, s["Gender"])

    @property
    def subjects(self):
        return np.unique(self.subject)

    def signals(self, idx, channels=(ECG, PPG)):
        """Return (len(idx), len(channels), 1250) float32 for segment indices idx.

        h5py needs indices sorted and unique along the fancy-indexed axis, so we
        sort, read, then scatter back into the caller's order.
        """
        idx = np.asarray(idx)
        order = np.argsort(idx)
        uniq, inverse = np.unique(idx[order], return_inverse=True)
        raw = self._signals[:, list(channels), :][:, :, uniq]   # (1250, C, U)
        raw = np.transpose(raw, (2, 1, 0))                      # (U, C, 1250)
        out = np.empty((len(idx), len(channels), SEG_LEN), dtype=np.float32)
        out[order] = raw[inverse]
        return out

    def indices_for(self, subject_ids):
        """Segment indices belonging to the given subject IDs."""
        return np.flatnonzero(np.isin(self.subject, np.asarray(subject_ids)))

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"<PulseDBSubset {self.path.split('/')[-1]}: "
                f"{self.n} segments, {len(self.subjects)} subjects>")


def assert_subject_disjoint(a, b, name_a="A", name_b="B"):
    """Hard guard against the leakage this project exists to measure."""
    shared = np.intersect1d(a.subjects, b.subjects)
    if shared.size:
        raise AssertionError(
            f"LEAKAGE: {shared.size} subjects appear in both {name_a} and "
            f"{name_b} (e.g. {shared[:5].tolist()})"
        )
