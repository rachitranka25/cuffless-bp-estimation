"""Personalization uncertainty: the three-seed check elsewhere in this project
covers calibration-free only, so the personalization claims
(naive-split AAMI pass, chronological-split AAMI fail, and the gap between the
two) carry no interval anywhere in the manuscript. This matters most for the
AAMI verdict at k=25, where the best chronological-split DBP SDE (8.75 mmHg,
Transformer) sits close enough to the 8 mmHg limit that sampling noise alone
could matter.

This script applies the same patient-level cluster bootstrap used in
bootstrap_ci.py to the k=25 personalization result, for both splits, and to
the naive-minus-chronological SDE difference directly (paired: the same
resampled patients are reused for both splits' corrections in each draw, so
the interval isolates genuine split-dependence rather than shared test-set
sampling noise).

Run:  python3 scripts/personalization_bootstrap.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from paths import model_dir, PROC_PULSEDB
from train import _affine_recalibration, _shot_indices
from evaluation import metrics

ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
         "resnet": "ResNet1D", "transformer": "Transformer"}
BASE_TAG = {"cnn": "schedfix", "resnet": "schedfix"}
K = 25
N_BOOT = 2000
RNG_SEED = 0

segidx = np.load(PROC_PULSEDB / "calfree_test" / "segidx.npy")
valid = ~np.isnan(segidx)


def base_name(m):
    tag = BASE_TAG.get(m, "")
    return f"{m}_calfree_{tag}" if tag else f"{m}_calfree"


def load(m):
    d = np.load(model_dir(m) / f"{base_name(m)}_predictions.npz")
    return d["y_pred"], d["y_true"], d["subjects"]


def naive_dbp_sde(p_pool, y_pool, s_pool, subject_subset):
    """k=25, storage-order first-k calibration, scored on the resampled
    (possibly repeated) subject list."""
    take = _shot_indices(s_pool, K, 0, "first", y_pool)
    corr, true_, subj = [], [], []
    for s in subject_subset:
        mask = np.flatnonzero(s_pool == s)
        if mask.size == 0:
            continue
        cal_idx = take.get(s, np.array([], dtype=int))
        keep = np.setdiff1d(mask, cal_idx, assume_unique=False)
        a, off = _affine_recalibration(p_pool[cal_idx, 1], y_pool[cal_idx, 1])
        corr.append(a * p_pool[keep, 1] + off)
        true_.append(y_pool[keep, 1])
        subj.append(s_pool[keep])
    return metrics(np.concatenate(true_), np.concatenate(corr), np.concatenate(subj))["sde"]


def chron_dbp_sde(p_pool, y_pool, s_pool, subject_subset):
    """k=25, true-chronological calibration (earliest-k by segidx), scored on
    the resampled (possibly repeated) subject list; mirrors
    personalization_early_vs_late.py's true_early_sweep, patient-subset aware."""
    corr, true_, subj = [], [], []
    for s in subject_subset:
        mask = np.flatnonzero(s_pool == s)
        mask = mask[valid[mask]]
        if mask.size < K + 5:
            continue
        order = mask[np.argsort(segidx[mask])]
        cal_idx, eval_idx = order[:K], order[K:]
        a, off = _affine_recalibration(p_pool[cal_idx, 1], y_pool[cal_idx, 1])
        corr.append(a * p_pool[eval_idx, 1] + off)
        true_.append(y_pool[eval_idx, 1])
        subj.append(s_pool[eval_idx])
    return metrics(np.concatenate(true_), np.concatenate(corr), np.concatenate(subj))["sde"]


print("=" * 90)
print(f"Patient-level bootstrap 95% CI, DBP SDE at k={K}, naive vs.\\ true-chronological split")
print("AAMI requires SDE <= 8.00 mmHg")
print("=" * 90)
print(f"{'Model':18s} {'naive SDE [95% CI]':26s} {'chron. SDE [95% CI]':26s} {'naive - chron [95% CI]':26s}")

for m in ORDER:
    p_pool, y_pool, s_pool = load(m)
    uniq = np.unique(s_pool)
    rng = np.random.default_rng(RNG_SEED)

    point_naive = naive_dbp_sde(p_pool, y_pool, s_pool, uniq)
    point_chron = chron_dbp_sde(p_pool, y_pool, s_pool, uniq)
    point_diff = point_naive - point_chron

    naive_draws = np.empty(N_BOOT)
    chron_draws = np.empty(N_BOOT)
    diff_draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        n_sde = naive_dbp_sde(p_pool, y_pool, s_pool, drawn)
        c_sde = chron_dbp_sde(p_pool, y_pool, s_pool, drawn)
        naive_draws[b] = n_sde
        chron_draws[b] = c_sde
        diff_draws[b] = n_sde - c_sde

    n_lo, n_hi = np.percentile(naive_draws, [2.5, 97.5])
    c_lo, c_hi = np.percentile(chron_draws, [2.5, 97.5])
    d_lo, d_hi = np.percentile(diff_draws, [2.5, 97.5])
    # recentre on the seed-0 point estimate (draws and point both already use
    # the same seed-0 predictions here, so no recentring offset is needed —
    # unlike bootstrap_ci.py, there is no 3-seed canonical point to match)
    sig = "excludes 8 -> AAMI fail is robust" if c_lo > 8.0 else "includes 8 -> borderline under resampling"
    print(f"{LABEL[m]:18s} "
          f"{point_naive:5.2f} [{n_lo:5.2f},{n_hi:5.2f}]{'':5s} "
          f"{point_chron:5.2f} [{c_lo:5.2f},{c_hi:5.2f}]{'':5s} "
          f"{point_diff:+5.2f} [{d_lo:+5.2f},{d_hi:+5.2f}]")
    print(f"{'':18s} chronological-split AAMI check: {sig}")
