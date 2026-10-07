"""Professor's concern #1: were personalization's calibration and evaluation
clips drawn from the same continuous recording?

Finding: PulseDB's calfree_test Subset files store each subject's 400 clips in
an order uncorrelated with true recording time (confirmed via SegIDX from the
official Info proxy file — corr(storage position, SegIDX) = 0.007 across all
144 subjects). So `personalize()`'s "first k" calibration set, despite the
docstring's chronological-stream framing, is not actually the earliest k
clips in time.

This script redoes the k=25 "first" ablation using the TRUE chronological
order (sorted by SegIDX from CalFree_Test_Info.mat, not storage order), so
calibration is genuinely the earliest clips and evaluation is genuinely later
in the same session — the closest test to real deployment this dataset
supports, since PulseDB provides no cross-session data for any subject here.

Run:  python3 scripts/check_session_leakage.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from paths import model_dir, PROC_PULSEDB
from train import _affine_recalibration, offset_headroom
from evaluation import metrics

ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
         "resnet": "ResNet1D", "transformer": "Transformer"}
K = 25

segidx = np.load(PROC_PULSEDB / "calfree_test" / "segidx.npy")
valid = ~np.isnan(segidx)
print(f"segidx loaded: {valid.sum():,} of {len(segidx):,} clips matched to a session position\n")

print(f"{'model':14s} {'SBP k=0':>8s} {'SBP true-k25':>13s} {'DBP k=0':>8s} "
      f"{'DBP true-k25':>13s} {'DBP AAMI':>9s}  avg cal->eval gap (min)")
print("-" * 100)

for m in ORDER:
    d = np.load(model_dir(m) / f"{m}_calfree_predictions.npz")
    p_pool, y_pool, s_pool = d["y_pred"], d["y_true"], d["subjects"]

    corrected = p_pool.copy()
    keep = np.ones(len(p_pool), bool)
    gaps = []
    for s in np.unique(s_pool):
        mask = np.flatnonzero(s_pool == s)
        mask = mask[valid[mask]]           # only clips we could time-locate
        if mask.size < K + 5:
            keep[np.flatnonzero(s_pool == s)] = False   # drop subject if too few timed clips
            continue
        order = mask[np.argsort(segidx[mask])]          # TRUE chronological order
        cal_idx = order[:K]
        eval_idx = order[K:]
        keep[np.flatnonzero(s_pool == s)] = False
        keep[eval_idx] = True
        gaps.append((segidx[eval_idx].mean() - segidx[cal_idx].mean()) * 10 / 60)  # minutes
        for t in (0, 1):
            a, off = _affine_recalibration(p_pool[cal_idx, t], y_pool[cal_idx, t])
            corrected[eval_idx, t] = a * p_pool[eval_idx, t] + off

    k0 = metrics(y_pool[:, 0], p_pool[:, 0], s_pool), metrics(y_pool[:, 1], p_pool[:, 1], s_pool)
    ktrue = metrics(y_pool[keep, 0], corrected[keep, 0], s_pool[keep]), \
            metrics(y_pool[keep, 1], corrected[keep, 1], s_pool[keep])
    print(f"{LABEL[m]:14s} {k0[0]['mae']:8.2f} {ktrue[0]['mae']:13.2f} "
          f"{k0[1]['mae']:8.2f} {ktrue[1]['mae']:13.2f} "
          f"{'PASS' if ktrue[1]['aami_pass'] else 'FAIL':>9s}   {np.mean(gaps):.1f}")
