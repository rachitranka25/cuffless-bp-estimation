"""Professor's priority #1 (2nd round): the early-vs-late personalization
table, side by side with the original, for all five models, at every k.

"Original" = personalize()'s storage-order first-k (results/models/*/​
*_personalization_calfree.json, pick='first'). "True chronological" = the
same first-k idea but sorted by the clip's real position in the recording
(SegIDX from the official CalFree_Test_Info.mat proxy), so calibration really
is the earliest k clips and evaluation really is later in the same session —
see scripts/check_session_leakage.py for the k=25-only version this extends.

Run:  python3 scripts/personalization_early_vs_late.py
"""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from paths import model_dir, PROC_PULSEDB, OVERVIEW
from train import _affine_recalibration
from evaluation import metrics

ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
         "resnet": "ResNet1D", "transformer": "Transformer"}
SHOTS = (1, 3, 5, 10, 25)

segidx = np.load(PROC_PULSEDB / "calfree_test" / "segidx.npy")
valid = ~np.isnan(segidx)


def true_early_sweep(m):
    """DBP {mae, aami_pass} at each k in SHOTS, true-chronological split."""
    d = np.load(model_dir(m) / f"{m}_calfree_predictions.npz")
    p_pool, y_pool, s_pool = d["y_pred"], d["y_true"], d["subjects"]
    out = {}
    for k in SHOTS:
        corrected = p_pool.copy()
        keep = np.ones(len(p_pool), bool)
        for s in np.unique(s_pool):
            mask = np.flatnonzero(s_pool == s)
            mask = mask[valid[mask]]
            if mask.size < k + 5:
                keep[np.flatnonzero(s_pool == s)] = False
                continue
            order = mask[np.argsort(segidx[mask])]
            cal_idx, eval_idx = order[:k], order[k:]
            keep[np.flatnonzero(s_pool == s)] = False
            keep[eval_idx] = True
            for t in (0, 1):
                a, off = _affine_recalibration(p_pool[cal_idx, t], y_pool[cal_idx, t])
                corrected[eval_idx, t] = a * p_pool[eval_idx, t] + off
        m_dbp = metrics(y_pool[keep, 1], corrected[keep, 1], s_pool[keep])
        out[k] = dict(mae=m_dbp["mae"], aami_pass=m_dbp["aami_pass"], sde=m_dbp["sde"])
    return out


def original_sweep(m):
    """DBP {mae, aami_pass} at each k in SHOTS, from the saved same-time
    (storage-order first-k) personalization curve, pick='first'."""
    p = model_dir(m) / f"{m}_personalization_calfree.json"
    d = json.loads(p.read_text())
    curve = {row["shots"]: row["results"]["dbp"] for row in d["curves"]["first"]}
    return {k: dict(mae=curve[k]["mae"], aami_pass=curve[k]["aami_pass"],
                    sde=curve[k]["sde"]) for k in SHOTS if k in curve}


rows = []
print(f"{'model':14s} " + "".join(f"{'k=' + str(k):>16s}" for k in SHOTS))
print(f"{'':14s} " + "".join(f"{'same / true-early':>16s}" for _ in SHOTS))
print("-" * (14 + 16 * len(SHOTS)))
for m in ORDER:
    npz = model_dir(m) / f"{m}_calfree_predictions.npz"
    pj = model_dir(m) / f"{m}_personalization_calfree.json"
    if not npz.exists():
        print(f"{LABEL[m]:14s}  -- no calfree predictions yet, skipped --")
        continue
    if not pj.exists():
        print(f"{LABEL[m]:14s}  -- no personalization curve yet, skipped --")
        continue
    orig = original_sweep(m)
    early = true_early_sweep(m)
    cells = []
    for k in SHOTS:
        o, e = orig.get(k), early.get(k)
        os_ = f"{o['mae']:.2f}{'*' if o['aami_pass'] else ' '}" if o else "  n/a"
        es = f"{e['mae']:.2f}{'*' if e['aami_pass'] else ' '}" if e else "n/a"
        cells.append(f"{os_}/{es}")
    print(f"{LABEL[m]:14s} " + "".join(f"{c:>16s}" for c in cells))
    rows.append(dict(model=m, original=orig, true_early=early))

out_path = OVERVIEW / "personalization_early_vs_late.json"
out_path.write_text(json.dumps(rows, indent=2, default=float))
print(f"\n(* = AAMI pass)  wrote {out_path}")
