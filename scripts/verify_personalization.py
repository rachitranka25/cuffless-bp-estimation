"""Independent re-verification of the personalization numbers reported in the paper.

Recomputes every personalization cell directly from the saved, released
per-clip predictions (no retraining), using the exact offset-only (k<3) /
30%-shrunk-slope (k>=3) correction of Section 3.6
(src/train.py:_affine_recalibration), under both calibration/evaluation
splits, and diffs each recomputed value against the value printed in the paper:

  Part 1  naive (storage-order first-k) split, via src.train.personalize()
          -> Table 12 naive column, Table 17 (full k-sweep), and the oracle
             ceiling of Table 9
  Part 2  chronological split (earliest-k clips by segment index calibrate,
          all later clips evaluate) -> Table 10 (MAE/SDE, full k-sweep) and
          Table 12 chronological column

Oracle ceiling (Table 9): each subject receives the single best constant
offset (the median of its own errors, fit on its own test labels) -- an
unreachable reference for how much error any per-subject offset could remove.
It is computed by src/train.py:offset_headroom.

AAMI flag: the AAMI/ISO 81060-2 criterion (|mean error| <= 5 mmHg AND error SD
<= 8 mmHg) is evaluated separately for SBP and for DBP, and both are printed.
The paper's personalization AAMI verdicts (Table 12) refer to DBP.

Run: python3 scripts/verify_personalization.py
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
from src.train import personalize, _affine_recalibration
from src.evaluation import metrics
from src.paths import model_dir, PROC_PULSEDB

MODELS = ["rf", "gb", "cnn", "resnet", "transformer"]
LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
         "resnet": "ResNet1D", "transformer": "Transformer"}
# CNN/ResNet1D's canonical calfree number uses the schedule-fixed checkpoint.
BASE_TAG = {"cnn": "schedfix", "resnet": "schedfix"}
SHOTS = (1, 3, 5, 10, 25)

# ---- values as printed in the paper ---------------------------------------
# Table 9: DBP/SBP raw -> oracle MAE and % recovered
PAPER_T9 = {
    "rf":          dict(sbp=(12.76, 8.48, 33), dbp=(8.38, 4.98, 41)),
    "gb":          dict(sbp=(14.63, 8.77, 40), dbp=(9.00, 5.15, 43)),
    "cnn":         dict(sbp=(14.89, 8.87, 40), dbp=(9.04, 4.90, 46)),
    "resnet":      dict(sbp=(13.94, 9.13, 35), dbp=(8.94, 5.40, 40)),
    "transformer": dict(sbp=(15.44, 9.34, 39), dbp=(8.47, 4.94, 42)),
}
# Table 17: naive-split DBP MAE at k = 1, 3, 5, 10, 25
PAPER_T17 = {
    "rf":          (7.17, 7.32, 5.53, 5.00, 4.81),
    "gb":          (7.40, 6.77, 5.41, 4.90, 4.74),
    "cnn":         (6.88, 6.40, 5.11, 4.80, 4.55),
    "resnet":      (7.86, 6.58, 5.18, 4.92, 4.77),
    "transformer": (6.61, 6.54, 4.96, 4.56, 4.40),
}
# Table 10: chronological-split DBP (MAE, SDE) at k = 1, 3, 5, 10, 25
PAPER_T10 = {
    "rf":          ((9.58, 12.74), (11.97, 18.96), (9.46, 12.68), (8.65, 11.86), (7.32, 9.68)),
    "gb":          ((9.11, 11.84), (13.11, 27.74), (9.23, 12.32), (7.95, 10.53), (7.11, 9.47)),
    "cnn":         ((9.79, 12.73), (11.37, 16.85), (9.03, 11.93), (8.02, 10.67), (7.15, 9.44)),
    "resnet":      ((10.45, 13.36), (10.56, 16.28), (9.23, 12.82), (7.98, 10.51), (7.31, 9.54)),
    "transformer": ((9.60, 12.59), (10.38, 14.81), (8.38, 11.23), (7.60, 10.08), (6.68, 8.75)),
}
# Table 12: DBP MAE at k=0, naive k=25 + AAMI, chronological k=25 + AAMI
PAPER_T12 = {
    "rf":          (8.38, 4.81, "PASS", 7.32, "FAIL"),
    "gb":          (9.00, 4.74, "PASS", 7.11, "FAIL"),
    "cnn":         (9.04, 4.55, "PASS", 7.15, "FAIL"),
    "resnet":      (8.94, 4.77, "PASS", 7.31, "FAIL"),
    "transformer": (8.47, 4.40, "PASS", 6.68, "FAIL"),
}

n_checked = 0
mismatches = []


def check(where, got, want):
    """Compare at the paper's printed precision."""
    global n_checked
    n_checked += 1
    if isinstance(want, str):
        ok = got == want
    elif isinstance(want, int):
        ok = round(got) == want
    else:
        ok = f"{got:.2f}" == f"{want:.2f}"
    if not ok:
        mismatches.append(f"{where}: recomputed {got}, paper {want}")


def flag(m):
    return "PASS" if m["aami_pass"] else "FAIL"


segidx = np.load(PROC_PULSEDB / "calfree_test" / "segidx.npy")
valid = ~np.isnan(segidx)


def chronological_sweep(model):
    """Earliest-k clips (by segment index) calibrate, every later clip of the
    same subject is scored. Same correction as the naive split."""
    tag = BASE_TAG.get(model, "")
    d = np.load(model_dir(model) / f"{model}_calfree{'_' + tag if tag else ''}_predictions.npz")
    p_pool, y_pool, s_pool = d["y_pred"], d["y_true"], d["subjects"]
    out = {}
    for k in SHOTS:
        corrected = p_pool.copy()
        keep = np.zeros(len(p_pool), bool)
        for s in np.unique(s_pool):
            mask = np.flatnonzero(s_pool == s)
            mask = mask[valid[mask]]
            if mask.size < k + 5:
                continue
            order = mask[np.argsort(segidx[mask])]
            cal_idx, eval_idx = order[:k], order[k:]
            keep[eval_idx] = True
            for t in (0, 1):
                a, off = _affine_recalibration(p_pool[cal_idx, t], y_pool[cal_idx, t])
                corrected[eval_idx, t] = a * p_pool[eval_idx, t] + off
        out[k] = dict(n=int(keep.sum()),
                      sbp=metrics(y_pool[keep, 0], corrected[keep, 0], s_pool[keep]),
                      dbp=metrics(y_pool[keep, 1], corrected[keep, 1], s_pool[keep]))
    return out


# ============================================================================
print("=" * 90)
print("PART 1 -- naive (storage-order first-k) split, via src.train.personalize()")
print("(offset-only k<3, 30%-shrunk-slope k>=3 -- src/train.py:_affine_recalibration)")
print("=" * 90)

naive = {}
for model in MODELS:
    res = personalize(model=model, source="calfree", shots=(0,) + SHOTS,
                      picks=("first",), save=False, base_tag=BASE_TAG.get(model, ""))
    naive[model] = res
    rows = {r["shots"]: r["results"] for r in res["curves"]["first"]}
    for t in ("sbp", "dbp"):
        h = res["headroom"][t]
        raw, orc, pct = PAPER_T9[model][t]
        check(f"Table 9 {LABEL[model]} {t.upper()} raw", h["mae"], raw)
        check(f"Table 9 {LABEL[model]} {t.upper()} oracle", h["oracle_offset_mae"], orc)
        check(f"Table 9 {LABEL[model]} {t.upper()} recovered %", h["recoverable_frac"] * 100, pct)
    for i, k in enumerate(SHOTS):
        check(f"Table 17 {LABEL[model]} k={k}", rows[k]["dbp"]["mae"], PAPER_T17[model][i])
    k0, nv, nv_flag, _, _ = PAPER_T12[model]
    check(f"Table 12 {LABEL[model]} k=0", rows[0]["dbp"]["mae"], k0)
    check(f"Table 12 {LABEL[model]} naive k=25", rows[25]["dbp"]["mae"], nv)
    check(f"Table 12 {LABEL[model]} naive AAMI (DBP)", flag(rows[25]["dbp"]), nv_flag)

# ============================================================================
print()
print("=" * 90)
print("PART 2 -- chronological split (earliest-k clips calibrate, later clips evaluate)")
print("=" * 90)

for model in MODELS:
    sweep = chronological_sweep(model)
    print(f"\n=== chronological: {model}, base_tag={BASE_TAG.get(model, '')!r} ===")
    for i, k in enumerate(SHOTS):
        r = sweep[k]
        print(f"    k={k:<3} n={r['n']:>6,}  "
              f"SBP MAE {r['sbp']['mae']:6.2f}  "
              f"DBP MAE {r['dbp']['mae']:6.2f}  DBP SDE {r['dbp']['sde']:6.2f}  "
              f"DBP ME {r['dbp']['me']:+5.2f}  "
              f"AAMI(|ME|<=5, SDE<=8) SBP {flag(r['sbp'])} DBP {flag(r['dbp'])}")
        mae, sde = PAPER_T10[model][i]
        check(f"Table 10 {LABEL[model]} k={k} MAE", r["dbp"]["mae"], mae)
        check(f"Table 10 {LABEL[model]} k={k} SDE", r["dbp"]["sde"], sde)
    _, _, _, ch, ch_flag = PAPER_T12[model]
    check(f"Table 12 {LABEL[model]} chron. k=25", sweep[25]["dbp"]["mae"], ch)
    check(f"Table 12 {LABEL[model]} chron. AAMI (DBP)", flag(sweep[25]["dbp"]), ch_flag)

# ============================================================================
print()
print("=" * 90)
print("RESULT -- recomputed values vs. values printed in Tables 9, 10, 12 and 17")
print("=" * 90)
print(f"cells checked: {n_checked}")
if mismatches:
    print(f"MISMATCHES: {len(mismatches)}")
    for line in mismatches:
        print("  " + line)
    sys.exit(1)
print("all cells match the paper exactly (at the paper's printed precision)")
