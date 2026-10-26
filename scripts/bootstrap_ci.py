"""Patient-level bootstrap confidence intervals and a formal significance
test for the architecture-dependent calibration-free leakage-gap claim.

The three-seed confidence intervals already reported elsewhere in this study
(results/overview/RESULTS.md, Table XII of the manuscript) capture training
stochasticity — would a different random initialization/data order change the
result. They say nothing about test-set sampling uncertainty — would a
different draw of the same number of patients from the same population give a
different MAE. This script adds that second, independent source of
uncertainty via a patient-level cluster bootstrap over the saved per-clip
predictions (results/models/<model>/<model>_<protocol>_predictions.npz),
without retraining anything.

Two things are reported:

1. A 95% bootstrap CI on calibration-free (weighted) SBP/DBP MAE, per model —
   resample patients with replacement from the test set, keep every clip
   belonging to a resampled patient (so a patient's clips move together,
   respecting the clustering MAE is actually computed over), recompute MAE,
   repeat N_BOOT times, take percentiles.

2. A paired bootstrap test of this study's central claim — that the
   calibration-free leakage gap is architecture-dependent — specifically for
   its two extremes, ResNet1D (largest gap) and the Transformer
   (smallest gap). Each bootstrap iteration draws one patient resample
   per test set (calfree, leaky, calbased) and reuses the SAME resample
   indices for both architectures before computing each one's gap, which is
   the paired design: it cancels shared test-set sampling variance and
   isolates the question "do these two architectures' gaps actually differ,
   or could sampling alone explain it." If the resulting 95% CI on
   gap_resnet - gap_transformer excludes zero, the difference is significant
   at the 5% level under this resampling scheme.

Run:  python3 scripts/bootstrap_ci.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from paths import MODEL_RESULTS

MODEL_ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
MODEL_LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
               "resnet": "ResNet1D", "transformer": "Transformer"}
N_BOOT = 2000
RNG_SEED = 0

# CNN/ResNet1D's plain calfree checkpoint is schedule-confounded (see the
# OneCycleLR-confound analysis); the canonical calfree result used everywhere
# else for these two models is the schedule-fixed checkpoint instead.
CALFREE_CANONICAL_TAG = {"cnn": "schedfix", "resnet": "schedfix"}


def load(model, protocol):
    tag = CALFREE_CANONICAL_TAG.get(model, "") if protocol == "calfree" else ""
    suffix = f"_{tag}" if tag else ""
    p = MODEL_RESULTS / model / f"{model}_{protocol}{suffix}_predictions.npz"
    z = np.load(p, allow_pickle=True)
    return z["y_true"], z["y_pred"], z["subjects"]


def mae(y_true, y_pred, idx, target):
    k = 0 if target == "sbp" else 1
    return np.abs(y_pred[idx, k] - y_true[idx, k]).mean()


# Exactly which (model, protocol) cells Table IV reports as a 3-seed mean
# (marked *) vs a single run — this is cell-specific, not uniform per model:
# calfree is 3-seed for rf/gb/transformer but a single schedule-fixed run
# (marked †, deliberately not seed-averaged) for cnn/resnet; leaky/calbased
# is 3-seed for cnn/resnet/transformer (extra seeds were run for all three
# to check the architecture-dependence finding is not a seed artefact,
# transformer's included since it sets the headline minimum gap) but a
# single seed-0 run for rf/gb, which were never re-seeded on leaky/calbased.
# Guessing a uniform per-model rule here was exactly the earlier bug (it
# silently 3-seed-averaged cnn/resnet's calfree using the wrong,
# non-schedule-fixed seed1/seed2 files) — this table is transcribed
# directly from Table IV's own */† annotations instead.
THREE_SEED_CELLS = {
    ("rf", "calfree"), ("gb", "calfree"), ("transformer", "calfree"),
    ("cnn", "leaky"), ("cnn", "calbased"),
    ("resnet", "leaky"), ("resnet", "calbased"),
    ("transformer", "leaky"), ("transformer", "calbased"),
}


def canonical_seen_patient_mae(model, protocol, target):
    """The exact Table-IV value for this cell: 3-seed mean where Table IV
    marks it with *, else the single canonical run (schedule-fixed where
    applicable, via load()'s CALFREE_CANONICAL_TAG)."""
    if (model, protocol) not in THREE_SEED_CELLS:
        y_true, y_pred, _ = load(model, protocol)
        return mae(y_true, y_pred, np.arange(len(y_true)), target)
    vals = []
    for tag in ("", "seed1", "seed2"):
        p = MODEL_RESULTS / model / f"{model}_{protocol}{'_' + tag if tag else ''}_predictions.npz"
        z = np.load(p, allow_pickle=True)
        y_true, y_pred = z["y_true"], z["y_pred"]
        vals.append(mae(y_true, y_pred, np.arange(len(y_true)), target))
    return float(np.mean(vals))


def patient_groups(subjects):
    """Precompute {patient_id: clip_indices} once, reused across every
    bootstrap draw for this (model, protocol) — avoids rebuilding it
    N_BOOT times."""
    uniq = np.unique(subjects)
    return uniq, {s: np.flatnonzero(subjects == s) for s in uniq}


def resample_indices(uniq, groups, rng):
    """One cluster-bootstrap draw: resample unique patients with replacement,
    return the (possibly repeated) clip indices belonging to the draw."""
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([groups[s] for s in drawn])


def bootstrap_mae_ci(y_true, y_pred, subjects, target, canonical_point, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    seed0_point = mae(y_true, y_pred, np.arange(len(subjects)), target)
    uniq, groups = patient_groups(subjects)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = resample_indices(uniq, groups, rng)
        draws[b] = mae(y_true, y_pred, idx, target)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # recentre on the canonical (Table-IV-matching) point estimate, keeping
    # the bootstrap-estimated width — see canonical_seen_patient_mae's note
    offset = canonical_point - seed0_point
    return canonical_point, lo + offset, hi + offset


print("=" * 78)
print("1. Patient-level bootstrap 95% CI, calibration-free (weighted) MAE")
print("=" * 78)
print(f"{'Model':18s} {'SBP MAE [95% CI]':28s} {'DBP MAE [95% CI]':28s}")
for m in MODEL_ORDER:
    y_true, y_pred, subjects = load(m, "calfree")
    sbp_canon = canonical_seen_patient_mae(m, "calfree", "sbp")
    dbp_canon = canonical_seen_patient_mae(m, "calfree", "dbp")
    sbp_pt, sbp_lo, sbp_hi = bootstrap_mae_ci(y_true, y_pred, subjects, "sbp", sbp_canon)
    dbp_pt, dbp_lo, dbp_hi = bootstrap_mae_ci(y_true, y_pred, subjects, "dbp", dbp_canon)
    print(f"{MODEL_LABEL[m]:18s} "
          f"{sbp_pt:5.2f} [{sbp_lo:5.2f}, {sbp_hi:5.2f}]{'':6s} "
          f"{dbp_pt:5.2f} [{dbp_lo:5.2f}, {dbp_hi:5.2f}]")

print()
print("=" * 78)
print("2. Bootstrap CI on the calibration-free leakage gap, per model")
print("   gap = calfree MAE / mean(leaky MAE, calbased MAE)")
print("=" * 78)


def gap_draw(data, groups, target, rng):
    """One bootstrap draw of the gap for one model: resample patients
    independently within each of the three test sets, compute the ratio."""
    gaps = {}
    for key, (y_true, y_pred, subjects) in data.items():
        uniq, grp = groups[key]
        idx = resample_indices(uniq, grp, rng)
        gaps[key] = mae(y_true, y_pred, idx, target)
    return gaps["calfree"] / ((gaps["leaky"] + gaps["calbased"]) / 2)


for target in ("sbp", "dbp"):
    print(f"\n{target.upper()}:")
    for m in MODEL_ORDER:
        data = {proto: load(m, proto) for proto in ("calfree", "leaky", "calbased")}
        groups = {proto: patient_groups(v[2]) for proto, v in data.items()}
        full = lambda yt, yp: mae(yt, yp, np.arange(len(yt)), target)

        # canonical point estimate (matches Table IV: 3-seed mean where it exists)
        point_gap = canonical_seen_patient_mae(m, "calfree", target) / (
            (canonical_seen_patient_mae(m, "leaky", target)
             + canonical_seen_patient_mae(m, "calbased", target)) / 2)
        # seed-0-only point, for recentring the bootstrap draws below (the
        # patient resample is only computed against seed 0's split; shifting
        # by the seed-0-vs-canonical offset keeps the bootstrap's estimated
        # width while centring the interval on the canonical point estimate)
        seed0_point = full(*data["calfree"][:2]) / (
            (full(*data["leaky"][:2]) + full(*data["calbased"][:2])) / 2)

        rng = np.random.default_rng(RNG_SEED)
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            draws[b] = gap_draw(data, groups, target, rng)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        offset = point_gap - seed0_point
        lo, hi = lo + offset, hi + offset
        print(f"  {MODEL_LABEL[m]:18s} gap = {point_gap:.2f}x  [{lo:.2f}x, {hi:.2f}x]")

print()
print("=" * 78)
print("3. Paired significance test: ResNet1D (largest gap) vs Transformer")
print("   (smallest gap) — is the architecture-dependence claim significant?")
print("=" * 78)


def paired_gap_diff(model_a, model_b, target, n_boot=N_BOOT, seed=RNG_SEED):
    data_a = {proto: load(model_a, proto) for proto in ("calfree", "leaky", "calbased")}
    data_b = {proto: load(model_b, proto) for proto in ("calfree", "leaky", "calbased")}
    # same patients/clips across models for a given protocol (verified: both
    # models trained with the same seed, hence the same split) — precompute
    # the groups once from model_a's subjects and reuse for model_b too.
    groups = {proto: patient_groups(data_a[proto][2]) for proto in data_a}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        # one shared resample per test set, reused for both models — paired design
        resampled = {}
        for proto in ("calfree", "leaky", "calbased"):
            uniq, grp = groups[proto]
            resampled[proto] = resample_indices(uniq, grp, rng)
        gap_a = mae(*data_a["calfree"][:2], resampled["calfree"], target) / (
            (mae(*data_a["leaky"][:2], resampled["leaky"], target)
             + mae(*data_a["calbased"][:2], resampled["calbased"], target)) / 2)
        gap_b = mae(*data_b["calfree"][:2], resampled["calfree"], target) / (
            (mae(*data_b["leaky"][:2], resampled["leaky"], target)
             + mae(*data_b["calbased"][:2], resampled["calbased"], target)) / 2)
        diffs[b] = gap_a - gap_b
    return diffs


for target in ("sbp", "dbp"):
    diffs = paired_gap_diff("resnet", "transformer", target)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    canonical_gap = lambda m: canonical_seen_patient_mae(m, "calfree", target) / (
        (canonical_seen_patient_mae(m, "leaky", target)
         + canonical_seen_patient_mae(m, "calbased", target)) / 2)
    canonical_diff = canonical_gap("resnet") - canonical_gap("transformer")
    offset = canonical_diff - diffs.mean()
    lo, hi = lo + offset, hi + offset

    sig = "excludes zero -> significant at 5%" if (lo > 0 or hi < 0) else "includes zero -> not significant at 5%"
    print(f"{target.upper()}: gap_ResNet1D - gap_Transformer = "
          f"{canonical_diff:+.2f}x  95% CI [{lo:+.2f}x, {hi:+.2f}x]  ({sig})")
