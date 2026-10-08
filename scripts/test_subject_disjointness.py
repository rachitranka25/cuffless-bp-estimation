"""Standalone reproducibility check: verifies every protocol's train/test
subject split has the leakage property it claims to have, without training
any model. Runs in seconds — it only loads subject-ID arrays, the same
_assert_split_sanity() check that already runs inside build_protocol() on
every training call, surfaced here as an independent, CI-friendly script a
reviewer can run without touching a GPU or the 20 GB raw dataset.

Exits non-zero (and prints FAIL) if any protocol's disjointness guarantee is
violated. leaky and calbased are expected, by protocol design, to share
subjects between train and test — only calfree and aami are required to be
fully subject-disjoint.

Run:  python3 scripts/test_subject_disjointness.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from train import build_protocol, PROTOCOLS, _subjects

EXPECT_DISJOINT = {"leaky": False, "calbased": False, "calfree": True, "aami": True}

failures = []
print(f"{'protocol':10s} {'train subj':>10s} {'test subj':>9s} {'shared':>7s} {'expected':>18s}  result")
print("-" * 72)

for protocol in PROTOCOLS:
    try:
        b = build_protocol(protocol, seed=0)
    except AssertionError as e:
        print(f"{protocol:10s} FAIL — build_protocol itself raised: {e}")
        failures.append(protocol)
        continue

    tr_subj = _subjects(b["train_split"])
    te_subj = _subjects(b["test_split"])
    tr = set(tr_subj[b["train_idx"]].tolist())
    te = set(te_subj[b["test_idx"]].tolist())
    shared = len(tr & te)

    want_disjoint = EXPECT_DISJOINT[protocol]
    ok = (shared == 0) if want_disjoint else True   # leaky/calbased: any count is fine
    label = "disjoint required" if want_disjoint else "overlap by design"
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(protocol)
    print(f"{protocol:10s} {len(tr):>10,} {len(te):>9,} {shared:>7,} {label:>18s}  {status}")

print("-" * 72)
if failures:
    print(f"FAIL: {', '.join(failures)} violated their expected disjointness property.")
    sys.exit(1)
print("PASS: every protocol's train/test subject split matches its documented guarantee.")
