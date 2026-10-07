"""Test whether PulseDB's ECG, PPG and ABP channels are mutually time-aligned.

Motivation: pulse arrival time (R peak -> PPG landmark) is the classical feature
of cuffless BP estimation, so it matters whether the channels share a clock.

Three measurements:

1. R peak -> ABP foot. ABP is an invasive arterial line, so this delay is a real
   pulse transit time and should be tight and physiological (~0.1-0.2 s).
2. R peak -> PPG foot. Should be similar if the channels share a clock.
3. The ECG/PPG phase offset per segment, aggregated within and across subjects,
   using circular statistics. Cross-correlating two near-periodic signals is
   ambiguous modulo the beat period, so the offset is expressed as a phase
   (fraction of a beat) and summarised with the circular concentration R, which
   runs from 0 (uniformly spread) to 1 (one fixed offset).

Run:  python3 scripts/check_alignment.py [n_segments]
"""

import collections
import sys
import os

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import PULSEDB_SUBSETS                      # noqa: E402
from pulsedb_loader import PulseDBSubset, ECG, PPG, ABP  # noqa: E402
from fiducials import detect_rpeaks, detect_ppg_fiducials, FS  # noqa: E402

SUBSET = "calfree_test"


def abp_feet(abp):
    x = (abp - abp.min()) / (np.ptp(abp) + 1e-9)
    pk, _ = sps.find_peaks(x, distance=int(0.35 * FS), prominence=0.25)
    return np.array([a + int(np.argmin(x[a:b + 1]))
                     for a, b in zip(np.r_[0, pk[:-1]], pk)], dtype=int)


def circ_mean(ph):
    z = np.mean(np.exp(1j * np.asarray(ph)))
    return np.angle(z), np.abs(z)


def main(n=1200):
    d = PulseDBSubset(str(PULSEDB_SUBSETS[SUBSET]))
    rng = np.random.default_rng(3)
    idx = rng.choice(d.n, min(n, d.n), replace=False)
    sig = d.signals(idx, channels=(ECG, PPG, ABP))
    subj = d.subject[idx]

    r_abp, r_ppg = [], []
    seg_phase, seg_subj = [], []

    for j in range(len(idx)):
        ecg, ppg, abp = (sig[j, 0].astype(float), sig[j, 1].astype(float),
                         sig[j, 2].astype(float))
        rp, q = detect_rpeaks(ecg)
        if q < 0.95 or rp.size < 4:
            continue
        fid = detect_ppg_fiducials(ppg)
        if fid["quality"] < 0.95 or fid["foot"].size < 3:
            continue

        af = abp_feet(abp)
        for r in rp:
            nxt = af[af > r]
            if nxt.size:
                r_abp.append((nxt[0] - r) / FS)
            nxt = fid["foot"][fid["foot"] > r]
            if nxt.size:
                r_ppg.append((nxt[0] - r) / FS)

        rr = np.median(np.diff(rp)) / FS
        ph = [2 * np.pi * ((((fid["foot"][fid["foot"] > r][0] - r) / FS) % rr) / rr)
              for r in rp if (fid["foot"] > r).any()]
        if len(ph) >= 3:
            ang, R = circ_mean(ph)
            if R > 0.8:                      # offset stable within the segment
                seg_phase.append(ang)
                seg_subj.append(subj[j])

    print(f"subset={SUBSET}  sampled={len(idx)}  usable={len(seg_phase)}\n")
    for name, v in [("R -> ABP foot", np.array(r_abp)),
                    ("R -> PPG foot", np.array(r_ppg))]:
        print(f"{name:16s} n={v.size:6d}  median={np.median(v):.3f}s  "
              f"p25={np.percentile(v, 25):.3f}  p75={np.percentile(v, 75):.3f}")

    seg_phase = np.array(seg_phase)
    seg_subj = np.array(seg_subj)
    _, R_all = circ_mean(seg_phase)
    print(f"\nECG/PPG phase concentration across all segments: R={R_all:.3f}")

    per = collections.defaultdict(list)
    for s, p in zip(seg_subj, seg_phase):
        per[s].append(p)
    multi = {k: np.array(v) for k, v in per.items() if len(v) >= 5}
    Rw = [circ_mean(v)[1] for v in multi.values()]
    _, Rb = circ_mean([circ_mean(v)[0] for v in multi.values()])
    print(f"  within-subject  ({len(multi)} subjects): mean R={np.mean(Rw):.3f} "
          f"median R={np.median(Rw):.3f}")
    print(f"  between-subject                      : R={Rb:.3f}")
    print("\nInterpretation: a tight physiological R->ABP delay together with a "
          "long, scattered R->PPG delay means the PPG channel does not share the "
          "ECG/ABP clock. High within-subject and low across-subject concentration "
          "means the offset is a per-record property, so it cannot be removed "
          "globally, and the only available reference (ABP) is the label.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1200)
