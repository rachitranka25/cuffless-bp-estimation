"""Extract hand-crafted features for every PulseDB subset into features/*.parquet.

Segments are read from the HDF5 in contiguous blocks (random access into a
13.8 GB file is far slower than sequential reads), then the per-segment feature
computation is fanned out across processes.

Run:  python3 scripts/extract_features.py [subset ...]
      python3 scripts/extract_features.py            # all subsets
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import PULSEDB_SUBSETS, FEATURES          # noqa: E402
from pulsedb_loader import PulseDBSubset             # noqa: E402
from features import extract                         # noqa: E402

BLOCK = 4000
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def _one(ecg, ppg, demo):
    return extract(ecg, ppg, demo)


def run(name):
    path = PULSEDB_SUBSETS[name]
    out = FEATURES / f"{name}.parquet"
    d = PulseDBSubset(str(path))
    print(f"{name}: {d.n} segments, {len(d.subjects)} subjects -> {out.name}")

    rows = []
    t0 = time.time()
    for start in range(0, d.n, BLOCK):
        stop = min(start + BLOCK, d.n)
        raw = d._signals[:, :, start:stop]              # (1250, 3, B) sequential read
        ecg = np.ascontiguousarray(raw[:, 0, :].T, dtype=np.float64)
        ppg = np.ascontiguousarray(raw[:, 1, :].T, dtype=np.float64)
        demos = [dict(age=d.age[i], gender=d.gender[i], height=d.height[i],
                      weight=d.weight[i], bmi=d.bmi[i]) for i in range(start, stop)]
        rows.extend(Parallel(n_jobs=N_JOBS, batch_size=64)(
            delayed(_one)(ecg[k], ppg[k], demos[k]) for k in range(stop - start)))
        done = stop
        el = time.time() - t0
        print(f"\r  {done}/{d.n}  {el:6.0f}s elapsed, "
              f"{el / done * (d.n - done):6.0f}s left", end="", flush=True)
    print()

    df = pd.DataFrame(rows)
    df.insert(0, "subject", d.subject)
    df.insert(1, "sbp", d.sbp)
    df.insert(2, "dbp", d.dbp)
    FEATURES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    d.close()
    print(f"  wrote {out}  {df.shape[0]} rows x {df.shape[1]} cols  "
          f"{out.stat().st_size / 1e6:.0f} MB  in {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    names = sys.argv[1:] or list(PULSEDB_SUBSETS)
    for n in names:
        run(n)
