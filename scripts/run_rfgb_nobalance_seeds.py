"""RF/GB, calfree_nobalance, seeds 1 and 2 — the balanced seeds already ran."""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_rfgb_nobalance.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


jobs = [(m, s) for s in (1, 2) for m in ("rf", "gb")]
log(f"=== rf/gb nobalance seed queue starting, {len(jobs)} jobs ===")
for i, (m, seed) in enumerate(jobs, 1):
    label = f"{m} calfree_nobalance seed={seed}"
    log(f"[{i}/{len(jobs)}] START {label}")
    t0 = time.time()
    try:
        run(model=m, protocol="calfree", seed=seed, balance=False, tag=f"nobalance_seed{seed}")
        log(f"[{i}/{len(jobs)}] OK {label} ({time.time()-t0:.0f}s)")
    except Exception:
        log(f"[{i}/{len(jobs)}] FAILED {label}\n{traceback.format_exc()}")
log("=== rf/gb nobalance seed queue finished ===")
