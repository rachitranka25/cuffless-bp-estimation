"""RF/GB, calfree, seeds 1 and 2 — fast, local, no GPU needed."""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_rfgb.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


jobs = [(m, s) for s in (1, 2) for m in ("rf", "gb")]
log(f"=== rf/gb seed queue starting, {len(jobs)} jobs ===")
for i, (m, seed) in enumerate(jobs, 1):
    log(f"[{i}/{len(jobs)}] START {m} calfree seed={seed}")
    t0 = time.time()
    try:
        run(model=m, protocol="calfree", seed=seed, tag=f"seed{seed}")
        log(f"[{i}/{len(jobs)}] OK {m} seed={seed} ({time.time()-t0:.0f}s)")
    except Exception:
        log(f"[{i}/{len(jobs)}] FAILED {m} seed={seed}\n{traceback.format_exc()}")
log("=== rf/gb seed queue finished ===")
