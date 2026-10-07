"""Item 2's clean re-run: CNN then ResNet, 50 epochs with lr_decay_epochs=20 so
the learning-rate schedule for the first 20 epochs is identical to the
original 20-epoch run — isolating "more training" from "a different schedule
shape". calfree, seed 0, balanced.

Run:  python3 scripts/run_schedule_fix.py
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_schedule_fix.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    # balanced only for this first pass (~9h/model on Mac MPS already) —
    # nobalance can follow once these land, same pattern with balance=False
    jobs = [(m, True) for m in ("cnn", "resnet")]
    log(f"=== schedule-fix queue starting, {len(jobs)} jobs ===")
    for i, (m, bal) in enumerate(jobs, 1):
        tag = "schedfix" if bal else "schedfix_nobalance"
        label = f"{m} calfree balance={bal} lr_decay_epochs=20 tag={tag}"
        log(f"[{i}/{len(jobs)}] START {label}")
        t0 = time.time()
        try:
            run(model=m, protocol="calfree", epochs=50, lr_decay_epochs=20,
                balance=bal, tag=tag, num_workers=4)
            log(f"[{i}/{len(jobs)}] OK {label} ({time.time()-t0:.0f}s)")
        except Exception:
            log(f"[{i}/{len(jobs)}] FAILED {label}\n{traceback.format_exc()}")
    log("=== schedule-fix queue finished ===")


if __name__ == "__main__":
    main()
