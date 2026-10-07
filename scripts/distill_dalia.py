"""Distill PPG-DaLiA into compact per-subject .npz files.

The released .pkl files are ~1.4 GB each (20 GB total) because they carry the
full 700 Hz chest sensor suite from the stress-detection side of the protocol.
Project B needs the wrist PPG, wrist accelerometer, chest ECG, and the label
streams; EMG / EDA / Temp / Respiration / chest ACC are dropped.

One .pkl is unpacked from data.zip at a time and deleted straight after, so peak
disk use stays around 1.5 GB instead of 20 GB. data.zip is left untouched, so
any dropped channel can be recovered later.

Sampling rates (from PPG_FieldStudy_readme.pdf):
    chest ECG     700 Hz
    wrist BVP      64 Hz
    wrist ACC      32 Hz
    activity        4 Hz
    HR label      0.5 Hz  (8 s window, 2 s shift)
    rpeaks        sample indices into the 700 Hz chest ECG
"""

import os
import pickle
import shutil
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import RAW_DALIA, PROC_DALIA

ZIP = str(RAW_DALIA / "data.zip")
OUT = str(PROC_DALIA)
TMP = str(RAW_DALIA / "_tmp")
SUBJECTS = [f"S{i}" for i in range(1, 16)]

FS = dict(ecg=700, bvp=64, acc=32, activity=4, label=0.5)


def distill(subject):
    dest = os.path.join(OUT, f"{subject}.npz")
    if os.path.exists(dest):
        print(f"[skip] {subject}")
        return

    os.makedirs(TMP, exist_ok=True)
    subprocess.run(
        ["unzip", "-o", "-q", "-j", ZIP, f"PPG_FieldStudy/{subject}/{subject}.pkl",
         "-d", TMP],
        check=True,
    )
    pkl = os.path.join(TMP, f"{subject}.pkl")

    with open(pkl, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")

    q = d.get("questionnaire", {}) or {}
    np.savez_compressed(
        dest,
        ecg=np.asarray(d["signal"]["chest"]["ECG"], dtype=np.float32).ravel(),
        bvp=np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float32).ravel(),
        acc=np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float32),
        rpeaks=np.asarray(d["rpeaks"], dtype=np.int64),
        hr=np.asarray(d["label"], dtype=np.float32).ravel(),
        activity=np.asarray(d["activity"], dtype=np.int8).ravel(),
        subject=np.array(str(d.get("subject", subject))),
        # demographics let us relate PPG-DaLiA to PulseDB's age/sex distribution
        questionnaire=np.array(
            [f"{k}={v}" for k, v in sorted(q.items())], dtype=object
        ),
        fs=np.array([FS[k] for k in ("ecg", "bvp", "acc", "activity", "label")],
                    dtype=np.float32),
        fs_names=np.array(["ecg", "bvp", "acc", "activity", "label"]),
    )

    os.remove(pkl)
    mb = os.path.getsize(dest) / 1e6
    print(f"[ok  ] {subject}  ecg={d['signal']['chest']['ECG'].shape[0]:>9d}  "
          f"bvp={d['signal']['wrist']['BVP'].shape[0]:>8d}  "
          f"hr={len(d['label']):>6d}  ->  {mb:6.1f} MB")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in SUBJECTS:
        distill(s)
    shutil.rmtree(TMP, ignore_errors=True)
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\nDone. {len(os.listdir(OUT))} files, {total/1e6:.0f} MB total in {OUT}")
