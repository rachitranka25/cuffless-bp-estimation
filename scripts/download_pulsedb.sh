#!/bin/bash
# Downloads PulseDB official VitalDB subsets from the authors' Kaggle mirror.
# Source: https://github.com/pulselabteam/PulseDB  (DOI 10.34740/KAGGLE/DS/2447469)
# License: CC BY-NC-SA 4.0
# Each file downloads as a .zip, is extracted, then the .zip is deleted to save disk.

set -u
BASE="https://www.kaggle.com/api/v1/datasets/download/weinanwangrutgers/pulsedb-balanced-training-and-testing"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/raw/pulsedb"
mkdir -p "$OUT"

get() {
  local name="$1"
  if [ -f "$OUT/$name" ]; then
    echo "[skip] $name already present"
    return
  fi
  echo "[get ] $name"
  curl -fL -C - --retry 5 --retry-delay 5 \
       -o "$OUT/$name.zip" "$BASE?file_name=$name" || { echo "[FAIL] $name"; return 1; }
  unzip -o -q "$OUT/$name.zip" -d "$OUT" && rm -f "$OUT/$name.zip"
  echo "[ok  ] $name"
}

# Order: smallest first, so useful data lands early.
get "VitalDB_AAMI_Test_Subset.mat"      #  ~20 MB - AAMI-compliant held-out test
get "VitalDB_CalBased_Test_Subset.mat"  # ~1.5 GB - calibration-based test (personalization)
get "VitalDB_CalFree_Test_Subset.mat"   # ~1.7 GB - calibration-free test (honest baseline)
get "VitalDB_AAMI_Cal_Subset.mat"       # ~2.1 GB - AAMI calibration segments
get "VitalDB_Train_Subset.mat"          # ~13.8 GB - needs ~28 GB free (zip + extract).
                                        # To halve that, stream instead:
                                        #   curl -fL "$BASE?file_name=VitalDB_Train_Subset.mat" \
                                        #     | tar -x -C "$OUT" -f -

echo "Done. Contents of $OUT:"
ls -lh "$OUT"
