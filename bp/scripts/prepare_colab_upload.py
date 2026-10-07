"""What to copy to Google Drive so the Colab notebooks can train, and a check
that it arrived intact.

Nothing is duplicated locally — the processed arrays are 3.6 GB and the disk has
less headroom than that. Instead this prints the exact folders to drag across and
writes a manifest of sizes, which the notebooks re-check on the Drive copy. A
half-finished upload is otherwise invisible until a run dies twenty minutes in.

Run:  python3 scripts/prepare_colab_upload.py          # list and write manifest
      python3 scripts/prepare_colab_upload.py --verify # check a copy (on Colab)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import ROOT, PROC_PULSEDB, PROC_DALIA, FEATURES     # noqa: E402

MANIFEST = ROOT / "data" / "colab_manifest.json"

# repo-relative paths, in upload order — code first, so a partial upload still
# lets the notebook run and tell you what is missing
WANTED = [
    "src",
    "data/features",
    "data/processed/pulsedb",
    "data/processed/ppg_dalia/windows",
]


def _walk(rel):
    """(relative path, size) for every file under one wanted entry."""
    base = ROOT / rel
    if base.is_file():
        yield rel, base.stat().st_size
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in sorted(filenames):
            if f.startswith("."):
                continue
            p = os.path.join(dirpath, f)
            yield os.path.relpath(p, ROOT), os.path.getsize(p)


def build():
    entries, total = {}, 0
    print(f"{'folder':40s} {'files':>7s} {'size':>10s}")
    print("-" * 60)
    for rel in WANTED:
        if not (ROOT / rel).exists():
            print(f"{rel:40s} {'MISSING':>7s}")
            continue
        files = dict(_walk(rel))
        entries.update(files)
        n, size = len(files), sum(files.values())
        total += size
        print(f"{rel:40s} {n:>7,} {size / 1e9:>9.2f} GB")
    print("-" * 60)
    print(f"{'TOTAL':40s} {len(entries):>7,} {total / 1e9:>9.2f} GB")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(files=entries, total_bytes=total), indent=1))
    print(f"\nmanifest -> {MANIFEST.relative_to(ROOT)}")

    print(f"""
Upload
  1. Create a folder in Drive:   MyDrive/bp
  2. Drag these folders into it, keeping the same structure:

        MyDrive/bp/src/
        MyDrive/bp/data/features/
        MyDrive/bp/data/processed/pulsedb/
        MyDrive/bp/data/processed/ppg_dalia/windows/
        MyDrive/bp/data/colab_manifest.json

  3. The notebook verifies on its own whether everything arrived.

Note: src/ is small ({sum(s for f, s in entries.items() if f.startswith('src')) / 1e6:.1f} MB)
      — only re-upload that when the code changes, not the data.""")
    return entries


def verify(root=None):
    """Check a copy against the manifest. Returns the list of problems.

    Sizes are only enforced for the data files. Source files change every time
    the code changes, and a stale manifest then fails an upload that is actually
    fine — three times so far. What matters is that the 3.8 GB of arrays arrived
    whole; code is checked for presence only.
    """
    from pathlib import Path
    base = Path(root) if root else ROOT
    man = json.loads((base / "data" / "colab_manifest.json").read_text())
    missing, wrong = [], []
    for rel, size in man["files"].items():
        if not rel.startswith("data/"):
            if not (base / rel).exists():
                missing.append(rel)
            continue
        p = base / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != size:
            wrong.append((rel, size, p.stat().st_size))

    n = len(man["files"])
    if not missing and not wrong:
        print(f"OK — all {n:,} files present, {man['total_bytes'] / 1e9:.2f} GB")
        return []
    print(f"PROBLEM — {len(missing)} missing, {len(wrong)} wrong size, of {n:,}")
    for rel in missing[:10]:
        print(f"  missing   {rel}")
    for rel, want, got in wrong[:10]:
        print(f"  size      {rel}  expected {want:,} got {got:,}")
    if len(missing) + len(wrong) > 20:
        print("  ... (truncated)")
    return missing + [w[0] for w in wrong]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    verify(a.root) if a.verify else build()
