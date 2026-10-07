"""Assemble one folder to drag into Google Drive.

Builds `bp/` at the project root with exactly what the Colab notebooks need, in
the layout they expect. Files are hard-linked rather than copied: the bundle is
3.8 GB and the disk has less headroom than that, and a hard link costs a
directory entry. The data files are never written to, so sharing inodes with the
originals is safe — deleting `bp/` afterwards leaves the project untouched.

`--copy` forces real copies if the staging folder has to live on another disk.

Run:  python3 scripts/make_drive_folder.py
      python3 scripts/make_drive_folder.py --copy
"""

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from paths import ROOT      # noqa: E402

OUT = ROOT / "bp"

# (source relative to ROOT, destination relative to OUT)
ITEMS = [
    ("HANDOFF.md", "HANDOFF.md"),
    ("src", "src"),
    ("scripts/prepare_colab_upload.py", "scripts/prepare_colab_upload.py"),
    ("notebooks", "notebooks"),
    ("data/features", "data/features"),
    ("data/processed/pulsedb", "data/processed/pulsedb"),
    ("data/processed/ppg_dalia/windows", "data/processed/ppg_dalia/windows"),
    ("data/colab_manifest.json", "data/colab_manifest.json"),
]

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}

README = """\
Drop this whole folder into Google Drive.

Picking the project up cold? Read HANDOFF.md first — what is done, what broke
on Windows and how it was fixed, and what comes next.

  1. Drag `bp` into MyDrive (anywhere -- the notebooks find it by themselves).

  2. In Drive, right-click a notebook under bp/notebooks/
     -> Open with -> Google Colaboratory

  3. Runtime > Change runtime type > T4 GPU

  4. Run the setup cells top to bottom. The third one copies the data onto
     Colab's local disk (Drive is a network mount and would starve the GPU)
     and stops immediately if the upload is incomplete.

Notebooks
  02_baselines.ipynb            RF + GB       run this on the LAPTOP, not Colab
  03a_cnn_colab.ipynb           CNN           Colab, ~1.5 h
  03b_resnet_colab.ipynb        ResNet1D      Colab, ~2.5 h
  03c_transformer_colab.ipynb   Transformer   Colab, ~4 h
  04_personalization.ipynb      few-shot      Colab, after the above

Results and checkpoints are written back to bp/results/models/<model>/ on
Drive, so a dropped Colab session does not take the run with it.

Changing the pipeline means editing src/train.py and re-uploading src/
(0.1 MB). The data never needs re-uploading.
"""


def _link(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:                      # different device, or a filesystem
        shutil.copy2(src, dst)           # that will not hard-link


def build(copy=False):
    # Rebuild the manifest first. It records a size per file and the notebook
    # refuses to run if the copy disagrees with it, so a manifest written before
    # the last code edit fails the upload for files that are perfectly fine.
    import prepare_colab_upload
    print("refreshing the manifest\n")
    prepare_colab_upload.build()
    print()

    if OUT.exists():
        shutil.rmtree(OUT)
    n, total = 0, 0

    for rel_src, rel_dst in ITEMS:
        src = ROOT / rel_src
        if not src.exists():
            print(f"  MISSING  {rel_src}")
            continue
        if src.is_file():
            _link(src, OUT / rel_dst, copy)
            n, total = n + 1, total + src.stat().st_size
            continue
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in sorted(filenames):
                if f.startswith("."):
                    continue
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, src)
                _link(p, OUT / rel_dst / rel, copy)
                n, total = n + 1, os.path.getsize(p) + total

    (OUT / "README.txt").write_text(README)
    # results/models/<name>/ is created per model at run time; the parent has to
    # exist so the Colab notebook can symlink it back to Drive
    (OUT / "results" / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "overview").mkdir(parents=True, exist_ok=True)

    print(f"\n{OUT.relative_to(ROOT)}/  ->  {n:,} files, {total / 1e9:.2f} GB")
    print("  " + ("real copies" if copy else "hard links — no extra disk used"))
    for d in sorted(p for p in OUT.rglob("*") if p.is_dir() and
                    not any(q.is_dir() for q in p.iterdir())):
        files = list(d.glob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"    {str(d.relative_to(OUT)) + '/':44s} {len(files):>3} files "
              f"{size / 1e6:>8.1f} MB")

    free = shutil.disk_usage(ROOT).free
    print(f"\n  disk free: {free / 1e9:.1f} GB")
    print(f"\nAb `{OUT.name}` folder ko Drive pe drag kar do. "
          f"Andar README.txt me steps likhe hain.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="real copies instead of hard links")
    build(ap.parse_args().copy)
