These five notebooks are a **regeneration**, not the original files.

The actual 20-epoch notebook files were overwritten in place when
`scripts/build_notebooks.py` was edited for the 50-epoch retrain (that script
regenerates notebooks in place and keeps no version history), with no backup.

What's here instead: `scripts/build_experiment1_notebooks.py` reconstructs the
same cells — `02_baselines.ipynb` and `04_personalization.ipynb` are byte-for-byte
what the current generator would still produce (neither ever depended on epoch
count), and `03a/03b/03c_*_colab.ipynb` are rebuilt with `epochs=20` and without
the seed-variance cells (which didn't exist yet in Experiment 1). The underlying
pipeline code (`src/train.py`) never changed between Experiment 1 and 2, so
running these reproduces the same training that produced the results in
`../results/`.

These carry no execution-output cells (they were never actually run through this
generator) — only the source code, which is what matters for reproducibility.
