# Experiment 1 — 20-epoch results archive

Snapshot of the original 20-epoch run, before the professor's feedback triggered the
50-epoch / 3-seed retrain (Experiment 2).

## What's genuine (copied straight from the original run's output files)

- `results/models/rf/` — Random Forest. No epoch concept (tree ensemble); this is the
  seed-0 run from before seed-variance was added. Predictions, metrics, and figures.
- `results/models/gb/` — Gradient Boosting. Same as above.
- `results/models/cnn/` — recovered from the user's own backup (`models_result_20epoch.zip`,
  Downloads). All 4 protocols, dalia domain-shift (balanced + nobalance), predictions,
  checkpoints. Confirmed `epochs: 20` in every run's config, and the metrics match exactly
  what was documented at the time (SBP/DBP MAE 14.86/9.01 balanced, 12.07/7.98 nobalance).
- `results/models/resnet/` — same recovery, same verification (13.58/8.97 balanced,
  13.15/8.67 nobalance).
- `results/models/transformer/` — the original lightweight custom Transformer (20 epochs,
  our own recipe per spec §I, not DMT's). Genuinely intact, nothing was ever deleted for
  this model. `results/models/transformer/from_recovered_zip/` holds a second `calfree`-only
  run for this model found in the same recovered zip — its numbers differ slightly from the
  intact copy (16.99/9.49 vs 16.29/9.77 SBP/DBP MAE), kept for reference rather than merged in.

## Figures

`results/overview/` and `results/models/<name>/figures/` were regenerated from the files
above by running `scripts/plot_results.py` with `BP_ROOT` pointed at this folder — the
exact same figure code the main project uses, just reading this archive instead of the
live `results/`.

## Notebooks

`notebooks/` — regenerated, not the original files (which were overwritten in place with
no backup). See `notebooks/NOTE.md` — the code is reconstructed to match exactly what
produced the results above (20 epochs, no seed-variance cells), via
`scripts/build_experiment1_notebooks.py`.
