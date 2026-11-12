# Licence for data-derived files

The code in this repository (`src/`, `scripts/`, `notebooks/`) is released under the
MIT License in [`LICENSE`](LICENSE).

The files listed below are not code. They were produced from PulseDB, whose VitalDB-derived
subset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International licence (CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/).
Under its share-alike terms they are released here under the same licence, CC BY-NC-SA 4.0,
and may not be used for commercial purposes:

- trained model weights: `results/models/*/checkpoints/` and
  `experiment_1_20epoch/results/models/*/checkpoints/`
- per-clip predictions, which include the reference SBP/DBP labels and PulseDB subject IDs:
  `*_predictions.npz` under `results/` and `experiment_1_20epoch/results/`
- metrics, personalization results and logs under `results/` and `experiment_1_20epoch/results/`
- figures under `results/` that show PulseDB signals or results computed from them
- the split manifests `data/splits/manifest.json` and `data/colab_manifest.json`

Files computed from PPG-DaLiA (CC BY 4.0) by models trained on PulseDB fall under the same
CC BY-NC-SA 4.0 terms, because the models that produced them do.

No raw or processed signals from PulseDB, VitalDB or PPG-DaLiA are included. Download
instructions are in the README (section "The data").

## Attribution

- PulseDB: W. Wang, P. Mohseni, K. L. Kilgore, L. Najafizadeh, PulseDB: A large, cleaned
  dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation
  methods, Frontiers in Digital Health 4 (2022) 1090854. https://doi.org/10.3389/fdgth.2022.1090854
- VitalDB: H.-C. Lee et al., VitalDB, a high-fidelity multi-parameter vital signs database in
  surgical patients, Scientific Data 9 (2022) 279. https://doi.org/10.1038/s41597-022-01411-5
- PPG-DaLiA: A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven, Deep PPG: Large-scale
  heart rate estimation with convolutional neural networks, Sensors 19 (2019) 3079.
  https://doi.org/10.3390/s19143079
