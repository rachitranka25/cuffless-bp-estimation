This folder is a staging area, not source — it is assembled on demand by
`python3 scripts/make_drive_folder.py`, which hard-links the repository's own
`src/`, `notebooks/`, `HANDOFF.md`, and `scripts/prepare_colab_upload.py` into
`bp/`, plus the processed data, in the layout Colab expects. It is not
committed with its own copy of those files to avoid two copies of the same
code drifting apart; run the script to populate `bp/src/` and
`bp/notebooks/` locally before dragging `bp/` into Drive.

  1. python3 scripts/make_drive_folder.py
  2. Drag the resulting `bp` into MyDrive (anywhere — the notebooks find it
     by themselves).
  3. In Drive, right-click a notebook under bp/notebooks/
     -> Open with -> Google Colaboratory
  4. Runtime > Change runtime type > T4 GPU
  5. Run the setup cells top to bottom. The third one copies the data onto
     Colab's local disk (Drive is a network mount and would starve the GPU)
     and stops immediately if the upload is incomplete.

Notebooks (same files as the repository root's notebooks/)
  02_baselines.ipynb            RF + GB       run this on the LAPTOP, not Colab
  03a_cnn_colab.ipynb           CNN           Colab, ~1.5 h
  03b_resnet_colab.ipynb        ResNet1D      Colab, ~2.5 h
  03c_transformer_colab.ipynb   Transformer   Colab, ~4 h (or see
                                               scripts/run_transformer_sweep.py
                                               for the non-interactive version)
  04_personalization.ipynb      few-shot      Colab, after the above

Results and checkpoints are written back to bp/results/models/<model>/ on
Drive, so a dropped Colab session does not take the run with it; copy that
folder back into the repository's own results/models/ afterward.

Changing the pipeline means editing the repository's src/train.py and
re-running make_drive_folder.py to re-link it into bp/ before re-uploading
(0.1 MB). The data never needs re-uploading.
