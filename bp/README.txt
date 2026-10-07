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
