# Checkpoint placement

The MarkCleaner generator checkpoint is hosted separately on Google Drive and is not committed to GitHub because it is approximately 1.26 GB.

Temporary Google Drive placeholder: [download checkpoint](https://drive.google.com/drive/folders/REPLACE_WITH_PUBLIC_FOLDER_ID).

The checkpoint is not available at this placeholder yet. Replace `REPLACE_WITH_PUBLIC_FOLDER_ID` with the public Google Drive folder ID after upload.

After downloading, place it at:

```text
checkpoints/markcleaner.pth
```

`markcleaner.pth` is `epoch481_step90000.pth` from the `gsinpaint_exp_ori_inpainting_merge_noise_mfm_mask_324_with_imgloss_shift3_20_nodt_mask0.2-0.6_rotant5_ansys` experiment. Only the public filename has changed.

SHA-256:

```text
6f9f3a3e448ec5cfc31535761728c6c9f2fd3d674b23d5d4e0f4ffe24320900f  markcleaner.pth
```

The inference configuration expects a checkpoint produced by `code/scripts/train.py`, with at least a `model` key (or a directly loadable model state dict) compatible with `src.model.MarkCleanerModel`.

Other workspace checkpoints are intentionally excluded.  In particular, `checkpoints-512-48bits-markcleaner/checkpoints/state_dict_0.pth` contains `sec_encoder` and `sec_decoder` and is not compatible with this release entry point.
