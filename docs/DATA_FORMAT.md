# Data format

The invisible-watermark inference entry point consumes a directory with one newline-delimited path list:

```text
dataset/
└── hidden_test.txt
```

Each line in `hidden_test.txt` is a watermarked image path. Paths may be absolute or relative to the directory containing the list. Images are resized to 256 × 256 by the supplied loader.

For training, use the analogous `cover_train.txt` and `hidden_train.txt` files.  The training script then generates spatial masks and its micro-geometric supervision internally.

## Visible-watermark data format

The visible-watermark script requires `hidden_test.txt` and `mask_test.txt`. The lists must have the same non-zero length. By default, white mask pixels retain the input and black pixels identify the region to restore. Set `mask_is_removal_region=True` for the opposite convention. The script masks the model input and locally composites the prediction, while the generator itself remains blind rather than internally mask-conditioned.
