# MarkCleaner

Official PyTorch implementation of **MarkCleaner: High-Fidelity Watermark Attack via Imperceptible Micro-Geometric Perturbation**.

## Installation

```bash
conda create -n markcleaner python=3.10 -y
conda activate markcleaner
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r code/requirements.txt
pip install -e code/src/gaussian_cuda
```

## Pretrained Models

Download the MarkCleaner checkpoint from the link in [checkpoints/README.md](checkpoints/README.md) and place it at:

```text
checkpoints/markcleaner.pth
```

Download the official DINOv2 ViT-B/14 checkpoint and place it at:

```text
code/pretrained/dinov2_vitb14_pretrain.pth
```

## Inference

### Invisible Watermarks

```text
invisible_eval/
└── hidden_test.txt
```

`hidden_test.txt` contains the paths to watermarked images, one path per line.

```bash
python code/scripts/infer_invisible.py \
  --config_path code/configs/infer_invisible.py \
  --data_path /path/to/invisible_eval
```

### Visible Watermarks

```text
visible_eval/
├── hidden_test.txt
└── mask_test.txt
```

By default, white mask pixels are retained and black pixels are restored. Set `mask_is_removal_region=True` in `code/configs/infer_visible.py` for the opposite convention.

```bash
python code/scripts/infer_visible.py \
  --config_path code/configs/infer_visible.py \
  --data_path /path/to/visible_eval
```

Use `--checkpoint` and `--output_dir` to override paths from the config file.

## Training

Prepare `cover_train.txt` and `hidden_train.txt` as described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md), update `code/configs/train.py`, and run:

```bash
python code/scripts/train.py --config_path code/configs/train.py
```

## License

See [LICENSE](LICENSE). Use this code only for authorized research and evaluation.

## Acknowledgements

This project uses DINOv2 and GLM. Their original licenses apply.
