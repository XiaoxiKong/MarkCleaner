# Reproducibility notes

The paper evaluates watermark removal effectiveness across twelve watermarking schemes and reports TPR@1%FPR/ACC, visual quality, speed, phase analysis, and ablations. Analysis outputs are not bundled with the source release.

Exact reproduction of the full tables requires external watermark encoders/detectors and their licensed/generated datasets.  These are not bundled.  Use the official upstream implementations and their evaluation protocols, and record version, detector threshold, image set, GPU, and random seed for each run.

The included checkpoint and sample pair reproduce the released MarkCleaner inference path itself. They do not, by themselves, reproduce detector-side results for all twelve third-party watermark schemes. The released model configuration is fixed to 256 × 256, `src.model.MarkCleanerModel`, 324 Gaussians per patch, overlap rasterization, and DINOv2 ViT-B/14 conditioning. The training entry point uses the rotate/translate MFM loader, a mask-ratio range of 0.2–0.6, an 800-epoch schedule, and frequency-mask augmentation. Its application probability increases linearly from 0 to 0.6 over the first 200,000 global steps.

## Tested environment

The one-image release smoke test passed with Python 3.10.19, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, NumPy 1.26.4, mmengine 0.10.7, OpenCV 4.11.0, an NVIDIA L40 GPU, and the locally compiled `gspaint_cuda` extension. Other compatible versions may work, but this is the verified reference environment. The CUDA extension must be rebuilt for the user's PyTorch/CUDA toolchain.
