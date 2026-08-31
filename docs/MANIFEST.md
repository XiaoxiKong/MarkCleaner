# Paper-to-release manifest

This manifest was prepared from `scripts/PAPER.tex` so that each released item has a clear purpose.

| Paper component | Release material | Status |
| --- | --- | --- |
| MarkCleaner encoder, 2D Gaussian decoder, mask-guided representation, and losses | `code/src/` | Included |
| Training procedure and micro-geometric supervision | `code/scripts/train.py` and `code/configs/train.py` | Included |
| Blind invisible-watermark inference and timing/PSNR/SSIM reporting | `code/scripts/infer_invisible.py` and `code/configs/infer_invisible.py` | Included |
| Mask-localized visible-watermark inference | `code/scripts/infer_visible.py` and `code/configs/infer_visible.py` | Included; masked-input restoration plus local output compositing, not internal mask conditioning |
| Paper tables using twelve external watermark schemes | External watermark encoders, decoders, and authorized benchmark images | Not redistributed |
| Final paper figures (`figs/*.pdf`) and bibliography source | Not present in the workspace | Not included |
| Paper-compatible trained checkpoint | Google Drive; placement documented in `checkpoints/README.md` | Hosted separately; selected `epoch481_step90000.pth` |

The copied source tree intentionally excludes compiled artifacts, Python caches, private paths, experiment logs, raw datasets, analysis outputs, and unrelated model outputs.
