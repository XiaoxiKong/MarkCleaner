#!/usr/bin/env python3
"""Visible-watermark removal by masked input and local compositing.

The no-DT generator does not consume the mask inside its encoder. Following
the original visible-watermark experiment, the removal region is zeroed before
inference. The prediction is then pasted only into that region so pixels
outside the user mask remain unchanged.
"""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmengine import Config
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from src.dino import create_model as create_dino_model
from src.dino import get_feature as get_dino_feature
from src.model import MarkCleanerModel


def read_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    items = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = path.parent / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Listed file does not exist: {candidate}")
        items.append(candidate)
    return items


class VisibleWatermarkDataset(Dataset):
    def __init__(self, root: str, image_size: int, mask_is_removal_region: bool):
        self.root = Path(root)
        self.images = read_list(self.root / "hidden_test.txt")
        self.masks = read_list(self.root / "mask_test.txt")
        lengths = {len(self.images), len(self.masks)}
        if len(lengths) != 1 or not self.images:
            raise ValueError("The image and mask lists must have equal non-zero lengths")
        output_names = [path.stem + ".png" for path in self.images]
        if len(output_names) != len(set(output_names)):
            raise ValueError("hidden_test.txt contains duplicate filename stems that would overwrite outputs")
        self.image_size = image_size
        self.mask_is_removal_region = mask_is_removal_region

    def __len__(self):
        return len(self.images)

    def _rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return to_tensor(image)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = self._rgb(image_path)
        mask = Image.open(self.masks[index]).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask = (to_tensor(mask) >= 0.5).float()
        removal_mask = mask if self.mask_is_removal_region else 1.0 - mask
        return {"image": image, "removal_mask": removal_mask,
                "name": image_path.stem + ".png"}


def load_checkpoint_strict(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        epoch = checkpoint.get("epoch", "unknown")
        step = checkpoint.get("global_step", "unknown")
    else:
        state, epoch, step = checkpoint, "unknown", "unknown"
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return epoch, step


def expand_removal_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = 2 * pixels + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pixels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="code/configs/infer_visible.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", help="Override data_param.path")
    parser.add_argument("--checkpoint", help="Override inference_param.model_path")
    parser.add_argument("--output_dir", help="Override inference_param.sample_dir")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    inference_cfg = dict(cfg.inference_param)
    data_cfg = dict(cfg.data_param)
    model_cfg = dict(cfg.model_param)
    if args.data_path:
        data_cfg["path"] = args.data_path
    if args.checkpoint:
        inference_cfg["model_path"] = args.checkpoint
    if args.output_dir:
        inference_cfg["sample_dir"] = args.output_dir
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian rasterizer")
    image_size = int(model_cfg["image_size"][0])

    dino = None
    if model_cfg.get("use_dino_cls", False):
        dino = create_dino_model().to(device).eval()
        dino.requires_grad_(False)
    model = MarkCleanerModel(**model_cfg, is_train=False, dino_model=dino)
    epoch, step = load_checkpoint_strict(model, inference_cfg["model_path"])
    model = model.to(device).eval()
    model.requires_grad_(False)

    dataset = VisibleWatermarkDataset(
        data_cfg["path"], image_size, bool(data_cfg.get("mask_is_removal_region", False)))
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("bs", 1)), shuffle=False,
                        num_workers=int(data_cfg.get("num_workers", 4)), pin_memory=True)
    output_dir = (Path(inference_cfg["sample_dir"]) / inference_cfg["exp_name"]
                  / f"Epoch{epoch}_Step{step}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_debug = bool(inference_cfg.get("save_debug", False))
    if save_debug:
        (output_dir / "prediction").mkdir(exist_ok=True)
        (output_dir / "mask").mkdir(exist_ok=True)

    latency = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Visible-watermark inference"):
            image = batch["image"].to(device, non_blocking=True)
            removal_mask = batch["removal_mask"].to(device, non_blocking=True)
            removal_mask = expand_removal_mask(removal_mask, int(inference_cfg.get("mask_dilation", 0)))
            keep_mask = 1.0 - removal_mask
            masked_input = image * keep_mask
            dino_feature = get_dino_feature(masked_input, dino) if dino is not None else None
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = model(masked_input, keep_mask, dino_cls_token=dino_feature)
            torch.cuda.synchronize(device)
            latency.append((time.perf_counter() - start) * 1000.0 / image.shape[0])
            prediction = prediction.clamp(0.0, 1.0)
            restored = prediction * removal_mask + image * keep_mask

            restored_cpu, prediction_cpu = restored.cpu(), prediction.cpu()
            mask_cpu = removal_mask.cpu()
            for i, name in enumerate(batch["name"]):
                save_image(restored_cpu[i], output_dir / name)
                if save_debug:
                    save_image(prediction_cpu[i], output_dir / "prediction" / name)
                    save_image(mask_cpu[i], output_dir / "mask" / name)
    print(f"Images: {len(dataset)}")
    print(f"Latency: {np.mean(latency):.2f} ms/image")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
