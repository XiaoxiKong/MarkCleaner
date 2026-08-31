#!/usr/bin/env python3
"""Blind invisible-watermark removal with the released MarkCleaner model."""

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


class InvisibleWatermarkDataset(Dataset):
    def __init__(self, root: str, image_size: int):
        self.root = Path(root)
        self.images = read_list(self.root / "hidden_test.txt")
        if not self.images:
            raise ValueError("hidden_test.txt must be non-empty")
        names = [path.stem + ".png" for path in self.images]
        if len(names) != len(set(names)):
            raise ValueError("hidden_test.txt contains duplicate filename stems")
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def _rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return to_tensor(image)

    def __getitem__(self, index):
        return {"image": self._rgb(self.images[index]),
                "name": self.images[index].stem + ".png"}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="code/configs/infer_invisible.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", help="Override data_param.path")
    parser.add_argument("--checkpoint", help="Override inference_param.model_path")
    parser.add_argument("--output_dir", help="Override inference_param.sample_dir")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    inference_cfg, data_cfg = dict(cfg.inference_param), dict(cfg.data_param)
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

    dino = None
    if model_cfg.get("use_dino_cls", False):
        dino = create_dino_model().to(device).eval()
        dino.requires_grad_(False)
    model = MarkCleanerModel(**model_cfg, is_train=False, dino_model=dino)
    epoch, step = load_checkpoint_strict(model, inference_cfg["model_path"])
    model = model.to(device).eval()
    model.requires_grad_(False)

    image_size = int(model_cfg["image_size"][0])
    dataset = InvisibleWatermarkDataset(data_cfg["path"], image_size)
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("bs", 1)), shuffle=False,
                        num_workers=int(data_cfg.get("num_workers", 4)), pin_memory=True)
    output_dir = (Path(inference_cfg["sample_dir"]) / inference_cfg["exp_name"]
                  / f"Epoch{epoch}_Step{step}")
    output_dir.mkdir(parents=True, exist_ok=True)

    latency = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Invisible-watermark inference"):
            image = batch["image"].to(device, non_blocking=True)
            dino_feature = get_dino_feature(image, dino) if dino is not None else None
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            restored = model(image, dino_cls_token=dino_feature).clamp(0.0, 1.0)
            torch.cuda.synchronize(device)
            latency.append((time.perf_counter() - start) * 1000.0 / image.shape[0])
            restored = restored.cpu()
            for i, name in enumerate(batch["name"]):
                save_image(restored[i], output_dir / name)
    print(f"Images: {len(dataset)}")
    print(f"Latency: {np.mean(latency):.2f} ms/image")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
