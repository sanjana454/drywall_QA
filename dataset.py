"""
dataset.py

PyTorch Dataset for CLIPSeg fine-tuning.

Each sample:
  image    — PIL Image (RGB, 640×640 → resized to CLIPSeg input size)
  prompt   — str, e.g. "segment crack" or "segment taping area"
  mask     — torch.FloatTensor (1, target_size, target_size), values in [0, 1]

The dataset combines both domain splits, randomly sampling prompts from a
configurable prompt pool per domain so the model learns synonym robustness.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

# CLIPSeg decoder output is 352×352
CLIPSEG_SIZE = 352

# Prompt pools per domain
CRACK_PROMPTS = [
    "segment crack",
    "segment wall crack",
    "segment surface crack",
    "crack",
]

DRYWALL_PROMPTS = [
    "segment taping area",
    "segment joint tape",
    "segment drywall seam",
    "segment drywall joint",
    "taping area",
]


class SegmentationSplit(Dataset):
    """Single-domain dataset: pairs images with a sampled text prompt + mask."""

    def __init__(
        self,
        ann_path: str | Path,
        img_dir: str | Path,
        mask_dir: str | Path,
        prompts: list[str],
        clipseg_processor,
        augment: bool = True,
        seed: Optional[int] = None,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.prompts = prompts
        self.processor = clipseg_processor
        self.augment = augment
        self.rng = random.Random(seed)

        with open(ann_path) as f:
            data = json.load(f)

        # Only keep images whose mask file exists
        self.samples = []
        for img_meta in data["images"]:
            img_id = img_meta["id"]
            mask_path = self.mask_dir / f"{img_id:06d}.png"
            img_path = self.img_dir / img_meta["file_name"]
            if mask_path.exists() and img_path.exists():
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        prompt = self.rng.choice(self.prompts)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        if self.augment:
            # Horizontal flip
            if self.rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
            if self.rng.random() < 0.3:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Random rotation ±20°
            if self.rng.random() < 0.5:
                angle = self.rng.uniform(-20, 20)
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)

            # Random scale + crop (zoom in 80–100% of original)
            if self.rng.random() < 0.5:
                w, h = image.size
                scale = self.rng.uniform(0.8, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                left = self.rng.randint(0, w - new_w)
                top  = self.rng.randint(0, h - new_h)
                image = image.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)
                mask  = mask.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.NEAREST)

            # Color jitter (image only, not mask)
            if self.rng.random() < 0.6:
                image = TF.adjust_brightness(image, 1 + self.rng.uniform(-0.3, 0.3))
            if self.rng.random() < 0.6:
                image = TF.adjust_contrast(image, 1 + self.rng.uniform(-0.3, 0.3))
            if self.rng.random() < 0.4:
                image = TF.adjust_saturation(image, 1 + self.rng.uniform(-0.3, 0.3))
            if self.rng.random() < 0.3:
                image = TF.adjust_hue(image, self.rng.uniform(-0.05, 0.05))

        # CLIPSeg processor handles image normalization & resizing internally
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Resize mask to CLIPSeg output size (352×352)
        mask_resized = mask.resize((CLIPSEG_SIZE, CLIPSEG_SIZE), Image.NEAREST)
        mask_tensor = torch.from_numpy(
            np.array(mask_resized, dtype=np.float32) / 255.0
        ).unsqueeze(0)  # (1, 352, 352)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),   # (3, H, W)
            "input_ids": encoding["input_ids"].squeeze(0),          # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "mask": mask_tensor,                                     # (1, 352, 352)
            "prompt": prompt,
        }


def build_datasets(root: str | Path, clipseg_processor, augment: bool = True):
    """Return combined train dataset from both domains."""
    root = Path(root)

    cracks_ds = SegmentationSplit(
        ann_path=root / "cracks/train/_annotations.coco.json",
        img_dir=root / "cracks/train",
        mask_dir=root / "masks/cracks",
        prompts=CRACK_PROMPTS,
        clipseg_processor=clipseg_processor,
        augment=augment,
    )

    drywall_ds = SegmentationSplit(
        ann_path=root / "drywall/train/_annotations.coco.json",
        img_dir=root / "drywall/train",
        mask_dir=root / "masks/drywall",
        prompts=DRYWALL_PROMPTS,
        clipseg_processor=clipseg_processor,
        augment=augment,
    )

    print(f"Cracks  split: {len(cracks_ds)} samples")
    print(f"Drywall split: {len(drywall_ds)} samples")

    return ConcatDataset([cracks_ds, drywall_ds])


def collate_fn(batch):
    """Pad input_ids to the same length within a batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])

    input_ids_padded = []
    attention_masks_padded = []
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad = max_len - seq_len
        input_ids_padded.append(
            torch.nn.functional.pad(item["input_ids"], (0, pad))
        )
        attention_masks_padded.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad))
        )

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_masks_padded),
        "mask": masks,
    }
