"""
check_masks.py — visual + statistical mask verification.

Saves a grid of (image, mask overlay) pairs for random samples from each domain.
Also prints stats: % foreground pixels, empty/near-empty mask counts.

Output:
  mask_check_cracks.png
  mask_check_drywall.png
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SAMPLE_N = 12  # images per grid


def overlay(img: Image.Image, mask: Image.Image, alpha=0.45):
    """Red overlay of mask on image."""
    img_np = np.array(img.convert("RGB"), dtype=np.float32)
    mask_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    overlay_np = img_np.copy()
    overlay_np[..., 0] = np.clip(img_np[..., 0] * (1 - alpha * mask_np) + 255 * alpha * mask_np, 0, 255)
    overlay_np[..., 1] = np.clip(img_np[..., 1] * (1 - alpha * mask_np), 0, 255)
    overlay_np[..., 2] = np.clip(img_np[..., 2] * (1 - alpha * mask_np), 0, 255)
    return Image.fromarray(overlay_np.astype(np.uint8))


def check_domain(name: str, ann_path: Path, img_dir: Path, mask_dir: Path):
    with open(ann_path) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    all_ids = list(img_info.keys())
    sample_ids = random.sample(all_ids, min(SAMPLE_N, len(all_ids)))

    # Stats
    fg_ratios = []
    empty_count = 0
    for img_id in all_ids:
        mask_path = mask_dir / f"{img_id:06d}.png"
        if not mask_path.exists():
            continue
        m = np.array(Image.open(mask_path).convert("L"))
        ratio = (m > 0).mean()
        fg_ratios.append(ratio)
        if ratio < 0.001:
            empty_count += 1

    print(f"\n=== {name} ===")
    print(f"  Masks found : {len(fg_ratios)} / {len(all_ids)}")
    print(f"  Foreground  : mean={np.mean(fg_ratios)*100:.1f}%  "
          f"min={np.min(fg_ratios)*100:.2f}%  max={np.max(fg_ratios)*100:.1f}%")
    print(f"  Empty masks : {empty_count}  (fg < 0.1%)")

    # Grid plot
    cols = 4
    rows = (SAMPLE_N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for ax_idx, img_id in enumerate(sample_ids):
        meta = img_info[img_id]
        img_path = img_dir / meta["file_name"]
        mask_path = mask_dir / f"{img_id:06d}.png"

        if not img_path.exists() or not mask_path.exists():
            axes[ax_idx].axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        vis = overlay(img, mask)
        ratio = (np.array(mask) > 0).mean() * 100

        axes[ax_idx].imshow(vis)
        axes[ax_idx].set_title(f"id={img_id}  fg={ratio:.1f}%", fontsize=8)
        axes[ax_idx].axis("off")

    for ax in axes[len(sample_ids):]:
        ax.axis("off")

    plt.suptitle(f"{name} — mask overlay (red = foreground)", fontsize=11)
    plt.tight_layout()
    out = ROOT / f"mask_check_{name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Grid saved  : {out}")


if __name__ == "__main__":
    random.seed(0)
    check_domain(
        "cracks",
        ann_path=ROOT / "cracks/train/_annotations.coco.json",
        img_dir=ROOT / "cracks/train",
        mask_dir=ROOT / "masks/cracks",
    )
    check_domain(
        "drywall",
        ann_path=ROOT / "drywall/train/_annotations.coco.json",
        img_dir=ROOT / "drywall/train",
        mask_dir=ROOT / "masks/drywall",
    )
