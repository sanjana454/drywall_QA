"""
prepare_masks.py

Converts COCO annotations to binary mask PNGs:
  - Cracks: polygon segmentation → filled binary mask
  - Drywall: no polygons → uses SAM with bbox prompt for pseudo-masks
             (falls back to filled bbox rectangle if SAM unavailable)

Outputs:
  masks/cracks/  {image_id:06d}.png    — single-channel, values {0, 255}
  masks/drywall/ {image_id:06d}.png    — single-channel, values {0, 255}
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Polygon-based mask (Cracks dataset)
# ---------------------------------------------------------------------------

def coco_polygons_to_mask(segmentation, height, width):
    """Rasterize COCO polygon segmentation into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def build_cracks_masks(ann_path: Path, img_dir: Path, out_dir: Path):
    """Build binary masks from polygon annotations (cracks dataset)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path) as f:
        data = json.load(f)

    # Map image_id → image info
    img_info = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    skipped = 0
    for img_id, img_meta in tqdm(img_info.items(), desc="Cracks masks"):
        h, w = img_meta["height"], img_meta["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in ann_by_img.get(img_id, []):
            segs = ann.get("segmentation", [])
            if segs:
                mask = np.maximum(mask, coco_polygons_to_mask(segs, h, w))
            else:
                # Fall back to bbox rectangle if polygon missing
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                mask[y:y + bh, x:x + bw] = 255

        out_path = out_dir / f"{img_id:06d}.png"
        Image.fromarray(mask).save(out_path)

    print(f"Cracks: saved {len(img_info)} masks to {out_dir}  (skipped {skipped})")


# ---------------------------------------------------------------------------
# SAM-based mask (Drywall dataset — bboxes only, no polygons)
# ---------------------------------------------------------------------------

def build_drywall_masks_sam(ann_path: Path, img_dir: Path, out_dir: Path,
                             sam_checkpoint: str, sam_model_type: str = "vit_h"):
    """Generate pseudo-masks with SAM prompted by COCO bboxes."""
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print("segment-anything not installed; falling back to bbox-fill masks.")
        build_drywall_masks_bbox(ann_path, img_dir, out_dir)
        return

    if not os.path.exists(sam_checkpoint):
        print(f"SAM checkpoint not found at {sam_checkpoint}; "
              "falling back to bbox-fill masks.")
        build_drywall_masks_bbox(ann_path, img_dir, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"Loading SAM ({sam_model_type}) on {device} …")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    with open(ann_path) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_meta in tqdm(img_info.items(), desc="Drywall masks (SAM)"):
        img_path = img_dir / img_meta["file_name"]
        if not img_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        predictor.set_image(image_rgb)

        combined = np.zeros((h, w), dtype=np.uint8)
        anns = ann_by_img.get(img_id, [])

        if anns:
            # Build SAM input boxes: COCO [x,y,w,h] → xyxy
            boxes = []
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                boxes.append([x, y, x + bw, y + bh])
            boxes_np = np.array(boxes, dtype=np.float32)

            import torch
            boxes_t = __import__("torch").tensor(boxes_np, device=device)
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_t, image_rgb.shape[:2]
            )
            masks_sam, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            # masks_sam: (N, 1, H, W) bool tensor
            for m in masks_sam:
                combined = np.maximum(combined,
                                      (m[0].cpu().numpy() * 255).astype(np.uint8))
        else:
            pass  # no annotations → all-zero mask

        out_path = out_dir / f"{img_id:06d}.png"
        Image.fromarray(combined).save(out_path)

    print(f"Drywall: saved {len(img_info)} SAM masks to {out_dir}")


def build_drywall_masks_bbox(ann_path: Path, img_dir: Path, out_dir: Path):
    """Fallback: fill bbox rectangles as pseudo-masks (no SAM required)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_meta in tqdm(img_info.items(), desc="Drywall masks (bbox)"):
        h, w = img_meta["height"], img_meta["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in ann_by_img.get(img_id, []):
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            mask[y:y + bh, x:x + bw] = 255
        out_path = out_dir / f"{img_id:06d}.png"
        Image.fromarray(mask).save(out_path)

    print(f"Drywall: saved {len(img_info)} bbox masks to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare binary mask PNGs from COCO annotations")
    parser.add_argument("--sam-checkpoint", default="",
                        help="Path to SAM checkpoint (.pth). If empty or missing, "
                             "drywall falls back to bbox-fill masks.")
    parser.add_argument("--sam-model-type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type (must match checkpoint)")
    parser.add_argument("--skip-cracks", action="store_true")
    parser.add_argument("--skip-drywall", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent

    if not args.skip_cracks:
        build_cracks_masks(
            ann_path=root / "cracks/train/_annotations.coco.json",
            img_dir=root / "cracks/train",
            out_dir=root / "masks/cracks",
        )

    if not args.skip_drywall:
        if args.sam_checkpoint:
            build_drywall_masks_sam(
                ann_path=root / "drywall/train/_annotations.coco.json",
                img_dir=root / "drywall/train",
                out_dir=root / "masks/drywall",
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
            )
        else:
            build_drywall_masks_bbox(
                ann_path=root / "drywall/train/_annotations.coco.json",
                img_dir=root / "drywall/train",
                out_dir=root / "masks/drywall",
            )


if __name__ == "__main__":
    main()
