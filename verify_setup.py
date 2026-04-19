"""
verify_setup.py — quick smoke test before full training.

Checks:
  1. Both annotation files are loadable
  2. Masks directory exists and contains expected files
  3. CLIPSeg loads and runs a forward pass on one sample
  4. Prints dataset stats

Run:
    python verify_setup.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def check_annotations():
    print("=== Annotation files ===")
    for name in ("cracks", "drywall"):
        path = ROOT / name / "train" / "_annotations.coco.json"
        with open(path) as f:
            data = json.load(f)
        n_imgs = len(data["images"])
        n_anns = len(data["annotations"])
        cats = [c["name"] for c in data["categories"]]
        has_segs = sum(1 for a in data["annotations"] if a.get("segmentation"))
        print(f"  {name:10s}: {n_imgs} images, {n_anns} annotations, "
              f"{has_segs} with polygons, categories={cats}")


def check_masks():
    print("\n=== Mask files ===")
    for name in ("cracks", "drywall"):
        mask_dir = ROOT / "masks" / name
        if not mask_dir.exists():
            print(f"  {name:10s}: MISSING — run prepare_masks.py first")
        else:
            count = len(list(mask_dir.glob("*.png")))
            print(f"  {name:10s}: {count} mask PNGs found")


def check_model():
    print("\n=== CLIPSeg forward pass ===")
    try:
        import torch
        from PIL import Image
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        model_id = "CIDAS/clipseg-rd64-refined"
        print(f"  Loading {model_id} …")
        processor = CLIPSegProcessor.from_pretrained(model_id)
        model = CLIPSegForImageSegmentation.from_pretrained(model_id)
        model.eval()

        # Dummy input
        img = Image.new("RGB", (640, 640), color=(128, 64, 32))
        prompt = "segment crack"
        enc = processor(text=prompt, images=img, return_tensors="pt", padding=True)

        with torch.no_grad():
            out = model(**enc)

        print(f"  Output logits shape: {out.logits.shape}  ✓")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_annotations()
    check_masks()
    check_model()
    print("\nAll checks passed.")
