# drywall_QA
Prompted Segmentation for Drywall QA

Text-conditioned segmentation using fine-tuned CLIPSeg.
Given an image + natural-language prompt, produces a binary mask PNG.

## Prompts
| Domain | Example prompts |
|--------|----------------|
| Cracks | "segment crack", "segment wall crack", "segment surface crack" |
| Drywall taping area | "segment taping area", "segment joint tape", "segment drywall seam" |

## Output format
Single-channel PNG, same spatial size as input, pixel values `{0, 255}`.
Filenames: `{image_id}__{prompt_slug}.png` (e.g. `42__segment_crack.png`).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare binary mask PNGs from COCO annotations
#    Cracks: uses polygon annotations directly
#    Drywall: falls back to bbox-fill (pass --sam-checkpoint for SAM-based masks)
python prepare_masks.py
# With SAM (better drywall masks):
# python prepare_masks.py --sam-checkpoint /path/to/sam_vit_h.pth

# 3. Verify everything looks OK
python verify_setup.py

# 4. Fine-tune CLIPSeg
python train.py --epochs 10 --batch-size 8 --lr 5e-5
# Freeze CLIP encoder, train only decoder (faster, less GPU):
# python train.py --freeze-encoder --epochs 15 --batch-size 16

# 5. Run inference
python predict.py --dataset cracks --prompt "segment crack"
python predict.py --dataset drywall --prompts "segment taping area" "segment joint tape"
# Single image:
# python predict.py --image path/to/img.jpg --prompt "segment crack"
```

## Files
| File | Purpose |
|------|---------|
| `prepare_masks.py` | COCO annotations → binary mask PNGs |
| `dataset.py` | PyTorch Dataset + DataLoader utilities |
| `train.py` | CLIPSeg fine-tuning (BCE + Dice loss) |
| `predict.py` | Inference → `{id}__{prompt}.png` masks |
| `verify_setup.py` | Sanity check before training |

## Datasets
- `cracks/train/` — 5,369 images with polygon segmentation (crack detection)
- `drywall/train/` — 1,022 images with bounding boxes (drywall join/tape detection)

## Notes on drywall dataset
The drywall Roboflow export has **no polygon segmentation**, only bounding boxes.
`prepare_masks.py` offers two strategies:
- **bbox-fill** (default, no extra deps): fills the bounding-box rectangle as foreground.
- **SAM** (`--sam-checkpoint`): uses Segment Anything with bbox prompts for tighter masks.
  Download `sam_vit_h_4b8939.pth` from the SAM repo.
