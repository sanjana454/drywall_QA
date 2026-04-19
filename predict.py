"""
predict.py — Advanced inference with prompt ensemble, adaptive thresholding,
             and optional SAM refinement.

Improvements over baseline:
  3. Prompt ensemble  — average sigmoid across N prompt variants before threshold
  5. Adaptive threshold — Otsu's method per image instead of fixed 0.5
  4. SAM refinement   — use CLIPSeg mask as SAM prompt for sharper boundaries
                        (requires --sam-checkpoint)

Output:
  {out_dir}/{image_id}__{prompt_slug}.png
  Single-channel PNG, same spatial size as input, values {0, 255}.

Usage:
  # Ensemble + adaptive threshold (recommended)
  python predict.py --dataset cracks --ensemble --adaptive-threshold

  # With SAM refinement
  python predict.py --dataset cracks --ensemble --adaptive-threshold \
      --sam-checkpoint /path/to/sam_vit_h.pth

  # Single image
  python predict.py --image img.jpg --prompt "segment crack" --ensemble
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# Default prompt pools (used when --ensemble flag is set)
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
DOMAIN_PROMPTS = {"cracks": CRACK_PROMPTS, "drywall": DRYWALL_PROMPTS}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def otsu_threshold(prob: np.ndarray,
                   floor: float = 0.25,
                   ceiling: float = 0.55) -> float:
    """
    Adaptive per-image threshold via Otsu's method, clamped to [floor, ceiling].

    For sparse foreground (thin cracks = ~4% of pixels) the histogram is
    heavily skewed toward 0, making Otsu pick a threshold that is either
    too high (erases cracks) or meaningless.  The clamp keeps it in a
    sensible range regardless of histogram shape.
    """
    if prob.max() - prob.min() < 0.05:
        return floor   # near-blank image → use floor so faint cracks survive

    hist, bin_edges = np.histogram(prob.ravel(), bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return floor

    best_var, best_t = -1.0, floor
    cumsum, cummu = 0.0, 0.0
    total_mu = (hist * bin_centers).sum()
    for h, c in zip(hist, bin_centers):
        cumsum += h
        cummu  += h * c
        w0 = cumsum / total
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        mu0 = cummu / cumsum
        mu1 = (total_mu - cummu) / (total - cumsum + 1e-9)
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, c

    return float(np.clip(best_t, floor, ceiling))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, checkpoint: str | None, device: torch.device):
    processor_path = Path(checkpoint).parent / "processor" if checkpoint else None
    if processor_path and processor_path.exists():
        processor = CLIPSegProcessor.from_pretrained(str(processor_path))
    else:
        processor = CLIPSegProcessor.from_pretrained(model_id)

    model = CLIPSegForImageSegmentation.from_pretrained(model_id)

    if checkpoint and Path(checkpoint).exists():
        print(f"Loading weights from {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        print(f"  Epoch {state.get('epoch','?')}, val_loss={state.get('val_loss','?')}")
    else:
        print("No checkpoint — using pretrained CLIPSeg (zero-shot).")

    model.eval().to(device)
    return processor, model


# ---------------------------------------------------------------------------
# SAM refinement (optional)
# ---------------------------------------------------------------------------

def load_sam(sam_checkpoint: str, model_type: str, device):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print("segment-anything not installed; skipping SAM refinement.")
        return None
    if not Path(sam_checkpoint).exists():
        print(f"SAM checkpoint not found at {sam_checkpoint}; skipping refinement.")
        return None
    print(f"Loading SAM ({model_type}) …")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def sam_refine(image_rgb: np.ndarray, clipseg_mask: np.ndarray,
               predictor) -> np.ndarray:
    """
    Use CLIPSeg binary mask to derive SAM point prompts, then refine.
    Positive points: centroids of connected components in CLIPSeg mask.
    Negative points: high-confidence background pixels.
    """
    import cv2

    predictor.set_image(image_rgb)
    fg = (clipseg_mask > 127).astype(np.uint8)

    if fg.sum() == 0:
        return clipseg_mask   # nothing to refine

    # Positive points: centroids of each connected component (up to 10)
    num_labels, labels = cv2.connectedComponents(fg)
    pos_pts = []
    for lab in range(1, min(num_labels, 11)):
        ys, xs = np.where(labels == lab)
        if len(xs) > 0:
            pos_pts.append([int(xs.mean()), int(ys.mean())])

    # Negative points: sample from confident background (prob < 0.1 region)
    bg = (clipseg_mask == 0)
    bg_ys, bg_xs = np.where(bg)
    neg_pts = []
    if len(bg_ys) > 0:
        idx = np.random.choice(len(bg_ys), size=min(3, len(bg_ys)), replace=False)
        neg_pts = [[int(bg_xs[i]), int(bg_ys[i])] for i in idx]

    all_pts    = pos_pts + neg_pts
    all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

    if not all_pts:
        return clipseg_mask

    import torch as _torch
    pts_np  = np.array(all_pts,    dtype=np.float32)
    labs_np = np.array(all_labels, dtype=np.int32)

    masks_sam, scores, _ = predictor.predict(
        point_coords  = pts_np,
        point_labels  = labs_np,
        multimask_output = True,
    )
    best_idx  = scores.argmax()
    refined   = (masks_sam[best_idx] * 255).astype(np.uint8)

    # Resize to original size if needed
    if refined.shape != clipseg_mask.shape:
        refined = np.array(
            Image.fromarray(refined).resize(
                (clipseg_mask.shape[1], clipseg_mask.shape[0]), Image.NEAREST
            )
        )
    return refined


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_prob_map(image: Image.Image, prompt: str,
                 processor, model, device) -> np.ndarray:
    """Return sigmoid probability map (H, W) in [0,1] at original image size."""
    orig_w, orig_h = image.size
    enc = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    out = model(
        pixel_values   = enc["pixel_values"].to(device),
        input_ids      = enc["input_ids"].to(device),
        attention_mask = enc["attention_mask"].to(device),
    )
    logits = out.logits   # (1, H_dec, W_dec)  or (H_dec, W_dec)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.dim() == 3:
        logits = logits.unsqueeze(1)

    up = F.interpolate(logits.float(), size=(orig_h, orig_w),
                       mode="bilinear", align_corners=False)
    return torch.sigmoid(up).squeeze().cpu().numpy()


def predict_image(image: Image.Image, prompts: list[str],
                  processor, model, device,
                  use_ensemble: bool,
                  adaptive_threshold: bool,
                  fixed_threshold: float,
                  sam_predictor=None) -> np.ndarray:
    """
    Full inference pipeline for one image:
      1. Prompt ensemble (or single prompt)
      2. Adaptive threshold (Otsu) or fixed
      3. Optional SAM refinement
    Returns binary mask (H, W) uint8 {0,255}.
    """
    if use_ensemble:
        # Average probability maps across all prompts
        prob_maps = [get_prob_map(image, p, processor, model, device)
                     for p in prompts]
        prob = np.mean(prob_maps, axis=0)
    else:
        prob = get_prob_map(image, prompts[0], processor, model, device)

    # Threshold
    thresh = otsu_threshold(prob) if adaptive_threshold else fixed_threshold
    mask = ((prob >= thresh) * 255).astype(np.uint8)

    # SAM refinement
    if sam_predictor is not None:
        img_rgb = np.array(image.convert("RGB"))
        mask = sam_refine(img_rgb, mask, sam_predictor)

    return mask


# ---------------------------------------------------------------------------
# Dataset-level prediction
# ---------------------------------------------------------------------------

def predict_dataset(ann_path: Path, img_dir: Path, domain: str,
                    prompts: list[str], out_dir: Path,
                    processor, model, device,
                    use_ensemble: bool, adaptive_threshold: bool,
                    fixed_threshold: float, sam_predictor):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        data = json.load(f)

    # Output filename uses first prompt as slug when ensembling
    slug = slugify(prompts[0])

    for img_meta in tqdm(data["images"], desc=f"Predicting ({domain})"):
        img_path = img_dir / img_meta["file_name"]
        if not img_path.exists():
            continue
        image  = Image.open(img_path).convert("RGB")
        img_id = img_meta["id"]

        mask = predict_image(image, prompts, processor, model, device,
                             use_ensemble, adaptive_threshold,
                             fixed_threshold, sam_predictor)

        out_path = out_dir / f"{img_id}__{slug}.png"
        Image.fromarray(mask).save(out_path)

    print(f"Saved masks to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument("--dataset", choices=["cracks", "drywall"])
    parser.add_argument("--image",   help="Single image path")
    # Prompts
    parser.add_argument("--prompt",  help="Single text prompt")
    parser.add_argument("--prompts", nargs="+")
    # Model
    parser.add_argument("--model-id",    default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    # Inference improvements
    parser.add_argument("--ensemble",            action="store_true",
                        help="Average predictions across all domain prompts")
    parser.add_argument("--adaptive-threshold",  action="store_true",
                        help="Use per-image Otsu threshold instead of fixed 0.5")
    parser.add_argument("--threshold",           type=float, default=0.5,
                        help="Fixed threshold (ignored when --adaptive-threshold)")
    # SAM refinement
    parser.add_argument("--sam-checkpoint",  default="",
                        help="Path to SAM .pth for mask refinement")
    parser.add_argument("--sam-model-type",  default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"])
    # Output
    parser.add_argument("--out-dir", default="predictions")
    parser.add_argument("--root",    default=".")
    args = parser.parse_args()

    # Resolve prompts
    if args.ensemble and args.dataset:
        prompts = DOMAIN_PROMPTS[args.dataset]
        print(f"Ensemble mode: {len(prompts)} prompts → {prompts}")
    elif args.prompts:
        prompts = args.prompts
    elif args.prompt:
        prompts = [args.prompt]
    else:
        parser.error("Provide --prompt, --prompts, or --ensemble with --dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint   = args.checkpoint if Path(args.checkpoint).exists() else None
    processor, model = load_model(args.model_id, checkpoint, device)

    sam_predictor = None
    if args.sam_checkpoint:
        sam_predictor = load_sam(args.sam_checkpoint, args.sam_model_type, device)

    out_dir = Path(args.out_dir)

    if args.image:
        image    = Image.open(args.image).convert("RGB")
        img_stem = Path(args.image).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = predict_image(image, prompts, processor, model, device,
                             args.ensemble, args.adaptive_threshold,
                             args.threshold, sam_predictor)
        slug = slugify(prompts[0])
        out_path = out_dir / f"{img_stem}__{slug}.png"
        Image.fromarray(mask).save(out_path)
        print(f"Saved: {out_path}")

    elif args.dataset:
        root = Path(args.root)
        predict_dataset(
            ann_path   = root / args.dataset / "train" / "_annotations.coco.json",
            img_dir    = root / args.dataset / "train",
            domain     = args.dataset,
            prompts    = prompts,
            out_dir    = out_dir / args.dataset,
            processor  = processor,
            model      = model,
            device     = device,
            use_ensemble        = args.ensemble,
            adaptive_threshold  = args.adaptive_threshold,
            fixed_threshold     = args.threshold,
            sam_predictor       = sam_predictor,
        )
    else:
        parser.error("Provide --dataset or --image")


if __name__ == "__main__":
    main()
