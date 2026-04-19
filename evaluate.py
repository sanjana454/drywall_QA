"""
evaluate.py

For a given dataset + prompt, computes per-image IoU & Dice against GT masks,
prints aggregate stats, and saves a grid of N examples:
    original image | GT mask | predicted mask | overlay diff

Usage:
    python evaluate.py --dataset cracks --prompt "segment crack"
    python evaluate.py --dataset drywall --prompt "segment taping area"
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from tqdm import tqdm


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    """pred and gt are binary uint8 arrays (0 or 255)."""
    p = pred > 127
    g = gt > 127
    intersection = (p & g).sum()
    union = (p | g).sum()
    iou = intersection / union if union > 0 else float("nan")
    dice = 2 * intersection / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else float("nan")
    return iou, dice


def diff_overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Green  = true positive
    Red    = false positive
    Blue   = false negative
    """
    p = pred > 127
    g = gt > 127
    vis = img.copy().astype(np.float32)
    alpha = 0.55
    tp = p & g
    fp = p & ~g
    fn = ~p & g
    vis[tp] = vis[tp] * (1 - alpha) + np.array([0, 220, 0]) * alpha
    vis[fp] = vis[fp] * (1 - alpha) + np.array([220, 0, 0]) * alpha
    vis[fn] = vis[fn] * (1 - alpha) + np.array([0, 60, 220]) * alpha
    return vis.astype(np.uint8)


def evaluate(dataset: str, prompt: str, root: Path, n_examples: int = 4):
    slug = slugify(prompt)
    ann_path = root / dataset / "train" / "_annotations.coco.json"
    img_dir  = root / dataset / "train"
    gt_dir   = root / "masks" / dataset
    pred_dir = root / "predictions" / dataset

    with open(ann_path) as f:
        data = json.load(f)
    img_info = {img["id"]: img for img in data["images"]}

    ious, dices = [], []
    valid_samples = []   # (img_id, iou, dice) for all found

    print(f"Computing metrics for {len(img_info)} images …")
    for img_id, meta in tqdm(img_info.items(), desc="eval"):
        gt_path   = gt_dir   / f"{img_id:06d}.png"
        pred_path = pred_dir / f"{img_id}__{slug}.png"
        if not gt_path.exists() or not pred_path.exists():
            continue

        gt   = np.array(Image.open(gt_path).convert("L"))
        pred = np.array(Image.open(pred_path).convert("L"))

        # Resize pred to GT size if needed
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            )

        iou, dice = compute_metrics(pred, gt)
        ious.append(iou)
        dices.append(dice)
        valid_samples.append((img_id, meta, iou, dice))

    # Filter out NaN (images with no GT foreground AND no prediction)
    valid_ious  = [v for v in ious  if not np.isnan(v)]
    valid_dices = [v for v in dices if not np.isnan(v)]

    print(f"\n=== Results: {dataset} | prompt='{prompt}' ===")
    print(f"  Images evaluated : {len(valid_samples)}")
    print(f"  mIoU             : {np.mean(valid_ious):.4f}  ± {np.std(valid_ious):.4f}")
    print(f"  mean Dice        : {np.mean(valid_dices):.4f}  ± {np.std(valid_dices):.4f}")
    print(f"  IoU > 0.5        : {sum(v > 0.5 for v in valid_ious)} / {len(valid_ious)}"
          f"  ({100*sum(v>0.5 for v in valid_ious)/max(len(valid_ious),1):.1f}%)")
    print(f"  IoU > 0.3        : {sum(v > 0.3 for v in valid_ious)} / {len(valid_ious)}"
          f"  ({100*sum(v>0.3 for v in valid_ious)/max(len(valid_ious),1):.1f}%)")

    # -----------------------------------------------------------------------
    # Pick n_examples: best, worst, and median samples (with GT foreground)
    # -----------------------------------------------------------------------
    fg_samples = [(iid, meta, iou, dice)
                  for iid, meta, iou, dice in valid_samples
                  if not np.isnan(iou)]
    fg_samples.sort(key=lambda x: x[2])  # sort by IoU

    n = len(fg_samples)
    if n == 0:
        print("No valid samples to visualise.")
        return

    pick_indices = sorted(set([
        0,                   # worst
        n // 4,              # low-median
        n // 2,              # median
        3 * n // 4,          # high-median
        n - 1,               # best
    ]))[:n_examples]

    picks = [fg_samples[i] for i in pick_indices]

    # -----------------------------------------------------------------------
    # Grid: rows = examples, cols = [image, GT, prediction, diff overlay]
    # -----------------------------------------------------------------------
    cols = 4
    rows = len(picks)
    fig = plt.figure(figsize=(cols * 3.8, rows * 3.8))
    gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.05)

    col_titles = ["Original", "Ground Truth", "Prediction", "TP/FP/FN overlay"]

    for col_idx, title in enumerate(col_titles):
        fig.add_subplot(gs[0, col_idx]).set_title(title, fontsize=10, fontweight="bold")
        plt.gca().axis("off")

    for row_idx, (img_id, meta, iou, dice) in enumerate(picks):
        img_path  = img_dir  / meta["file_name"]
        gt_path   = gt_dir   / f"{img_id:06d}.png"
        pred_path = pred_dir / f"{img_id}__{slug}.png"

        img_np  = np.array(Image.open(img_path).convert("RGB"))
        gt_np   = np.array(Image.open(gt_path).convert("L"))
        pred_np = np.array(Image.open(pred_path).convert("L"))

        if pred_np.shape != gt_np.shape:
            pred_np = np.array(
                Image.fromarray(pred_np).resize(
                    (gt_np.shape[1], gt_np.shape[0]), Image.NEAREST
                )
            )

        row_label = f"id={img_id}  IoU={iou:.3f}  Dice={dice:.3f}"

        panels = [
            img_np,
            np.stack([gt_np, gt_np, gt_np], axis=-1),   # grey GT
            np.stack([pred_np, pred_np, pred_np], axis=-1),
            diff_overlay(img_np, pred_np, gt_np),
        ]

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(panel)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=7, rotation=0,
                              labelpad=120, va="center")

    suptitle = (f"{dataset} — '{prompt}'\n"
                f"mIoU={np.mean(valid_ious):.4f}   mean Dice={np.mean(valid_dices):.4f}")
    fig.suptitle(suptitle, fontsize=11, y=1.01)

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0, 220/255, 0),   label="True positive"),
        Patch(facecolor=(220/255, 0, 0),   label="False positive"),
        Patch(facecolor=(0, 60/255, 220/255), label="False negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))

    out_path = root / f"eval_{dataset}_{slug}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cracks", "drywall"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-examples", type=int, default=4)
    args = parser.parse_args()

    evaluate(args.dataset, args.prompt, Path(args.root), args.n_examples)


if __name__ == "__main__":
    main()
