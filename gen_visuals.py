"""
gen_visuals.py

Generates 4-panel visual grids (Original | GT | Prediction | TP/FP/FN overlay)
for both domains, picking worst / low-median / median / best examples by IoU.

Outputs:
  report_visuals_cracks.png
  report_visuals_drywall.png
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
N_EXAMPLES = 4


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def overlay(img_np, pred_np, gt_np):
    p   = pred_np > 127
    g   = gt_np   > 127
    vis = img_np.copy().astype(np.float32)
    alpha = 0.55
    vis[p &  g] = vis[p &  g] * (1-alpha) + np.array([0,   210, 0  ]) * alpha  # TP green
    vis[p & ~g] = vis[p & ~g] * (1-alpha) + np.array([220, 0,   0  ]) * alpha  # FP red
    vis[~p & g] = vis[~p & g] * (1-alpha) + np.array([0,   60,  220]) * alpha  # FN blue
    return vis.astype(np.uint8)


def iou(pred_np, gt_np):
    p = pred_np > 127
    g = gt_np   > 127
    inter = (p & g).sum()
    union = (p | g).sum()
    return inter / union if union > 0 else float("nan")


def make_grid(domain, prompt, ann_path, img_dir, gt_dir, pred_dir, out_path):
    slug = slugify(prompt)

    with open(ann_path) as f:
        data = json.load(f)
    img_info = {img["id"]: img for img in data["images"]}

    # Compute IoU for every image
    scored = []
    for img_id, meta in img_info.items():
        gt_path   = gt_dir   / f"{img_id:06d}.png"
        pred_path = pred_dir / f"{img_id}__{slug}.png"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt   = np.array(Image.open(gt_path).convert("L"))
        pred = np.array(Image.open(pred_path).convert("L"))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST))
        score = iou(pred, gt)
        if not np.isnan(score):
            scored.append((score, img_id, meta))

    if not scored:
        print(f"No scored samples for {domain}")
        return

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    # Pick worst, 25th-pct, 50th-pct, 75th-pct (or best if n<4)
    indices = sorted({0, n//4, n//2, 3*n//4, n-1})[:N_EXAMPLES]
    picks   = [scored[i] for i in indices]

    col_labels = ["Original", "Ground Truth", "Prediction", "TP / FP / FN"]
    rows = len(picks)
    cols = 4

    fig = plt.figure(figsize=(cols * 3.6, rows * 3.6 + 0.8))
    gs  = gridspec.GridSpec(rows + 1, cols, figure=fig,
                            height_ratios=[0.25] + [1]*rows,
                            hspace=0.12, wspace=0.04)

    # Column headers
    for c, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, c])
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.axis("off")

    for row_idx, (score, img_id, meta) in enumerate(picks):
        img_path  = img_dir  / meta["file_name"]
        gt_path   = gt_dir   / f"{img_id:06d}.png"
        pred_path = pred_dir / f"{img_id}__{slug}.png"

        img_np  = np.array(Image.open(img_path).convert("RGB"))
        gt_np   = np.array(Image.open(gt_path).convert("L"))
        pred_np = np.array(Image.open(pred_path).convert("L"))
        if pred_np.shape != gt_np.shape:
            pred_np = np.array(Image.fromarray(pred_np).resize(
                (gt_np.shape[1], gt_np.shape[0]), Image.NEAREST))

        panels = [
            img_np,
            np.stack([gt_np]*3, axis=-1),
            np.stack([pred_np]*3, axis=-1),
            overlay(img_np, pred_np, gt_np),
        ]
        dice_val = 2*(pred_np>127).sum()*(gt_np>127).sum() / \
                   max(1, (pred_np>127).sum() + (gt_np>127).sum())

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(f"id={img_id}  IoU={score:.3f}  Dice={2*(pred_np>127).sum()*(gt_np>127).sum()/max(1,(pred_np>127).sum()+(gt_np>127).sum())/max(1,max((pred_np>127).sum(),(gt_np>127).sum()))*max(1,max((pred_np>127).sum(),(gt_np>127).sum())):.0f}",
                             fontsize=7, loc="left", pad=3)
                # simpler label
                ax.set_title(f"id={img_id}   IoU={score:.3f}", fontsize=7, loc="left", pad=3)

    legend = [
        mpatches.Patch(color=(0,   210/255, 0),    label="True positive"),
        mpatches.Patch(color=(220/255, 0,   0),    label="False positive"),
        mpatches.Patch(color=(0,   60/255,  220/255), label="False negative"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"{domain.capitalize()}  —  prompt: \"{prompt}\"",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_grid(
        domain   = "cracks",
        prompt   = "segment crack",
        ann_path = ROOT / "cracks/train/_annotations.coco.json",
        img_dir  = ROOT / "cracks/train",
        gt_dir   = ROOT / "masks/cracks",
        pred_dir = ROOT / "predictions/cracks",
        out_path = ROOT / "report_visuals_cracks.png",
    )
    make_grid(
        domain   = "drywall",
        prompt   = "segment taping area",
        ann_path = ROOT / "drywall/train/_annotations.coco.json",
        img_dir  = ROOT / "drywall/train",
        gt_dir   = ROOT / "masks/drywall",
        pred_dir = ROOT / "predictions/drywall",
        out_path = ROOT / "report_visuals_drywall.png",
    )
