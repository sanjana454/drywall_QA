"""
train.py — Advanced CLIPSeg fine-tuning for drywall QA segmentation.

Improvements over baseline:
  1. Boundary-aware supervision   — BCE weighted 5× near mask edges
  2. Multi-scale supervision      — loss at 3 decoder scales (1.0 / 0.5 / 0.25)
  3. Thin-structure attention bias — morphological weight uplifting thin regions
  4. Hard negative mining (OHEM)  — back-prop through hardest 70% of pixels
  5. Graduated layer-wise LR      — later CLIP layers get higher LR than earlier
  6. Freeze less of CLIP          — all vision encoder layers trainable, graduated LR

Usage:
    python train.py --epochs 25 --batch-size 8 --lr 5e-5 --unfreeze-layers 8
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from dataset import build_datasets, collate_fn


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Soft Dice loss. Inputs: (B, 1, H, W)."""
    pred = torch.sigmoid(pred_logits)
    pred   = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    inter  = (pred * target).sum(dim=1)
    return (1 - (2 * inter + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)).mean()


def boundary_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                  dilation: int = 7, boundary_weight: float = 4.0) -> torch.Tensor:
    """
    BCE weighted higher near mask boundaries.
    Boundary region = dilated_mask − eroded_mask.
    Pixels inside boundary get (1 + boundary_weight) × normal loss weight.
    """
    k = dilation
    pad = k // 2
    kernel = torch.ones(1, 1, k, k, device=target.device, dtype=target.dtype) / (k * k)

    dilated = F.conv2d(target,       kernel, padding=pad).clamp(0, 1)
    eroded  = F.conv2d(1.0 - target, kernel, padding=pad).clamp(0, 1)
    boundary_mask = (dilated - (1.0 - eroded)).clamp(0, 1)

    weight = 1.0 + boundary_weight * boundary_mask
    return F.binary_cross_entropy_with_logits(pred_logits, target, weight=weight)


def thin_structure_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    Attention bias for thin structures (cracks, seams).
    Uses morphological erosion: pixels that vanish under 3×3 erosion are 'thin'.
    Applies an extra Dice term weighted toward those thin foreground pixels.
    """
    # 3×3 erosion: min-pool with negation trick
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    thin   = (target - eroded).clamp(0, 1)        # thin foreground pixels

    pred  = torch.sigmoid(pred_logits)
    # Focus Dice on thin regions (+0.1 prevents zero-division on all-background)
    p_thin = pred   * (thin + 0.1)
    t_thin = target * (thin + 0.1)
    inter  = (p_thin * t_thin).sum(dim=[1, 2, 3])
    denom  = p_thin.sum(dim=[1, 2, 3]) + t_thin.sum(dim=[1, 2, 3])
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def ohem_bce(pred_logits: torch.Tensor, target: torch.Tensor,
             hard_ratio: float = 0.7) -> torch.Tensor:
    """
    Online Hard Example Mining at pixel level.
    Computes BCE per pixel, keeps only the hardest hard_ratio fraction.
    Forces the model to focus on its worst mistakes each step.
    """
    pixel_loss = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    ).view(-1)
    k = max(1, int(pixel_loss.numel() * hard_ratio))
    hard, _ = pixel_loss.topk(k)
    return hard.mean()


def multiscale_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                    scales=((1.0, 0.5), (0.5, 0.3), (0.25, 0.2))) -> torch.Tensor:
    """
    Multi-scale supervision: compute BCE+Dice at 3 spatial scales.
    Coarse scales capture global structure; fine scale captures detail.
    """
    loss = 0.0
    for scale, weight in scales:
        if scale < 1.0:
            p = F.interpolate(pred_logits, scale_factor=scale,
                              mode="bilinear", align_corners=False)
            t = F.interpolate(target, scale_factor=scale, mode="nearest")
        else:
            p, t = pred_logits, target
        bce  = F.binary_cross_entropy_with_logits(p, t)
        dice = dice_loss(p, t)
        loss += weight * (0.5 * bce + 0.5 * dice)
    return loss


def compute_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                 boundary_w: float = 0.3,
                 thin_w:     float = 0.2,
                 hard_ratio: float = 0.7) -> torch.Tensor:
    """
    Combined loss:
      multi-scale BCE+Dice  (main supervision)
    + boundary-aware BCE    (edge precision)
    + thin-structure Dice   (crack/seam focus)
    + OHEM BCE              (hard negative mining)
    """
    ms   = multiscale_loss(pred_logits, target)
    bnd  = boundary_loss(pred_logits, target)
    thin = thin_structure_loss(pred_logits, target)
    ohem = ohem_bce(pred_logits, target, hard_ratio)
    return ms + boundary_w * bnd + thin_w * thin + 0.2 * ohem


# ---------------------------------------------------------------------------
# Graduated layer-wise LR parameter groups
# ---------------------------------------------------------------------------

def build_param_groups(model, decoder_lr: float, base_encoder_lr: float,
                       unfreeze_layers: int):
    """
    Returns AdamW param groups with graduated LR across CLIP vision layers.

    Vision encoder (12 layers):
      - Layers 0 … (12 - unfreeze_layers - 1): frozen (requires_grad=False)
      - Layers (12 - unfreeze_layers) … 11:    graduated LR,
            from base_encoder_lr (earliest unfrozen) to base_encoder_lr*10 (last)
    Text encoder:    base_encoder_lr * 0.5  (light fine-tuning)
    Decoder:         decoder_lr             (main learning)
    """
    vision_layers = list(model.clip.vision_model.encoder.layers)
    n_layers = len(vision_layers)
    freeze_up_to = n_layers - unfreeze_layers

    # Freeze early layers
    for i, layer in enumerate(vision_layers):
        for p in layer.parameters():
            p.requires_grad = (i >= freeze_up_to)

    # Also freeze vision embeddings if freezing most layers
    if freeze_up_to > 0:
        for p in model.clip.vision_model.embeddings.parameters():
            p.requires_grad = False

    groups = []

    # Decoder
    decoder_params = [p for n, p in model.named_parameters()
                      if ("clip_based_decode" in n or "decoder" in n)
                      and p.requires_grad]
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr,
                       "name": "decoder"})

    # Graduated vision encoder layers
    for i in range(freeze_up_to, n_layers):
        depth = (i - freeze_up_to) / max(1, unfreeze_layers - 1)   # 0→1
        lr_i  = base_encoder_lr * (1 + 9 * depth)                  # 1×→10×
        params = [p for p in vision_layers[i].parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr_i,
                           "name": f"vision_layer_{i}"})

    # Vision post-norm (if unfrozen)
    post_norm_params = [p for n, p in model.clip.vision_model.named_parameters()
                        if "post_layernorm" in n and p.requires_grad]
    if post_norm_params:
        groups.append({"params": post_norm_params,
                       "lr": base_encoder_lr * 5, "name": "vision_post_norm"})

    # Text encoder (light)
    text_params = [p for n, p in model.clip.text_model.named_parameters()
                   if p.requires_grad]
    if text_params:
        groups.append({"params": text_params,
                       "lr": base_encoder_lr * 0.5, "name": "text_encoder"})

    # Summary
    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"  Param groups: {len(groups)}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Vision layers frozen: {freeze_up_to}/{n_layers}  "
          f"unfrozen: {unfreeze_layers}  "
          f"(LR range {base_encoder_lr:.1e}→{base_encoder_lr*10:.1e})")
    print(f"  Decoder LR: {decoder_lr:.1e}  |  Text encoder LR: {base_encoder_lr*0.5:.1e}")

    return groups


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler,
                    boundary_w, thin_w, hard_ratio):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        pixel_values  = batch["pixel_values"].to(device)
        input_ids     = batch["input_ids"].to(device)
        attn_mask     = batch["attention_mask"].to(device)
        masks         = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attn_mask)
            logits = outputs.logits.unsqueeze(1)   # (B, 1, 352, 352)
            loss   = compute_loss(logits, masks, boundary_w, thin_w, hard_ratio)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, boundary_w, thin_w):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  val  ", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        masks        = batch["mask"].to(device)

        outputs = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attn_mask)
        logits = outputs.logits.unsqueeze(1)
        # Val uses full loss minus OHEM so it's comparable
        loss = (multiscale_loss(logits, masks)
                + boundary_w * boundary_loss(logits, masks)
                + thin_w     * thin_structure_loss(logits, masks))
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",           default=".")
    parser.add_argument("--model-id",       default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--epochs",         type=int,   default=25)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=5e-5,
                        help="Decoder learning rate")
    parser.add_argument("--encoder-lr",     type=float, default=5e-6,
                        help="Base LR for earliest unfrozen encoder layer")
    parser.add_argument("--unfreeze-layers",type=int,   default=8,
                        help="How many CLIP vision encoder layers to unfreeze (0–12)")
    parser.add_argument("--warmup-epochs",  type=int,   default=2)
    parser.add_argument("--boundary-weight",type=float, default=0.3)
    parser.add_argument("--thin-weight",    type=float, default=0.2)
    parser.add_argument("--hard-ratio",     type=float, default=0.7,
                        help="OHEM: fraction of hardest pixels used (0–1)")
    parser.add_argument("--val-split",      type=float, default=0.1)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--output",         default="checkpoints")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} …")
    processor = CLIPSegProcessor.from_pretrained(args.model_id)
    model     = CLIPSegForImageSegmentation.from_pretrained(args.model_id)

    # Build graduated param groups (freeze less of CLIP)
    print("\nBuilding param groups (graduated layer-wise LR):")
    param_groups = build_param_groups(
        model,
        decoder_lr      = args.lr,
        base_encoder_lr = args.encoder_lr,
        unfreeze_layers = args.unfreeze_layers,
    )
    model.to(device)

    # Datasets
    full_ds = build_datasets(args.root, processor, augment=True)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\nLoss weights: boundary={args.boundary_weight}  "
          f"thin={args.thin_weight}  OHEM_ratio={args.hard_ratio}\n")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            args.boundary_weight, args.thin_weight, args.hard_ratio
        )
        val_loss = evaluate(
            model, val_loader, device,
            args.boundary_weight, args.thin_weight
        )
        scheduler.step()

        decoder_lr = scheduler.get_last_lr()[0]
        print(f"  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"decoder_lr={decoder_lr:.2e}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, output_dir / f"clipseg_ep{epoch:02d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, output_dir / "best.pt")
            print(f"  *** New best (val={val_loss:.4f})")

    processor.save_pretrained(str(output_dir / "processor"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {output_dir}")


if __name__ == "__main__":
    main()
