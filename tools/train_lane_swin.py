#!/usr/bin/env python3
"""Train a compact Swin V2 encoder + decoder for binary lane segmentation."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Swin_V2_T_Weights, swin_v2_t
from torchvision.transforms import functional as TF

try:
    from transformers.models.upernet.modeling_upernet import UperNetHead
except ImportError:
    UperNetHead = None


class LaneDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        size: tuple[int, int],
        augment: bool = False,
        mask_dilation: int = 5,
        soft_mask: bool = True,
        crop_top: float = 0.0,
    ):
        self.images = sorted((root / "images" / split).glob("*"))
        self.masks = root / "masks" / split
        self.size = size
        self.augment = augment
        self.soft_mask = soft_mask
        self.crop_top = min(max(crop_top, 0.0), 0.8)
        self.mask_dilation = mask_dilation if mask_dilation % 2 == 1 else mask_dilation + 1
        if not self.images:
            raise FileNotFoundError(f"No lane images found in {root / 'images' / split}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        mask_path = self.masks / image_path.name
        if not mask_path.exists():
            mask_path = self.masks / f"{image_path.stem}.png"
        mask = Image.open(mask_path).convert("L")
        if self.crop_top > 0:
            top = int(round(image.height * self.crop_top))
            crop_height = image.height - top
            image = TF.crop(image, top, 0, crop_height, image.width)
            mask = TF.crop(mask, top, 0, crop_height, mask.width)
        if not self.soft_mask:
            mask = mask.point(lambda value: 255 if value >= 128 else 0)
        image = TF.resize(image, self.size, antialias=True)
        # Preserve sub-pixel occupancy of thin lane markings when requested.
        interpolation = TF.InterpolationMode.BILINEAR if self.soft_mask else TF.InterpolationMode.NEAREST
        mask = TF.resize(mask, self.size, interpolation=interpolation)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        if not self.soft_mask:
            mask = (mask >= 0.5).float()
        if self.mask_dilation > 1:
            padding = self.mask_dilation // 2
            mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=self.mask_dilation, stride=1, padding=padding).squeeze(0)
        return image, mask


class SwinV2Lane(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
        backbone = swin_v2_t(weights=weights)
        self.features = backbone.features
        self.decode3 = self._block(768 + 384, 384)
        self.decode2 = self._block(384 + 192, 192)
        self.decode1 = self._block(192 + 96, 96)
        self.decode0 = self._block(96, 64)
        self.head = nn.Conv2d(64, 1, 1)
        # Lane pixels are sparse; start near the observed foreground prior.
        nn.init.constant_(self.head.bias, -2.5)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep intermediate Swin stages so thin lane markings retain spatial detail.
        skips = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (1, 3, 5, 7):
                skips.append(x.permute(0, 3, 1, 2).contiguous())
        stage1, stage2, stage3, stage4 = skips
        x = F.interpolate(stage4, size=stage3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode3(torch.cat([x, stage3], dim=1))
        x = F.interpolate(x, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode2(torch.cat([x, stage2], dim=1))
        x = F.interpolate(x, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode1(torch.cat([x, stage1], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decode0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


class SwinV2LaneDetail(nn.Module):
    """Swin V2 encoder with input-detail fusion for lane boundaries."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
        backbone = swin_v2_t(weights=weights)
        self.features = backbone.features
        self.decode3 = SwinV2Lane._block(768 + 384, 384)
        self.decode2 = SwinV2Lane._block(384 + 192, 192)
        self.decode1 = SwinV2Lane._block(192 + 96, 96)
        self.detail_half = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.detail_full = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.decode0 = SwinV2Lane._block(96 + 32, 64)
        self.decode_full = SwinV2Lane._block(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.head.bias, -2.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        skips = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (1, 3, 5, 7):
                skips.append(x.permute(0, 3, 1, 2).contiguous())
        stage1, stage2, stage3, stage4 = skips
        x = F.interpolate(stage4, size=stage3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode3(torch.cat([x, stage3], dim=1))
        x = F.interpolate(x, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode2(torch.cat([x, stage2], dim=1))
        x = F.interpolate(x, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode1(torch.cat([x, stage1], dim=1))
        detail_half = self.detail_half(inputs)
        x = F.interpolate(x, size=detail_half.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode0(torch.cat([x, detail_half], dim=1))
        detail_full = self.detail_full(inputs)
        x = F.interpolate(x, size=detail_full.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode_full(torch.cat([x, detail_full], dim=1))
        return self.head(x)


class SwinV2LaneUPerNet(nn.Module):
    """Swin V2 encoder with a UPerNet pyramid pooling and FPN decoder."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        if UperNetHead is None:
            raise ImportError("SwinV2LaneUPerNet requires the transformers package")
        weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
        self.features = swin_v2_t(weights=weights).features
        config = SimpleNamespace(pool_scales=(1, 2, 3, 6), hidden_size=128, num_labels=1)
        self.decode_head = UperNetHead(config, in_channels=(96, 192, 384, 768))
        nn.init.constant_(self.decode_head.classifier.bias, -2.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        stages = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (1, 3, 5, 7):
                stages.append(x.permute(0, 3, 1, 2).contiguous())
        logits = self.decode_head(stages)
        return F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)


def build_lane_model(architecture: str, pretrained: bool = False) -> nn.Module:
    if architecture == "detail":
        return SwinV2LaneDetail(pretrained=pretrained)
    if architecture == "compact":
        return SwinV2Lane(pretrained=pretrained)
    if architecture == "upernet":
        return SwinV2LaneUPerNet(pretrained=pretrained)
    raise ValueError(f"Unsupported lane architecture: {architecture}")


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    smooth = 1.0
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - ((2 * intersection + smooth) / (denominator + smooth))).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    false_positive_weight: float = 0.7,
) -> torch.Tensor:
    """Penalize overly wide lane predictions more than missed edge pixels."""
    probabilities = logits.sigmoid()
    dimensions = (1, 2, 3)
    true_positive = (probabilities * target).sum(dim=dimensions)
    false_positive = (probabilities * (1 - target)).sum(dim=dimensions)
    false_negative = ((1 - probabilities) * target).sum(dim=dimensions)
    false_negative_weight = 1 - false_positive_weight
    score = (true_positive + 1.0) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + 1.0
    )
    return (1 - score).mean()


def lovasz_hinge(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lovasz hinge surrogate that directly optimizes the IoU ranking."""
    logits = logits.flatten(1)
    target = target.flatten(1)
    losses = []
    for score, truth in zip(logits, target):
        truth = truth.float()
        signs = 2.0 * truth - 1.0
        errors = 1.0 - score * signs
        errors, order = torch.sort(errors, descending=True)
        truth = truth[order]
        intersection = truth.sum() - truth.cumsum(0)
        union = truth.sum() + (1.0 - truth).cumsum(0)
        gradient = 1.0 - intersection / union.clamp_min(1.0)
        # Keep this term on the same scale as BCE/Dice for full-resolution masks.
        losses.append(torch.dot(F.relu(errors), gradient) / score.numel())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Keep easy background pixels from dominating thin lane supervision."""
    probability = logits.sigmoid()
    probability_true = probability * target + (1 - probability) * (1 - target)
    alpha_true = alpha * target + (1 - alpha) * (1 - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (alpha_true * (1 - probability_true).pow(gamma) * bce).mean()


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    positive_weight: float = 8.0,
    tversky_weight: float = 0.0,
    false_positive_weight: float = 0.7,
    lovasz_weight: float = 0.0,
) -> torch.Tensor:
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
    )
    loss = weighted_bce + 0.75 * dice_loss(logits, masks) + 0.25 * focal_loss(logits, masks, alpha=0.75, gamma=2.0)
    if tversky_weight > 0:
        loss = loss + tversky_weight * tversky_loss(
            logits,
            masks,
            false_positive_weight=false_positive_weight,
        )
    if lovasz_weight > 0:
        loss = loss + lovasz_weight * lovasz_hinge(logits, masks)
    return loss


def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    loss_total = 0.0
    iou_total = 0.0
    batches = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prediction = logits.sigmoid() > 0.5
        intersection = (prediction & masks.bool()).sum().item()
        union = (prediction | masks.bool()).sum().item()
        loss_total += loss.item()
        iou_total += intersection / max(union, 1)
        batches += 1
    return loss_total / max(batches, 1), iou_total / max(batches, 1)


def evaluate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> dict[str, float]:
    model.eval()
    counts = {threshold: [0, 0, 0, 0, 0] for threshold in thresholds}
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            target = (masks.to(device, non_blocking=True) >= 0.5)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = model(images).sigmoid()
            for threshold in thresholds:
                prediction = probabilities >= threshold
                intersection = int((prediction & target).sum())
                union = int((prediction | target).sum())
                predicted = int(prediction.sum())
                expected = int(target.sum())
                total = int(target.numel())
                bucket = counts[threshold]
                bucket[0] += intersection
                bucket[1] += union
                bucket[2] += predicted
                bucket[3] += expected
                bucket[4] += total
    best_threshold = max(thresholds, key=lambda value: counts[value][0] / max(counts[value][1], 1))
    intersection, union, predicted, expected, total = counts[best_threshold]
    false_pixels = predicted + expected - 2 * intersection
    precision = intersection / max(predicted, 1)
    recall = intersection / max(expected, 1)
    return {
        "threshold": best_threshold,
        "mean_iou": intersection / max(union, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "pixel_accuracy": 1 - false_pixels / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-backbone", type=Path)
    parser.add_argument("--architecture", choices=("compact", "detail", "upernet"), default="compact")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--mask-dilation", type=int, default=5)
    parser.add_argument("--hard-mask", action="store_true")
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--tversky-weight", type=float, default=0.0)
    parser.add_argument("--false-positive-weight", type=float, default=0.7)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--crop-top", type=float, default=0.0)
    parser.add_argument("--time-hours", type=float, default=1.8)
    parser.add_argument("--output", type=Path, default=Path("weights/lane_swin_v2.pt"))
    args = parser.parse_args()

    size = (args.height, args.width)
    mask_dilation = max(1, args.mask_dilation)
    train_set = LaneDataset(
        args.data,
        "train",
        size,
        augment=True,
        mask_dilation=mask_dilation,
        soft_mask=not args.hard_mask,
        crop_top=args.crop_top,
    )
    val_set = LaneDataset(
        args.val_data or args.data,
        "val",
        size,
        mask_dilation=1,
        soft_mask=False,
        crop_top=args.crop_top,
    )
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    model = build_lane_model(args.architecture, pretrained=args.pretrained).to(device)
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_architecture = resume_checkpoint.get("architecture", "compact")
        if resume_architecture != args.architecture:
            raise ValueError(
                f"Resume checkpoint architecture {resume_architecture!r} does not match {args.architecture!r}"
            )
        model.load_state_dict(resume_checkpoint["model"])
    if args.resume_backbone:
        backbone_checkpoint = torch.load(args.resume_backbone, map_location="cpu", weights_only=False)
        backbone_state = {
            name: value
            for name, value in backbone_checkpoint["model"].items()
            if name.startswith("features.")
        }
        incompatible = model.load_state_dict(backbone_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected backbone keys: {incompatible.unexpected_keys}")
        print(f"loaded_backbone={args.resume_backbone} tensors={len(backbone_state)}")
    decoder_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("features.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.lr * args.backbone_lr_factor},
            {"params": decoder_parameters, "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    thresholds = [round(index / 20, 2) for index in range(4, 20)]
    deadline = time.monotonic() + args.time_hours * 3600
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_backbone_epochs > 0:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        elif epoch == args.freeze_backbone_epochs + 1:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
        model.train()
        train_loss = train_iou = batches = 0.0
        for images, masks in train_loader:
            if time.monotonic() >= deadline:
                break
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = segmentation_loss(
                    logits,
                    masks,
                    positive_weight=args.positive_weight,
                    tversky_weight=args.tversky_weight,
                    false_positive_weight=args.false_positive_weight,
                    lovasz_weight=args.lovasz_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            prediction = logits.sigmoid() > 0.5
            intersection = (prediction & masks.bool()).sum().item()
            union = (prediction | masks.bool()).sum().item()
            train_loss += loss.item()
            train_iou += intersection / max(union, 1)
            batches += 1
        train_loss /= max(batches, 1)
        train_iou /= max(batches, 1)
        metrics = evaluate_thresholds(model, val_loader, device, thresholds)
        scheduler.step()
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_raw_iou={metrics['mean_iou']:.4f} threshold={metrics['threshold']:.2f} "
            f"f1={metrics['f1']:.4f} pixel_acc={metrics['pixel_accuracy']:.4f}"
        )
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": args.architecture,
                    "epoch": epoch,
                    "val_iou": metrics["mean_iou"],
                    "metrics": metrics,
                    "threshold": metrics["threshold"],
                    "size": size,
                    "crop_top": args.crop_top,
                    "mask_dilation": mask_dilation,
                    "soft_mask": not args.hard_mask,
                    "positive_weight": args.positive_weight,
                    "tversky_weight": args.tversky_weight,
                    "false_positive_weight": args.false_positive_weight,
                    "lovasz_weight": args.lovasz_weight,
                    "resume": str(args.resume) if args.resume else None,
                    "resume_backbone": str(args.resume_backbone) if args.resume_backbone else None,
                    "train_data": str(args.data),
                    "val_data": str(args.val_data or args.data),
                },
                args.output,
            )
        if time.monotonic() >= deadline:
            break
    print(f"best_val_iou={best_iou:.4f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
