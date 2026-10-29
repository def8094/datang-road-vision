#!/usr/bin/env python3
"""Evaluate the Swin V2 lane segmentation checkpoint with mean IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_lane_swin import LaneDataset, build_lane_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--mask-dilation", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mask_dilation = int(checkpoint.get("mask_dilation", 1))
    if args.mask_dilation is not None:
        mask_dilation = max(1, args.mask_dilation)
    dataset = LaneDataset(
        args.data,
        args.split,
        tuple(checkpoint.get("size", (288, 512))),
        mask_dilation=mask_dilation,
        soft_mask=False,
        crop_top=float(checkpoint.get("crop_top", 0.0)),
    )
    model = build_lane_model(checkpoint.get("architecture", "compact"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    intersection = union = predicted_pixels = target_pixels = total_pixels = 0
    with torch.no_grad():
        for images, masks in DataLoader(dataset, args.batch_size):
            prediction = model(images.to(args.device)).sigmoid() > threshold
            target = (masks >= 0.5).to(args.device)
            intersection += int((prediction & target).sum())
            union += int((prediction | target).sum())
            predicted_pixels += int(prediction.sum())
            target_pixels += int(target.sum())
            total_pixels += int(target.numel())
    false_pixels = predicted_pixels + target_pixels - 2 * intersection
    precision = intersection / max(predicted_pixels, 1)
    recall = intersection / max(target_pixels, 1)
    print(json.dumps({
        "mean_iou": intersection / max(union, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "pixel_accuracy": 1 - false_pixels / max(total_pixels, 1),
        "predicted_foreground_ratio": predicted_pixels / max(total_pixels, 1),
        "target_foreground_ratio": target_pixels / max(total_pixels, 1),
        "threshold": threshold,
    }))


if __name__ == "__main__":
    main()
