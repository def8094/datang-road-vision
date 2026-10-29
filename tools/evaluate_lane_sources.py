#!/usr/bin/env python3
"""Evaluate one lane checkpoint across source-specific validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_lane_swin import LaneDataset, build_lane_model


def evaluate(model: torch.nn.Module, dataset: LaneDataset, device: torch.device, thresholds: list[float], batch_size: int) -> dict:
    counts = {threshold: [0, 0] for threshold in thresholds}
    with torch.no_grad():
        for images, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            probabilities = model(images.to(device)).sigmoid()
            target = (masks >= 0.5).to(device)
            for threshold in thresholds:
                prediction = probabilities > threshold
                intersection = int((prediction & target).sum())
                union = int((prediction | target).sum())
                counts[threshold][0] += intersection
                counts[threshold][1] += union
    scores = {str(threshold): intersection / max(union, 1) for threshold, (intersection, union) in counts.items()}
    best_threshold = max(thresholds, key=lambda threshold: scores[str(threshold)])
    return {
        "best_threshold": best_threshold,
        "best_mean_iou": scores[str(best_threshold)],
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tusimple", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--rlmd", type=Path, default=Path("datasets/processed/lane_rlmd"))
    parser.add_argument("--mixed", type=Path, default=Path("datasets/processed/lane_tusimple_rlmd"))
    parser.add_argument("--mask-dilation-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = tuple(checkpoint.get("size", (288, 512)))
    mask_dilation = int(checkpoint.get("mask_dilation", 1))
    if args.mask_dilation_override is not None:
        mask_dilation = max(1, args.mask_dilation_override)
    model = build_lane_model(checkpoint.get("architecture", "compact"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    thresholds = [round(index / 20, 2) for index in range(4, 20)]

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mask_dilation": mask_dilation,
        "size": size,
        "sources": {},
    }
    for name, root in (("tusimple", args.tusimple), ("rlmd", args.rlmd), ("mixed", args.mixed)):
        dataset = LaneDataset(
            root,
            "val",
            size,
            mask_dilation=mask_dilation,
            soft_mask=False,
            crop_top=float(checkpoint.get("crop_top", 0.0)),
        )
        results["sources"][name] = evaluate(model, dataset, device, thresholds, args.batch_size)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
