#!/usr/bin/env python3
"""Search probability smoothing and morphology for lane-mask IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_lane_swin import LaneDataset, build_lane_model


def min_pool(inputs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return -F.max_pool2d(-inputs, kernel_size, stride=1, padding=kernel_size // 2)


def variants(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    average3 = F.avg_pool2d(probabilities, 3, stride=1, padding=1)
    average5 = F.avg_pool2d(probabilities, 5, stride=1, padding=2)
    return {
        "raw": probabilities,
        "average3": average3,
        "average5": average5,
        "erode3": min_pool(probabilities, 3),
        "erode5": min_pool(probabilities, 5),
        "erode7": min_pool(probabilities, 7),
        "close3": min_pool(F.max_pool2d(probabilities, 3, stride=1, padding=1), 3),
        "open3": F.max_pool2d(min_pool(probabilities, 3), 3, stride=1, padding=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = LaneDataset(
        args.data,
        args.split,
        tuple(checkpoint.get("size", (288, 512))),
        mask_dilation=1,
        soft_mask=False,
        crop_top=float(checkpoint.get("crop_top", 0.0)),
    )
    device = torch.device(args.device)
    model = build_lane_model(checkpoint.get("architecture", "compact"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    thresholds = [round(0.5 + index * 0.01, 2) for index in range(50)]
    counts: dict[tuple[str, float], list[int]] = {}

    with torch.no_grad():
        for images, masks in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            target = (masks.to(device) >= 0.5)
            probabilities = model(images.to(device)).sigmoid()
            for name, candidate in variants(probabilities).items():
                for threshold in thresholds:
                    prediction = candidate >= threshold
                    key = (name, threshold)
                    bucket = counts.setdefault(key, [0, 0, 0, 0, 0])
                    bucket[0] += int((prediction & target).sum())
                    bucket[1] += int((prediction | target).sum())
                    bucket[2] += int(prediction.sum())
                    bucket[3] += int(target.sum())
                    bucket[4] += int(target.numel())

    results = []
    for (name, threshold), (intersection, union, predicted, expected, total) in counts.items():
        precision = intersection / max(predicted, 1)
        recall = intersection / max(expected, 1)
        false_pixels = predicted + expected - 2 * intersection
        results.append(
            {
                "mode": name,
                "threshold": threshold,
                "mean_iou": intersection / max(union, 1),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "pixel_accuracy": 1 - false_pixels / max(total, 1),
                "predicted_foreground_ratio": predicted / max(total, 1),
                "target_foreground_ratio": expected / max(total, 1),
            }
        )
    results.sort(key=lambda item: item["mean_iou"], reverse=True)
    print(json.dumps({"best": results[0], "top": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
