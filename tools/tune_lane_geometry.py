#!/usr/bin/env python3
"""Tune OpenCV polynomial lane reconstruction against the original masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lane_swin import LaneDataset, build_lane_model


def reconstruct(mask: np.ndarray, degree: int, width_scale: float, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        if len(ys) < max(20, degree + 1):
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        if y_max - y_min < 12:
            continue
        fit_degree = min(degree, len(np.unique(ys)) - 1)
        coefficients = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), fit_degree)
        sample_y = np.arange(y_min, y_max + 1)
        center_x = np.rint(np.polyval(coefficients, sample_y)).astype(np.int32)
        estimated_width = area / max(y_max - y_min + 1, 1)
        half_width = max(1, int(round(estimated_width * width_scale / 2)))
        for y, x in zip(sample_y, center_x):
            left = max(0, int(x) - half_width)
            right = min(mask.shape[1], int(x) + half_width + 1)
            output[y, left:right] = 1
    return output.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = LaneDataset(
        args.data,
        "val",
        tuple(checkpoint.get("size", (288, 512))),
        mask_dilation=1,
        soft_mask=False,
        crop_top=float(checkpoint.get("crop_top", 0.0)),
    )
    device = torch.device(args.device)
    model = build_lane_model(checkpoint.get("architecture", "compact"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    configurations = [
        (degree, width_scale, min_area)
        for degree in (2, 3)
        for width_scale in (0.75, 1.0, 1.25, 1.5)
        for min_area in (8, 20, 40)
    ]
    thresholds = [round(value, 2) for value in np.arange(0.2, 0.81, 0.05)]
    counts: dict[tuple[int, float, int, float], list[int]] = {}
    with torch.no_grad():
        for images, masks in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            probabilities = model(images.to(device)).sigmoid().cpu().numpy()[:, 0]
            targets = (masks.numpy()[:, 0] >= 0.5)
            for threshold in thresholds:
                for index in range(len(probabilities)):
                    raw = probabilities[index] >= threshold
                    target = targets[index]
                    for degree, width_scale, min_area in configurations:
                        prediction = reconstruct(raw, degree, width_scale, min_area)
                        key = (degree, width_scale, min_area, threshold)
                        bucket = counts.setdefault(key, [0, 0, 0, 0, 0])
                        intersection = int((prediction & target).sum())
                        union = int((prediction | target).sum())
                        predicted = int(prediction.sum())
                        expected = int(target.sum())
                        bucket[0] += intersection
                        bucket[1] += union
                        bucket[2] += predicted
                        bucket[3] += expected
                        bucket[4] += int(target.size)

    results = []
    for (degree, width_scale, min_area, threshold), values in counts.items():
        intersection, union, predicted, expected, total = values
        precision = intersection / max(predicted, 1)
        recall = intersection / max(expected, 1)
        results.append(
            {
                "degree": degree,
                "width_scale": width_scale,
                "min_area": min_area,
                "threshold": threshold,
                "mean_iou": intersection / max(union, 1),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "pixel_accuracy": 1 - (predicted + expected - 2 * intersection) / max(total, 1),
            }
        )
    results.sort(key=lambda item: item["mean_iou"], reverse=True)
    print(json.dumps({"best": results[0], "top": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
