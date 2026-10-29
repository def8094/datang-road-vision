#!/usr/bin/env python3
"""Predict a lane mask and overlay it on one road image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_lane_swin import build_lane_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/lane_overlay.jpg"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    height, width = checkpoint.get("size", (288, 512))
    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read {args.source}")
    crop_top = float(checkpoint.get("crop_top", 0.0))
    top = int(round(image.shape[0] * crop_top))
    cropped = image[top:, :] if top > 0 else image
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model = build_lane_model(checkpoint.get("architecture", "compact"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    with torch.no_grad():
        mask = (model(tensor.to(args.device)).sigmoid()[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    overlay = resized.copy()
    overlay[mask > 0] = (0, 255, 0)
    result = cv2.addWeighted(resized, 0.65, overlay, 0.35, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
