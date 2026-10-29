#!/usr/bin/env python3
"""Classify one 48x48-style face image using the trained emotion CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from train_emotion_cnn import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = int(checkpoint.get("channels", 1))
    image_size = int(checkpoint.get("image_size", 48))
    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Unable to read {args.source}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if channels == 1:
        image = cv2.equalizeHist(image)
        tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).unsqueeze(0)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        tensor = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model = build_model(checkpoint.get("arch", "simple"), len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    with torch.no_grad():
        probabilities = model(tensor.to(args.device)).softmax(1)[0]
    index = int(probabilities.argmax())
    print({"class": checkpoint["classes"][index], "confidence": float(probabilities[index])})


if __name__ == "__main__":
    main()
