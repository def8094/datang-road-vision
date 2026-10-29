#!/usr/bin/env python3
"""Evaluate the PyTorch/OpenCV driver emotion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_emotion_cnn import EmotionDataset, build_model, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = EmotionDataset(
        args.data,
        "test",
        image_size=int(checkpoint.get("image_size", 48)),
        channels=int(checkpoint.get("channels", 1)),
    )
    model = build_model(checkpoint.get("arch", "simple"), len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    loss, accuracy = evaluate(model, DataLoader(dataset, args.batch_size), torch.device(args.device), torch.nn.CrossEntropyLoss())
    print(json.dumps({"loss": loss, "accuracy": accuracy, "classes": checkpoint["classes"]}))


if __name__ == "__main__":
    main()
