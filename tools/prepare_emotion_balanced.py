#!/usr/bin/env python3
"""Create a balanced FER training set with OpenCV-only image augmentation."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    angle = rng.uniform(-10.0, 10.0)
    scale = rng.uniform(0.92, 1.08)
    matrix = cv2.getRotationMatrix2D((24, 24), angle, scale)
    matrix[:, 2] += (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
    image = cv2.warpAffine(image, matrix, (48, 48), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.7:
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for class_dir in sorted((args.input / split).iterdir()):
            if not class_dir.is_dir():
                continue
            destination = args.output / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            sources = sorted(class_dir.glob("*.jpg"))
            for source in sources:
                shutil.copy2(source, destination / source.name)
            if split != "train" or len(sources) >= args.target_per_class:
                continue
            for index in range(len(sources), args.target_per_class):
                source = sources[index % len(sources)]
                image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise OSError(f"Unable to read {source}")
                output_name = destination / f"aug_{index:05d}_{source.stem}.jpg"
                if not cv2.imwrite(str(output_name), augment(image, rng)):
                    raise OSError(f"Unable to write {output_name}")
    print(f"balanced_dataset={args.output}")


if __name__ == "__main__":
    main()
