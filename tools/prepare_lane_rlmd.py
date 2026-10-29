#!/usr/bin/env python3
"""Materialize RLMD road-marking labels as binary lane-line masks.

RLMD provides RGB semantic masks for 25 road-marking classes.  This script
keeps lane-like line classes and excludes crosswalks, arrows, and other road
symbols so the result matches the binary mask contract used by the Swin lane
trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# Solid/dashed lane lines, double lines, and channelizing lines.
LINE_CLASS_IDS = {4, 5, 6, 7, 8, 9, 10, 16}


def _read_colors(classes_csv: Path) -> dict[int, tuple[int, int, int]]:
    colors: dict[int, tuple[int, int, int]] = {}
    with classes_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            colors[int(row["id"])] = (int(row["r"]), int(row["g"]), int(row["b"]))
    return colors


def _mask(label_path: Path, colors: dict[int, tuple[int, int, int]]) -> Image.Image:
    label = np.asarray(Image.open(label_path).convert("RGB"))
    encoded = (
        label[:, :, 0].astype(np.uint32) * 65536
        + label[:, :, 1].astype(np.uint32) * 256
        + label[:, :, 2].astype(np.uint32)
    )
    line_colors = np.array(
        [r * 65536 + g * 256 + b for class_id, (r, g, b) in colors.items() if class_id in LINE_CLASS_IDS],
        dtype=np.uint32,
    )
    output = np.where(np.isin(encoded, line_colors), 255, 0).astype(np.uint8)
    return Image.fromarray(output, mode="L")


def materialize(source: Path, output: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    source_root = source / "RLMD_1080p"
    classes_csv = source / "classes.csv"
    if not classes_csv.exists():
        classes_csv = source.parent / "classes.csv"
    if not classes_csv.exists():
        classes_csv = source_root / "classes.csv"
    colors = _read_colors(classes_csv)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        image_dir = output / "images" / split
        mask_dir = output / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted((source_root / "images" / split).glob("*.jpg"))
        for image_path in image_paths:
            name = f"rlmd_{image_path.stem}.jpg"
            shutil.copyfile(image_path, image_dir / name)
            _mask(source_root / "labels" / split / f"{image_path.stem}.png", colors).save(
                mask_dir / f"rlmd_{image_path.stem}.png", compress_level=1
            )
        counts[split] = len(image_paths)

    manifest = {
        "source": "https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset",
        "license": "MIT",
        "task": "binary_lane_line_segmentation",
        "source_image_size": [1920, 1080],
        "line_class_ids": sorted(LINE_CLASS_IDS),
        "excluded_classes": [0, 1, 2, 3, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24],
        "counts": counts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/raw/lane_rlmd/extracted"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/lane_rlmd"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.output, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
