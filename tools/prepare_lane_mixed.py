#!/usr/bin/env python3
"""Combine TuSimple and RLMD binary lane masks under one traceable dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy_split(source: Path, output: Path, split: str, prefix: str) -> int:
    image_dir = output / "images" / split
    mask_dir = output / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted((source / "images" / split).glob("*")):
        mask_path = source / "masks" / split / image_path.name
        if not mask_path.exists():
            mask_path = source / "masks" / split / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        shutil.copyfile(image_path, image_dir / f"{prefix}_{image_path.name}")
        shutil.copyfile(mask_path, mask_dir / f"{prefix}_{mask_path.stem}.png")
        count += 1
    return count


def materialize(tusimple: Path, rlmd: Path, output: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)
    counts = {}
    for split in ("train", "val"):
        counts[f"tusimple_{split}"] = _copy_split(tusimple, output, split, "tusimple")
        counts[f"rlmd_{split}"] = _copy_split(rlmd, output, split, "rlmd")
    manifest = {
        "task": "binary_lane_line_segmentation",
        "sources": [
            "datasets/processed/lane_tusimple_full",
            "datasets/processed/lane_rlmd",
        ],
        "counts": counts,
        "validation_is_mixed_source": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tusimple", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--rlmd", type=Path, default=Path("datasets/processed/lane_rlmd"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/lane_tusimple_rlmd"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.tusimple, args.rlmd, args.output, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
