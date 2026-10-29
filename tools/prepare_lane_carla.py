#!/usr/bin/env python3
"""Download and materialize a reproducible CARLA lane-line subset.

CARLA's raw semantic masks encode RoadLine as class id 24.  This script keeps
only that class and writes the same image/mask folder contract used by the
Swin V2 trainer.  Town01 is split into train/val; Town05 eval is kept as an
unseen-map test split.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter


REPO = "sungendary/carla-lane-segmentation"


def _download(cache: Path, filename: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(REPO, filename, repo_type="dataset", local_dir=cache))


def _extract_archive(archive: Path, destination: Path) -> Path:
    if destination.exists() and any(destination.iterdir()):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    return destination


def _pairs(root: Path) -> list[tuple[Path, Path]]:
    rgb = sorted(path for path in root.rglob("rgb/*.png"))
    pairs = []
    for image_path in rgb:
        label_path = image_path.parent.parent / "label_raw" / image_path.name
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _write_split(pairs: list[tuple[Path, Path]], output: Path, split: str, start: int = 0) -> int:
    image_dir = output / "images" / split
    mask_dir = output / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for offset, (image_path, label_path) in enumerate(pairs, start=start):
        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path).convert("RGB"))[:, :, 0]
        lane_mask = Image.fromarray(np.where(label == 24, 255, 0).astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(5))
        name = f"carla_{offset:06d}.png"
        image.save(image_dir / name, compress_level=3)
        lane_mask.save(mask_dir / name, compress_level=3)
        written += 1
    return written


def materialize(output: Path, cache: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    train_archive = _download(cache, "train/town01.tar.gz")
    test_archive = _download(cache, "eval/town05_eval.tar.gz")
    output.mkdir(parents=True, exist_ok=True)

    train_root = _extract_archive(train_archive, cache / "extracted" / "town01")
    train_pairs = _pairs(train_root)
    val_count = max(1, len(train_pairs) // 10)
    train_pairs, val_pairs = train_pairs[val_count:], train_pairs[:val_count]
    train_written = _write_split(train_pairs, output, "train")
    val_written = _write_split(val_pairs, output, "val", start=0)

    test_root = _extract_archive(test_archive, cache / "extracted" / "town05_eval")
    test_pairs = _pairs(test_root)
    test_written = _write_split(test_pairs, output, "test")

    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "license": "MIT",
        "label_mapping": {"road_line_raw_id": 24, "foreground": 255, "background": 0},
        "mask_dilation_kernel": 5,
        "archive_selection": ["train/town01.tar.gz", "eval/town05_eval.tar.gz"],
        "counts": {"train": train_written, "val": val_written, "test": test_written},
        "test_is_unseen_map": True,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/lane_carla"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/raw/carla_lane_archives"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.cache, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
