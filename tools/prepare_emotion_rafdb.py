#!/usr/bin/env python3
"""Materialize the public RAF-DB 7-emotion Hugging Face mirror.

The source Parquet files contain a Hugging Face feature metadata variant that
the installed ``datasets`` version cannot decode, while Polars can read the
image struct and label columns directly.  Exporting with Polars keeps the
source train/validation/test split and produces the image-folder contract used
by the existing OpenCV + PyTorch trainer: ``split/class/image.jpg``.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import polars as pl
from PIL import Image


SOURCE = "Pelmeshek/raf-db-7emotions-mediapipe-768"
LABEL_NAMES = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}


def _source_files() -> dict[str, list[str]]:
    return {
        "train": [f"train-{index:05d}-of-00004.parquet" for index in range(4)],
        "val": ["val-00000-of-00001.parquet"],
        "test": ["test-00000-of-00001.parquet"],
    }


def materialize(output: Path, raw_root: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    counts: dict[str, Counter[str]] = {}
    for split_name, file_names in _source_files().items():
        counts[split_name] = Counter()
        for file_name in file_names:
            source = raw_root / file_name
            if not source.exists() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing RAF-DB Parquet file: {source}")
            table = pl.read_parquet(source, columns=["image", "sample_id", "label"])
            for index, row in enumerate(table.iter_rows(named=True)):
                label_id = int(row["label"])
                class_name = LABEL_NAMES[label_id]
                sample_id = str(row["sample_id"] or f"{split_name}_{index:06d}")
                safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)
                target = output / split_name / class_name / f"{safe_id}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                image_bytes = row["image"]["bytes"]
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.convert("RGB").save(target, quality=95)
                counts[split_name][class_name] += 1

    manifest = {
        "source": f"https://huggingface.co/datasets/{SOURCE}",
        "source_license": "other; RAF-DB-derived mirror, review upstream terms before redistribution",
        "classes": [LABEL_NAMES[index] for index in range(7)],
        "splits": {split: dict(value) for split, value in counts.items()},
        "total": sum(sum(value.values()) for value in counts.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/driver_emotion_rafdb"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw/rafdb_parquet"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.raw_root, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
