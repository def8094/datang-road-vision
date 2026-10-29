#!/usr/bin/env python3
"""Materialize the MIT-licensed Teen-Different facial-expression dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")
CLASS_NAMES = {
    "angry": "anger",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "surprised": "surprise",
}


def materialize(source: Path, output: Path, seed: int = 42, val_ratio: float = 0.15, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)
    rng = random.Random(seed)
    counts: dict[str, Counter[str]] = {}
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for source_split in ("train", "test"):
        counts[source_split] = Counter()
        for source_class in CLASSES:
            files = sorted(path for path in (source / source_split / source_class).iterdir() if path.suffix.lower() in extensions)
            rng.shuffle(files)
            if source_split == "train":
                val_count = max(1, round(len(files) * val_ratio))
                assignments = (("val", files[:val_count]), ("train", files[val_count:]))
            else:
                assignments = (("test", files),)
            for split, split_files in assignments:
                target_dir = output / split / CLASS_NAMES[source_class]
                target_dir.mkdir(parents=True, exist_ok=True)
                for index, path in enumerate(split_files):
                    target = target_dir / f"{source_class}_{index:06d}{path.suffix.lower()}"
                    shutil.copy2(path, target)
                    counts[source_split][split] += 1
    manifest = {
        "source": "https://huggingface.co/datasets/Teen-Different/Facial-Expression",
        "license": "MIT",
        "classes": list(CLASS_NAMES.values()),
        "seed": seed,
        "val_ratio_from_train": val_ratio,
        "counts": {source: dict(value) for source, value in counts.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/raw/teen_facial_expression/extracted"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/driver_emotion_teen"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.output, args.seed, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
