#!/usr/bin/env python3
"""Download and materialize the public FERPlus 7-class dataset.

The source dataset is ``deanngkl/ferplus-7cls`` on Hugging Face.  It contains
the FERPlus majority-vote labels, which are cleaner than the original
FER2013 single-rater labels used by the first experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def materialize(output: Path, seed: int = 42, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    dataset = load_dataset("deanngkl/ferplus-7cls", split="train")
    first = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
    second = first["test"].train_test_split(test_size=0.5, seed=seed, stratify_by_column="label")
    splits = {"train": first["train"], "val": second["train"], "test": second["test"]}
    class_names = dataset.features["label"].names
    counts = {}
    for split_name, split in splits.items():
        counts[split_name] = Counter()
        for index, row in enumerate(split):
            label = int(row["label"])
            class_name = class_names[label]
            target = output / split_name / class_name / f"{split_name}_{index:06d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            image = row["image"].convert("L")
            image.save(target, quality=95)
            counts[split_name][class_name] += 1

    manifest = {
        "source": "https://huggingface.co/datasets/deanngkl/ferplus-7cls",
        "classes": class_names,
        "seed": seed,
        "counts": {split: dict(value) for split, value in counts.items()},
        "total": sum(sum(value.values()) for value in counts.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/driver_emotion_ferplus"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.seed, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
