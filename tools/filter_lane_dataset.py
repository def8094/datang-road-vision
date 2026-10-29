#!/usr/bin/env python3
"""Build a traceable TuSimple subset after lane-mask geometry quality checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def mask_path(root: Path, split: str, image_path: Path) -> Path:
    candidate = root / "masks" / split / image_path.name
    if candidate.exists():
        return candidate
    candidate = candidate.with_suffix(".png")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing mask for {image_path}")


def geometry(mask_file: Path) -> dict[str, float | int]:
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read {mask_file}")
    foreground = mask >= 128
    active_rows = np.flatnonzero(foreground.any(axis=1))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    minimum_area = max(20, int(foreground.size * 0.00005))
    large_components = int((stats[1:, cv2.CC_STAT_AREA] >= minimum_area).sum()) if component_count > 1 else 0
    return {
        "foreground_ratio": float(foreground.mean()),
        "row_coverage": float(len(active_rows) / foreground.shape[0]),
        "top_y_ratio": float(active_rows[0] / foreground.shape[0]) if len(active_rows) else 1.0,
        "large_components": large_components,
    }


def rejection_reasons(metrics: dict[str, float | int], args: argparse.Namespace) -> list[str]:
    reasons = []
    if metrics["foreground_ratio"] < args.min_foreground:
        reasons.append("foreground_low")
    if metrics["foreground_ratio"] > args.max_foreground:
        reasons.append("foreground_high")
    if metrics["row_coverage"] < args.min_row_coverage:
        reasons.append("row_coverage_low")
    if metrics["row_coverage"] > args.max_row_coverage:
        reasons.append("row_coverage_high")
    if metrics["top_y_ratio"] < args.min_top_y:
        reasons.append("top_y_low")
    if metrics["top_y_ratio"] > args.max_top_y:
        reasons.append("top_y_high")
    if not args.min_components <= metrics["large_components"] <= args.max_components:
        reasons.append("component_count")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/processed/lane_tusimple_full"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/lane_tusimple_filtered"))
    parser.add_argument("--min-foreground", type=float, default=0.03)
    parser.add_argument("--max-foreground", type=float, default=0.06)
    parser.add_argument("--min-row-coverage", type=float, default=0.58)
    parser.add_argument("--max-row-coverage", type=float, default=0.73)
    parser.add_argument("--min-top-y", type=float, default=0.27)
    parser.add_argument("--max-top-y", type=float, default=0.41)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=5)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")

    rows = []
    reason_counts: Counter[str] = Counter()
    accepted = 0
    for image_file in sorted((args.source / "images" / "train").glob("*")):
        source_mask = mask_path(args.source, "train", image_file)
        metrics = geometry(source_mask)
        reasons = rejection_reasons(metrics, args)
        reason_counts.update(reasons or ["accepted"])
        if not reasons:
            link_or_copy(image_file, args.output / "images" / "train" / image_file.name)
            link_or_copy(source_mask, args.output / "masks" / "train" / source_mask.name)
            accepted += 1
        rows.append({"image": image_file.name, **metrics, "accepted": not reasons, "reasons": ";".join(reasons)})

    validation_count = 0
    for image_file in sorted((args.source / "images" / "val").glob("*")):
        source_mask = mask_path(args.source, "val", image_file)
        link_or_copy(image_file, args.output / "images" / "val" / image_file.name)
        link_or_copy(source_mask, args.output / "masks" / "val" / source_mask.name)
        validation_count += 1

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": str(args.source),
        "task": "binary_lane_line_segmentation",
        "filter": {
            "foreground_ratio": [args.min_foreground, args.max_foreground],
            "row_coverage": [args.min_row_coverage, args.max_row_coverage],
            "top_y_ratio": [args.min_top_y, args.max_top_y],
            "large_components": [args.min_components, args.max_components],
        },
        "counts": {
            "train_source": len(rows),
            "train_accepted": accepted,
            "train_rejected": len(rows) - accepted,
            "val_unfiltered": validation_count,
        },
        "reason_counts": dict(reason_counts),
        "validation_policy": "The complete unfiltered source validation split is retained.",
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
