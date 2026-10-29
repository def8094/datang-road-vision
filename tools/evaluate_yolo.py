#!/usr/bin/env python3
"""Evaluate a trained YOLO26 detector and print machine-readable metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.model}")
    from ultralytics import YOLO
    metrics = YOLO(str(args.model)).val(data=str(args.data), device=args.device, verbose=False)
    result = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "model": str(args.model)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
