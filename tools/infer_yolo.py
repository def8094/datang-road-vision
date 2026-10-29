#!/usr/bin/env python3
"""Run YOLO26 inference on an image, directory, video, or camera source."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.model}")
    from ultralytics import YOLO
    results = YOLO(str(args.model)).predict(source=args.source, conf=args.conf, device=args.device, save=True, project=str(args.output.parent), name=args.output.name)
    print(f"saved={args.output}")
    for result in results:
        print(f"detections={0 if result.boxes is None else len(result.boxes)}")


if __name__ == "__main__":
    main()
