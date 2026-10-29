#!/usr/bin/env python3
"""Train or validate a YOLO26 detector on a prepared data.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO26 checkpoint/config; other YOLO versions are rejected")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="traffic_sign")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--time-hours", type=float, default=1.8, help="Ultralytics time budget in hours")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data config does not exist: {args.data}")

    from ultralytics import YOLO

    if "yolo26" not in args.model.lower():
        raise SystemExit(f"Only YOLO26 is allowed, received: {args.model}")
    model = YOLO(args.model)
    yaml_file = str(getattr(model.model, "yaml", {}).get("yaml_file", "")).lower()
    if "yolo26" not in yaml_file:
        raise SystemExit(f"Only YOLO26 is allowed, received model config: {yaml_file or 'unknown'}")
    if args.val_only:
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device, project=args.project, name=args.name)
        print(metrics)
        return
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        time=args.time_hours,
    )
    print(results)


if __name__ == "__main__":
    main()
