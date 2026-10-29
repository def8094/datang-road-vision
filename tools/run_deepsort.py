#!/usr/bin/env python3
"""Run YOLO detections through DeepSORT on a video or camera stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Video path or camera index")
    parser.add_argument("--detector", required=True, help="YOLO26 vehicle checkpoint")
    parser.add_argument("--output", type=Path, default=Path("runs/tracking.mp4"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from deep_sort_realtime.deepsort_tracker import DeepSort
    from ultralytics import YOLO

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = YOLO(args.detector)
    yaml_file = str(getattr(detector.model, "yaml", {}).get("yaml_file", "")).lower()
    if "yolo26" not in yaml_file:
        raise SystemExit("The detector must be a YOLO26 checkpoint or a local trained YOLO26 checkpoint")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
            detections = []
            if result.boxes is not None:
                names = result.names
                for box, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], float(confidence), names[int(class_id)]))
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                left, top, right, bottom = map(int, track.to_ltrb())
                label = f"{track.get_det_class() or 'object'} #{track.track_id}"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
                cv2.putText(frame, label, (left, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    print(f"tracking_output={args.output}")


if __name__ == "__main__":
    main()
