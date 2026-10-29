#!/usr/bin/env python3
"""Run the full four-model pipeline over a clip and save annotated video + preview."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import InferenceManager  # noqa: E402

VIDEO = r"D:/datang/项目数据集/datasets/main.mp4"
OUT_DIR = ROOT / "runs" / "reproduction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT_DIR / "annotated_main.mp4"
OUT_PNG = OUT_DIR / "annotated_preview.png"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

mgr = InferenceManager("cuda")
enabled = {"vehicle": True, "traffic_sign": True, "emotion": True, "lane": True}

N_FRAMES = 300  # ~12s clip
preview_saved = False
i = 0
while i < N_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    annotated, stats = mgr.process(frame, enabled)
    writer.write(annotated)
    if not preview_saved and (stats.get("vehicle") or stats.get("lane")):
        cv2.imwrite(str(OUT_PNG), annotated)
        preview_saved = True
    i += 1
cap.release()
writer.release()
print(f"processed {i} frames -> {OUT_VIDEO}")
print(f"preview: {OUT_PNG} (saved={preview_saved})")
print("last stats:", stats)
