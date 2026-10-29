#!/usr/bin/env python3
"""Smoke test: load all four models and run inference on one video frame."""
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from inference import InferenceManager, FEATURES  # noqa: E402

VIDEO = r"D:/datang/项目数据集/datasets/main.mp4"

print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"cannot read video {VIDEO}")
print(f"frame: {frame.shape} from {VIDEO}")

mgr = InferenceManager("cuda")
enabled = {"vehicle": True, "traffic_sign": True, "emotion": True, "lane": True}
import time
mgr.preload(enabled)
for key in enabled:
    t0 = time.time()
    annotated, info = mgr.process(frame, {k: (k == key) for k in enabled})
    dt = time.time() - t0
    print(f"[{FEATURES[key]}] ok in {dt:.2f}s -> {dict(info)}")
print("SMOKE TEST PASSED")
