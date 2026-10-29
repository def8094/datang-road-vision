#!/usr/bin/env python3
"""Unified real-time inference for the four driving-vision models."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_emotion_cnn import build_model as build_emotion_model  # noqa: E402
from train_lane_swin import build_lane_model  # noqa: E402


FEATURES = {
    "vehicle": "车辆检测与 DeepSORT",
    "traffic_sign": "交通标识检测",
    "emotion": "驾驶员表情识别",
    "lane": "车道线检测",
}

# The previous custom vehicle checkpoint was trained from a dataset whose class
# contract is undocumented (CLASS_NAMES_UNVERIFIED.txt). It produced no boxes
# on the user's road footage. The bundled YOLO26 COCO checkpoint is a known,
# valid vehicle detector and is still strictly YOLO26.
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def default_model_paths() -> dict[str, Path]:
    lane_candidate = ROOT / "training/lane_detection/weights/candidate_highres_precision.pt"
    return {
        "vehicle": ROOT / "yolo26n.pt",
        "traffic_sign": ROOT / "training/traffic_sign/weights/best.pt",
        "emotion": ROOT / "training/driver_emotion/weights/best.pt",
        "lane": lane_candidate if lane_candidate.exists() else ROOT / "training/lane_detection/weights/best.pt",
    }


class InferenceManager:
    """Lazy-load model families and annotate one OpenCV BGR frame."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = "cuda:0" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self.paths = default_model_paths()
        self.models: dict[str, Any] = {}
        self.trackers: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.last_vehicle_tracks = 0
        # OpenCV's FileStorage cannot open paths with non-ASCII characters on
        # Windows (e.g. a project folder named in Chinese). Copy the cascade
        # to an ASCII-only temp path before loading in that case.
        cascade_src = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        try:
            str(cascade_src).encode("ascii")
            cascade_path = cascade_src
        except UnicodeEncodeError:
            import shutil
            import tempfile
            ascii_dir = Path(tempfile.gettempdir()) / "datang_cascades"
            ascii_dir.mkdir(parents=True, exist_ok=True)
            cascade_path = ascii_dir / cascade_src.name
            if not cascade_path.exists():
                shutil.copy2(cascade_src, cascade_path)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

    def _load_yolo(self, key: str) -> Any:
        if key not in self.models:
            from ultralytics import YOLO

            path = self.paths[key]
            if not path.exists():
                raise FileNotFoundError(f"{FEATURES[key]}模型不存在: {path}")
            model = YOLO(str(path))
            yaml_file = str(getattr(model.model, "yaml", {}).get("yaml_file", "")).lower()
            if "yolo26" not in yaml_file:
                raise RuntimeError(f"模型 {path.name} 不是 YOLO26 权重，已拒绝加载")
            self.models[key] = model
        return self.models[key]

    def _load_vehicle(self) -> Any:
        model = self._load_yolo("vehicle")
        if "vehicle" not in self.trackers:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            # Show a track from the first matched frame. Raw YOLO boxes are
            # drawn independently, so confirmation latency cannot hide cars.
            self.trackers["vehicle"] = DeepSort(max_age=30, n_init=1, nms_max_overlap=1.0)
        return model

    def _load_emotion(self) -> Any:
        if "emotion" in self.models:
            return self.models["emotion"]
        path = self.paths["emotion"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        classes = list(checkpoint["classes"])
        model = build_emotion_model(checkpoint.get("arch", "resnet18"), len(classes))
        model.load_state_dict(checkpoint["model"])
        model.to(self.torch_device).eval()
        self.models["emotion"] = model
        self.metadata["emotion"] = {
            "classes": classes,
            "image_size": int(checkpoint.get("image_size", 224)),
            "channels": int(checkpoint.get("channels", 3)),
        }
        return model

    def _load_lane(self) -> Any:
        if "lane" in self.models:
            return self.models["lane"]
        path = self.paths["lane"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_lane_model(checkpoint.get("architecture", "compact"))
        model.load_state_dict(checkpoint["model"])
        model.to(self.torch_device).eval()
        self.models["lane"] = model
        self.metadata["lane"] = {
            "size": tuple(checkpoint.get("size", (288, 512))),
            "threshold": float(checkpoint.get("threshold", 0.2)),
            "crop_top": float(checkpoint.get("crop_top", 0.0)),
        }
        return model

    def preload(self, enabled: dict[str, bool]) -> None:
        for key, active in enabled.items():
            if not active:
                continue
            if key == "vehicle":
                self._load_vehicle()
            elif key == "traffic_sign":
                self._load_yolo(key)
            elif key == "emotion":
                self._load_emotion()
            elif key == "lane":
                self._load_lane()

    @staticmethod
    def _names(result: Any, class_id: int) -> str:
        names = result.names
        return str(names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, class_id))

    def _draw_yolo(self, frame: np.ndarray, result: Any, color: tuple[int, int, int], prefix: str) -> int:
        count = 0
        if result.boxes is None:
            return count
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{prefix} {self._names(result, int(class_id))} {float(confidence):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            count += 1
        return count

    def _vehicle(self, frame: np.ndarray) -> int:
        model = self._load_vehicle()
        result = model.predict(
            frame,
            conf=0.18,
            iou=0.5,
            imgsz=960,
            classes=list(COCO_VEHICLE_CLASSES),
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        raw_boxes: list[tuple[list[float], float, str]] = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(float, box.tolist())
                label = COCO_VEHICLE_CLASSES.get(int(class_id), self._names(result, int(class_id)))
                raw_boxes.append(([x1, y1, x2 - x1, y2 - y1], float(confidence), label))
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(confidence), label))

        # Raw detections are intentionally visible immediately, before a
        # tracker has a chance to confirm them.
        for (x, y, width, height), confidence, label in raw_boxes:
            left, top = int(x), int(y)
            right, bottom = int(x + width), int(y + height)
            cv2.rectangle(frame, (left, top), (right, bottom), (56, 210, 170), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (left, max(22, top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (56, 230, 190),
                1,
                cv2.LINE_AA,
            )

        tracks = self.trackers["vehicle"].update_tracks(detections, frame=frame)
        self.last_vehicle_tracks = 0
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            left, top, right, bottom = map(int, track.to_ltrb())
            label = f"vehicle {track.get_det_class() or 'object'}  #{track.track_id}"
            cv2.rectangle(frame, (left, top), (right, bottom), (56, 210, 170), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (56, 230, 190), 2, cv2.LINE_AA)
            self.last_vehicle_tracks += 1
        return len(raw_boxes)

    @staticmethod
    def _is_reasonable_sign_box(box: np.ndarray, confidence: float, frame_shape: tuple[int, ...]) -> bool:
        """Reject crop-like full-frame false positives from the GTSRB model."""

        height, width = frame_shape[:2]
        x1, y1, x2, y2 = map(float, box.tolist())
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_ratio = (box_width * box_height) / max(float(width * height), 1.0)
        width_ratio = box_width / max(float(width), 1.0)
        height_ratio = box_height / max(float(height), 1.0)
        aspect_ratio = box_width / max(box_height, 1.0)
        return (
            confidence >= 0.45
            and area_ratio <= 0.22
            and width_ratio <= 0.72
            and height_ratio <= 0.72
            and 0.18 <= aspect_ratio <= 5.5
        )

    @staticmethod
    def _draw_red_circular_signs(
        frame: np.ndarray,
        existing_boxes: list[np.ndarray],
    ) -> int:
        """Find clear red circular road-sign candidates missed by the crop model."""

        height, width = frame.shape[:2]
        upper_frame = frame[: int(height * 0.84)]
        hsv = cv2.cvtColor(upper_frame, cv2.COLOR_BGR2HSV)
        red_low = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(red_low | red_high, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def overlaps_existing(candidate: tuple[int, int, int, int]) -> bool:
            cx1, cy1, cx2, cy2 = candidate
            candidate_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            for box in existing_boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                ix1, iy1 = max(cx1, x1), max(cy1, y1)
                ix2, iy2 = min(cx2, x2), min(cy2, y2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if intersection / candidate_area >= 0.3:
                    return True
            return False

        count = 0
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 300 or perimeter <= 0:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if not 24 <= box_width <= min(260, width // 2) or not 24 <= box_height <= min(260, height // 2):
                continue
            aspect_ratio = box_width / max(float(box_height), 1.0)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if not 0.72 <= aspect_ratio <= 1.35 or circularity < 0.55:
                continue
            padding = max(6, int(max(box_width, box_height) * 0.16))
            candidate = (
                max(0, x - padding),
                max(0, y - padding),
                min(width, x + box_width + padding),
                min(height, y + box_height + padding),
            )
            if overlaps_existing(candidate):
                continue
            x1, y1, x2, y2 = candidate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (54, 166, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"sign red-circle {circularity:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (54, 166, 255),
                2,
                cv2.LINE_AA,
            )
            count += 1
        return count

    def _traffic_sign(self, frame: np.ndarray) -> int:
        model = self._load_yolo("traffic_sign")
        result = model.predict(frame, conf=0.45, iou=0.5, imgsz=960, device=self.device, verbose=False)[0]
        count = 0
        accepted_boxes: list[np.ndarray] = []
        if result.boxes is None:
            return self._draw_red_circular_signs(frame, accepted_boxes)
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            confidence = float(confidence)
            if not self._is_reasonable_sign_box(box, confidence, frame.shape):
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"sign {self._names(result, int(class_id))} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (54, 166, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (54, 166, 255), 2, cv2.LINE_AA)
            accepted_boxes.append(box)
            count += 1
        return count + self._draw_red_circular_signs(frame, accepted_boxes)

    def _emotion_tensor(self, crop: np.ndarray) -> torch.Tensor:
        meta = self.metadata["emotion"]
        size = meta["image_size"]
        channels = meta["channels"]
        image = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        if channels == 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            return torch.from_numpy(gray).float().div(255).unsqueeze(0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        tensor = torch.from_numpy(image).float().div(255).permute(2, 0, 1)
        return (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _emotion(self, frame: np.ndarray) -> int:
        model = self._load_emotion()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return 0
        tensors = [self._emotion_tensor(frame[y : y + h, x : x + w]) for x, y, w, h in faces]
        with torch.no_grad():
            logits = model(torch.stack(tensors).to(self.torch_device))
            probabilities = logits.softmax(1).cpu().numpy()
        classes = self.metadata["emotion"]["classes"]
        for (x, y, w, h), scores in zip(faces, probabilities):
            index = int(scores.argmax())
            color = (232, 112, 136)
            label = f"{classes[index]}  {float(scores[index]):.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return len(faces)

    def _lane(self, frame: np.ndarray) -> int:
        model = self._load_lane()
        meta = self.metadata["lane"]
        height, width = meta["size"]
        crop_top = int(round(frame.shape[0] * meta["crop_top"]))
        cropped = frame[crop_top:, :] if crop_top else frame
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            probability = model(tensor.to(self.torch_device)).sigmoid()[0, 0].cpu().numpy()
        small_mask = (probability >= meta["threshold"]).astype(np.uint8)
        mask = cv2.resize(small_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = np.zeros_like(frame)
        overlay[mask] = (38, 205, 120)
        frame[:] = cv2.addWeighted(frame, 0.78, overlay, 0.48, 0)

        anchors = 0
        # Render stable horizontal anchor points for the visible lane centerlines.
        for y in np.linspace(int(frame.shape[0] * 0.44), int(frame.shape[0] * 0.96), 10).astype(int):
            xs = np.flatnonzero(mask[y])
            if len(xs) == 0:
                continue
            groups = np.split(xs, np.where(np.diff(xs) > 8)[0] + 1)
            for group in groups:
                if len(group) >= 2:
                    x = int((int(group[0]) + int(group[-1])) / 2)
                    cv2.circle(frame, (x, int(y)), 4, (66, 248, 232), -1, cv2.LINE_AA)
                    anchors += 1
        return anchors

    def process(self, frame: np.ndarray, enabled: dict[str, bool]) -> tuple[np.ndarray, dict[str, Any]]:
        start = time.perf_counter()
        stats = {"vehicle": 0, "vehicle_tracks": 0, "traffic_sign": 0, "emotion": 0, "lane": 0, "loaded": 0}
        if enabled.get("lane"):
            stats["lane"] = self._lane(frame)
        if enabled.get("vehicle"):
            stats["vehicle"] = self._vehicle(frame)
            stats["vehicle_tracks"] = self.last_vehicle_tracks
        if enabled.get("traffic_sign"):
            stats["traffic_sign"] = self._traffic_sign(frame)
        if enabled.get("emotion"):
            stats["emotion"] = self._emotion(frame)
        stats["loaded"] = len(self.models)
        stats["latency_ms"] = (time.perf_counter() - start) * 1000.0
        return frame, stats
