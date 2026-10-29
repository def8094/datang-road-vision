#!/usr/bin/env python3
"""Inspect and materialize the renamed road-vision datasets.

The script never changes the source archives. It writes only to
``datasets/processed`` and ``reports``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


def safe_target(root: Path, relative_name: str) -> Path:
    target = (root / relative_name.replace("\\", "/")).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise ValueError(f"Unsafe archive path: {relative_name}")
    return target


def files(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [entry for entry in zf.infolist() if not entry.is_dir()]


def extract_entry(zf: zipfile.ZipFile, entry: zipfile.ZipInfo, output_root: Path, relative_name: str) -> Path:
    target = safe_target(output_root, relative_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(entry) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return target


def stable_val(filename: str, denominator: int = 10) -> bool:
    """Deterministic 10% validation split without leaking ordering assumptions."""
    return int(hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8], 16) % denominator == 0


def write_yaml(path: Path, dataset_root: str, train: str, val: str, test: str, names: list[str]) -> None:
    lines = [f"path: {dataset_root}", f"train: {train}", f"val: {val}", f"test: {test}", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_summary(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        entries = files(zf)
        extensions = Counter(Path(entry.filename).suffix.lower() or "(none)" for entry in entries)
        return {
            "archive": str(path),
            "files": len(entries),
            "compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": sum(entry.file_size for entry in entries),
            "extensions": dict(extensions),
        }


def inspect(source: Path) -> dict:
    emotion = source / "driver_emotion_fer2013" / "fer2013_images.zip"
    lane = source / "lane_tusimple_preprocessed" / "tusimple_preprocessed.zip"
    vehicle = source / "vehicle_detection_3class"
    gtsrb = source / "traffic_sign_gtsrb" / "gtsrb_training_fixed.zip"
    kitti = source / "vehicle_images_unlabeled" / "kitti_object_image_2.zip"
    report = {"source": str(source.resolve()), "archives": [], "assignments": []}
    paths = [emotion, lane, vehicle / "train_images.zip", vehicle / "train_labels.zip", vehicle / "test_images.zip", vehicle / "test_labels.zip", gtsrb, kitti]
    for path in paths:
        item = archive_summary(path)
        with zipfile.ZipFile(path) as zf:
            names = [entry.filename.replace("\\", "/") for entry in files(zf)]
            if path == emotion:
                item["classes"] = {split: dict(Counter(name.split("/")[1] for name in names if name.startswith(split + "/"))) for split in ("train", "test")}
            elif path == lane:
                frames = {name.replace("/frames/", "/lane-masks/") for name in names if "/frames/" in name}
                masks = {name for name in names if "/lane-masks/" in name}
                item.update({"paired": len(frames & masks), "unpaired_frames": len(frames - masks), "unpaired_masks": len(masks - frames)})
            elif path.name in {"train_labels.zip", "test_labels.zip"}:
                counts = Counter()
                malformed = 0
                for name in names:
                    for row in zf.read(name).decode("utf-8", errors="replace").splitlines():
                        fields = row.split()
                        if len(fields) != 5:
                            malformed += 1
                        else:
                            try:
                                counts[int(fields[0])] += 1
                            except ValueError:
                                malformed += 1
                item.update({"label_class_counts": dict(counts), "malformed_label_rows": malformed})
            elif path == gtsrb:
                class_counts = Counter()
                for name in names:
                    if name.startswith("GTSRB/Training/") and name.endswith(".ppm"):
                        class_counts[name.split("/")[2]] += 1
                item.update({"class_count": len(class_counts), "class_image_counts": dict(class_counts)})
            elif path == kitti:
                item["has_label_2"] = any("label_2/" in name for name in names)
        report["archives"].append(item)
    report["assignments"] = [
        {"model": "pytorch_cnn", "task": "driver_emotion_classification", "status": "trainable", "data": "driver_emotion_fer2013", "algorithm": "OpenCV preprocessing + PyTorch CNN"},
        {"model": "yolo26", "task": "traffic_sign_detection", "status": "trainable", "data": "traffic_sign_gtsrb", "algorithm": "YOLO26 detection; GTSRB ROI converted to YOLO boxes"},
        {"model": "yolo26_deepsort", "task": "vehicle_detection_and_tracking", "status": "trainable_and_inference_ready", "data": "vehicle_detection_3class", "algorithm": "YOLO26 detection + DeepSORT tracking"},
        {"model": "swin_v2", "task": "lane_segmentation", "status": "trainable", "data": "lane_tusimple_full", "algorithm": "PyTorch Swin Transformer V2 encoder + segmentation decoder"},
        {"model": "none", "task": "unlabeled_kitti_images", "status": "blocked_missing_labels", "data": "vehicle_images_unlabeled", "algorithm": "not used until label_2 is supplied"},
    ]
    return report


def extract_emotion(archive: Path, output: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for entry in files(zf):
            name = entry.filename.replace("\\", "/")
            if name.startswith(("train/", "test/")) and Path(name).suffix.lower() in IMAGE_SUFFIXES:
                extract_entry(zf, entry, output, name)
                count += 1
    return count


def extract_lane(archive: Path, output: Path) -> int:
    with zipfile.ZipFile(archive) as zf:
        frame_entries = {entry.filename.replace("\\", "/"): entry for entry in files(zf) if "/frames/" in entry.filename}
        mask_entries = {entry.filename.replace("\\", "/"): entry for entry in files(zf) if "/lane-masks/" in entry.filename}
        paired = 0
        for frame_name in sorted(frame_entries):
            mask_name = frame_name.replace("/frames/", "/lane-masks/")
            if mask_name not in mask_entries:
                continue
            prefix, filename = frame_name.split("/frames/", 1)
            unique_name = f"{prefix.replace('/', '__')}__{filename}"
            split = "val" if stable_val(unique_name) else "train"
            extract_entry(zf, frame_entries[frame_name], output, f"images/{split}/{unique_name}")
            extract_entry(zf, mask_entries[mask_name], output, f"masks/{split}/{unique_name}")
            paired += 1
    return paired


def extract_vehicle(source: Path, output: Path) -> int:
    image_archives = [("train_images.zip", "train"), ("test_images.zip", "test")]
    label_archives = {"train": "train_labels.zip", "test": "test_labels.zip"}
    count = 0
    for image_archive_name, source_split in image_archives:
        with zipfile.ZipFile(source / image_archive_name) as iz, zipfile.ZipFile(source / label_archives[source_split]) as lz:
            label_names = {Path(entry.filename).stem: entry for entry in files(lz) if entry.filename.endswith(".txt")}
            for image_entry in files(iz):
                filename = Path(image_entry.filename).name
                stem = Path(filename).stem
                if Path(filename).suffix.lower() not in IMAGE_SUFFIXES or stem not in label_names:
                    continue
                split = source_split if source_split == "test" else ("val" if stable_val(filename) else "train")
                extract_entry(iz, image_entry, output, f"images/{split}/{filename}")
                extract_entry(lz, label_names[stem], output, f"labels/{split}/{stem}.txt")
                count += 1
    write_yaml(output / "data.yaml", str(output.resolve()), "images/train", "images/val", "images/test", ["vehicle_class_0", "vehicle_class_1", "vehicle_class_2"])
    (output / "CLASS_NAMES_UNVERIFIED.txt").write_text("The source labels contain class ids 0, 1, 2 but no legend. Replace these names after confirming the dataset annotation contract.\n", encoding="utf-8")
    return count


def extract_gtsrb_detection(archive: Path, output: Path) -> int:
    """Convert GTSRB crop + ROI metadata into a YOLO26 detection dataset."""
    records: list[tuple[str, bytes, int, float, float, float, float]] = []
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        for entry in files(zf):
            name = entry.filename.replace("\\", "/")
            if not (name.startswith("GTSRB/Training/") and name.endswith(".csv")):
                continue
            class_id = int(name.split("/")[2])
            rows = csv.DictReader(zf.read(name).decode("utf-8").splitlines(), delimiter=";")
            for row in rows:
                image_name = f"GTSRB/Training/{class_id:05d}/{row['Filename']}"
                if image_name not in members:
                    continue
                width, height = float(row["Width"]), float(row["Height"])
                x1, y1 = float(row["Roi.X1"]), float(row["Roi.Y1"])
                x2, y2 = float(row["Roi.X2"]), float(row["Roi.Y2"])
                records.append((f"{class_id:05d}_{row['Filename']}", zf.read(image_name), class_id, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height))
    for filename, raw, class_id, x, y, w, h in records:
        split = "val" if stable_val(filename) else "train"
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode {filename}")
        image_path = output / "images" / split / f"{Path(filename).stem}.jpg"
        label_path = output / "labels" / split / f"{Path(filename).stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Unable to write {image_path}")
        label_path.write_text(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n", encoding="utf-8")
    write_yaml(output / "data.yaml", str(output.resolve()), "images/train", "images/val", "", [f"traffic_sign_{i:02d}" for i in range(43)])
    return len(records)


def materialize(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    return {
        "driver_emotion_images": extract_emotion(source / "driver_emotion_fer2013" / "fer2013_images.zip", output / "driver_emotion_fer2013"),
        "lane_pairs": extract_lane(source / "lane_tusimple_preprocessed" / "tusimple_preprocessed.zip", output / "lane_tusimple_full"),
        "vehicle_image_label_pairs": extract_vehicle(source / "vehicle_detection_3class", output / "vehicle_detection_3class"),
        "traffic_sign_roi_images": extract_gtsrb_detection(source / "traffic_sign_gtsrb" / "gtsrb_training_fixed.zip", output / "traffic_sign_gtsrb"),
    }


def markdown_report(report: dict, extracted: dict | None = None) -> str:
    lines = ["# Dataset Manifest", "", f"Source: `{report['source']}`", ""]
    if extracted:
        lines += ["## Materialized", "", "| Item | Count |", "|---|---:|"]
        lines += [f"| {key} | {value} |" for key, value in extracted.items()]
        lines.append("")
    lines += ["## Archives", "", "| Archive | Files | Extensions |", "|---|---:|---|"]
    for item in report["archives"]:
        extensions = ", ".join(f"{key}: {value}" for key, value in sorted(item["extensions"].items()))
        lines.append(f"| `{Path(item['archive']).name}` | {item['files']} | {extensions} |")
    lines += ["", "## Model Assignment", "", "| Model | Task | Status | Algorithm |", "|---|---|---|---|"]
    for item in report["assignments"]:
        lines.append(f"| `{item['model']}` | {item['task']} | **{item['status']}** | {item['algorithm']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    report = inspect(args.source)
    extracted = materialize(args.source, args.output) if args.extract and not args.report_only else None
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "dataset_manifest.json").write_text(json.dumps({"inspection": report, "extracted": extracted}, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "dataset_manifest.md").write_text(markdown_report(report, extracted), encoding="utf-8")
    print(markdown_report(report, extracted))


if __name__ == "__main__":
    main()
