#!/usr/bin/env python3
"""PySide6 desktop UI for the four driving-vision models."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .inference import FEATURES, InferenceManager
except ImportError:
    from inference import FEATURES, InferenceManager

ROOT = Path(__file__).resolve().parents[1]


class VideoWorker(QThread):
    frame_ready = Signal(object, dict)
    status_changed = Signal(str)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, source: str, device: str, enabled: dict[str, bool]) -> None:
        super().__init__()
        self.source = source
        self.device = device
        self.enabled = enabled

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.failed.emit(f"无法打开视频：{self.source}")
            return
        manager = InferenceManager(self.device)
        try:
            self.status_changed.emit("正在加载模型，请稍候...")
            manager.preload(self.enabled)
            self.status_changed.emit(f"模型已加载：{manager.device.upper()}")
            frame_count = 0
            started = time.perf_counter()
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, stats = manager.process(frame, self.enabled)
                frame_count += 1
                elapsed = max(time.perf_counter() - started, 1e-6)
                stats["fps"] = frame_count / elapsed
                stats["frame"] = frame_count
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    rgb.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image, stats)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"推理失败：{type(exc).__name__}: {exc}")
        finally:
            capture.release()


class StatTile(QFrame):
    def __init__(self, title: str, value: str = "--") -> None:
        super().__init__()
        self.setObjectName("statTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("statTitle")
        self.value = QLabel(value)
        self.value.setObjectName("statValue")
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: VideoWorker | None = None
        self.video_path = ""
        self.setWindowTitle("大唐道路 · 智能驾驶视觉工作台")
        self.setMinimumSize(1180, 760)
        self.resize(1440, 900)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._update_device_status()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._refresh_clock)
        self.clock.start(1000)
        self._refresh_clock()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(14)

        header = QHBoxLayout()
        brand = QVBoxLayout()
        brand.setSpacing(2)
        title = QLabel("大唐道路")
        title.setObjectName("brandTitle")
        subtitle = QLabel("DRIVING VISION CONTROL CENTER")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        self.runtime_label = QLabel("SMART ENVIRONMENT")
        self.runtime_label.setObjectName("runtimeBadge")
        header.addWidget(self.runtime_label)
        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")
        header.addWidget(self.time_label)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_view())
        splitter.setSizes([330, 1050])
        root.addWidget(splitter, 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("请选择视频文件，然后开始实时检测")

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        section = QLabel("检测配置")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)

        device_label = QLabel("加速设备")
        device_label.setObjectName("fieldLabel")
        layout.addWidget(device_label)
        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU · CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._update_device_status)
        layout.addWidget(self.device_combo)

        feature_label = QLabel("实时功能")
        feature_label.setObjectName("fieldLabel")
        layout.addWidget(feature_label)
        self.checkboxes: dict[str, QCheckBox] = {}
        descriptions = {
            "vehicle": "YOLO26 + DeepSORT · 车辆框与轨迹",
            "traffic_sign": "YOLO26 · 交通标识锚框",
            "emotion": "OpenCV + PyTorch · 驾驶员表情",
            "lane": "Swin V2 · 车道线与锚点",
        }
        for key in ("vehicle", "traffic_sign", "emotion", "lane"):
            box = QCheckBox(FEATURES[key])
            box.setChecked(True)
            box.setToolTip(descriptions[key])
            self.checkboxes[key] = box
            layout.addWidget(box)
            desc = QLabel(descriptions[key])
            desc.setObjectName("hintLabel")
            desc.setWordWrap(True)
            layout.addWidget(desc)

        layout.addSpacing(4)
        file_section = QLabel("视频源")
        file_section.setObjectName("fieldLabel")
        layout.addWidget(file_section)
        self.source_label = QLabel("尚未选择视频")
        self.source_label.setObjectName("sourceLabel")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.open_button = QPushButton("选择视频文件")
        self.open_button.clicked.connect(self._choose_video)
        layout.addWidget(self.open_button)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_detection)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_detection)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        layout.addLayout(action_row)
        layout.addStretch()

        note = QLabel("支持 MP4 / AVI / MOV / MKV\n默认优先使用 CUDA，模型按需加载")
        note.setObjectName("footerNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def _build_view(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        toolbar = QHBoxLayout()
        heading = QLabel("实时检测画面")
        heading.setObjectName("viewTitle")
        toolbar.addWidget(heading)
        toolbar.addStretch()
        self.source_state = QLabel("待机")
        self.source_state.setObjectName("stateBadge")
        toolbar.addWidget(self.source_state)
        layout.addLayout(toolbar)

        self.video_label = QLabel("选择视频后，检测画面将在这里显示")
        self.video_label.setObjectName("videoCanvas")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(680, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label, 1)

        stats = QGridLayout()
        stats.setHorizontalSpacing(10)
        self.fps_tile = StatTile("实时 FPS")
        self.object_tile = StatTile("车辆 / 轨迹")
        self.sign_tile = StatTile("交通标识")
        self.face_tile = StatTile("人脸 / 表情")
        self.anchor_tile = StatTile("车道锚点")
        for index, tile in enumerate((self.fps_tile, self.object_tile, self.sign_tile, self.face_tile, self.anchor_tile)):
            stats.addWidget(tile, 0, index)
        layout.addLayout(stats)
        return panel

    def _update_device_status(self) -> None:
        using_gpu = self.device_combo.currentData() == "cuda" and torch.cuda.is_available()
        if using_gpu:
            self.runtime_label.setText("SMART · CUDA READY")
            self.statusBar().showMessage("默认使用 GPU 加速")
        else:
            self.runtime_label.setText("SMART · CPU MODE")
            if self.device_combo.currentData() == "cuda":
                self.device_combo.setCurrentIndex(1)
            self.statusBar().showMessage("未检测到 CUDA，当前使用 CPU")

    def _refresh_clock(self) -> None:
        self.time_label.setText(time.strftime("%Y-%m-%d  %H:%M:%S"))

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", str(ROOT), "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if not path:
            return
        self.video_path = path
        self.source_label.setText(Path(path).name)
        self.source_label.setToolTip(path)
        self.source_state.setText("已选择")
        self.start_button.setEnabled(True)
        self.statusBar().showMessage(f"已选择视频：{path}")

    def _enabled(self) -> dict[str, bool]:
        return {key: box.isChecked() for key, box in self.checkboxes.items()}

    def _start_detection(self) -> None:
        if not self.video_path:
            return
        if not any(self._enabled().values()):
            QMessageBox.warning(self, "没有启用功能", "请至少选择一个检测功能。")
            return
        self._stop_detection()
        device = str(self.device_combo.currentData())
        self.worker = VideoWorker(self.video_path, device, self._enabled())
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_ok.connect(self._on_finished)
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.source_state.setText("检测中")
        self.worker.start()

    def _stop_detection(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        self.worker = None
        if self.video_path:
            self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.source_state.text() == "检测中":
            self.source_state.setText("已停止")

    def _on_frame(self, image: QImage, stats: dict) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.fps_tile.set_value(f"{stats.get('fps', 0.0):.1f}")
        vehicle_count = int(stats.get("vehicle", 0))
        track_count = int(stats.get("vehicle_tracks", 0))
        self.object_tile.set_value(f"{vehicle_count} / {track_count}")
        self.sign_tile.set_value(str(stats.get("traffic_sign", 0)))
        self.face_tile.set_value(str(stats.get("emotion", 0)))
        self.anchor_tile.set_value(str(stats.get("lane", 0)))
        self.statusBar().showMessage(f"第 {stats.get('frame', 0)} 帧 · 推理 {stats.get('latency_ms', 0.0):.0f} ms")

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_failed(self, message: str) -> None:
        self._stop_detection()
        QMessageBox.critical(self, "检测失败", message)
        self.source_state.setText("错误")

    def _on_finished(self) -> None:
        self._stop_detection()
        self.source_state.setText("播放完成")
        self.statusBar().showMessage("视频检测完成")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_detection()
        event.accept()


STYLESHEET = """
QMainWindow, QWidget { background: #f4f7fb; color: #243447; font-family: 'Microsoft YaHei', 'Segoe UI'; font-size: 13px; }
#sidebar { background: #ffffff; border: 1px solid #d8e2ec; border-radius: 14px; }
#brandTitle { color: #14263a; font-size: 28px; font-weight: 700; letter-spacing: 1px; }
#brandSubtitle { color: #71869b; font-size: 10px; letter-spacing: 2px; }
#runtimeBadge, #stateBadge { background: #e8f7f4; color: #0b8f82; border: 1px solid #a2dcd3; border-radius: 7px; padding: 7px 11px; font-size: 11px; font-weight: 600; }
#timeLabel { color: #708298; padding-left: 10px; }
#sectionTitle, #viewTitle { color: #172b40; font-size: 18px; font-weight: 650; }
#fieldLabel { color: #5e7185; font-size: 12px; font-weight: 600; }
#hintLabel { color: #70849a; font-size: 11px; padding-left: 27px; }
#sourceLabel { color: #263b50; background: #f8fafc; border: 1px solid #cbd9e6; border-radius: 7px; padding: 9px; }
#footerNote { color: #70849a; font-size: 11px; line-height: 1.4; }
QComboBox { background: #ffffff; border: 1px solid #b9cad9; border-radius: 7px; padding: 8px 10px; color: #243447; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #ffffff; color: #243447; selection-background-color: #dff4f0; selection-color: #116d65; }
QCheckBox { color: #243447; spacing: 9px; font-size: 13px; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 4px; border: 1px solid #8da7bc; background: #ffffff; }
QCheckBox::indicator:checked { background: #35c9b2; border-color: #22a99a; }
QPushButton { background: #edf3f8; color: #243b50; border: 1px solid #bfd0df; border-radius: 7px; padding: 9px 12px; font-weight: 600; }
QPushButton:hover { background: #e1edf4; }
QPushButton:disabled { color: #9aaaba; border-color: #d8e2ec; }
#primaryButton { background: #1b9b90; border-color: #28b9aa; color: #ffffff; }
#primaryButton:hover { background: #168b81; }
#secondaryButton { background: #e8eef4; }
#videoCanvas { background: #eef3f7; border: 1px solid #c9d7e4; border-radius: 12px; color: #71849a; font-size: 15px; }
#statTile { background: #ffffff; border: 1px solid #d8e2ec; border-radius: 9px; min-height: 62px; }
#statTitle { color: #71849a; font-size: 11px; }
#statValue { color: #172d43; font-size: 20px; font-weight: 700; }
QStatusBar { color: #708298; background: #f4f7fb; border-top: 1px solid #d8e2ec; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Datang Road Vision")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
