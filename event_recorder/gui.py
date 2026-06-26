from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from event_recorder.__main__ import default_config_path, resolve_config_path
from event_recorder.camera_discovery import CameraCandidate, discover_cameras
from event_recorder.config import AppConfig, ConfigError, load_config
from event_recorder.engine import EngineCallbacks, EngineFrame, RecorderEngine
from event_recorder.logging_utils import configure_logging
from event_recorder.model_metadata import (
    DEFAULT_TARGET_CLASSES,
    DetectionClass,
    default_selected_classes,
    load_model_classes,
)
from event_recorder.models import RecorderStatus
from event_recorder.runtime_config import with_runtime_selection

PREVIEW_BASE_STYLE = "background: #111; color: #ddd; border: 6px solid transparent;"
PREVIEW_RECORDING_STYLE = "background: #111; color: #ddd; border: 6px solid #d71920;"


class ModelClassesWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(load_model_classes(self.model_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class RecorderWorker(QObject):
    frame = pyqtSignal(object)
    status = pyqtSignal(str, str)
    failed = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, config: AppConfig, stop_event: threading.Event) -> None:
        super().__init__()
        self.config = config
        self.stop_event = stop_event

    @pyqtSlot()
    def run(self) -> None:
        code = 2
        callbacks = EngineCallbacks(
            on_frame=self._on_frame,
            on_status=self._on_status,
            on_error=self.failed.emit,
        )
        try:
            code = RecorderEngine(self.config, self.stop_event, callbacks).run()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit(code)

    def _on_frame(self, frame: EngineFrame) -> bool:
        self.frame.emit(frame)
        return False

    def _on_status(self, status: RecorderStatus, message: str) -> None:
        self.status.emit(status.value, message)


class ObjectsDialog(QDialog):
    def __init__(
        self,
        classes: tuple[DetectionClass, ...],
        selected_classes: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Objects to Detect")
        self._list = QListWidget(self)
        selected = set(selected_classes)
        for item in classes:
            list_item = QListWidgetItem(item.name)
            list_item.setData(Qt.UserRole, item.name)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(
                Qt.Checked if item.name in selected else Qt.Unchecked
            )
            self._list.addItem(list_item)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self._list)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.resize(360, 520)

    def selected_classes(self) -> tuple[str, ...]:
        selected: list[str] = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(str(item.data(Qt.UserRole)))
        return tuple(selected)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.available_classes: tuple[DetectionClass, ...] = ()
        self.selected_classes: tuple[str, ...] = DEFAULT_TARGET_CLASSES
        self.cameras: list[CameraCandidate] = []
        self._model_thread: QThread | None = None
        self._model_worker: ModelClassesWorker | None = None
        self._recorder_thread: QThread | None = None
        self._recorder_worker: RecorderWorker | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._last_frame: EngineFrame | None = None

        self.setWindowTitle("Event Recorder")
        self._build_ui()
        self._discover_cameras()
        self._load_model_classes()
        self._update_controls()

    def _build_ui(self) -> None:
        self.preview_label = QLabel("No preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 450)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(PREVIEW_BASE_STYLE)

        self.camera_combo = QComboBox()
        self.objects_button = QPushButton("Objects to Detect")
        self.objects_button.clicked.connect(self._open_objects_dialog)
        self.rec_button = QPushButton("Rec")
        self.rec_button.clicked.connect(self._toggle_recording)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camera"))
        controls.addWidget(self.camera_combo, stretch=1)
        controls.addWidget(self.objects_button)
        controls.addWidget(self.rec_button)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, stretch=1)
        layout.addLayout(controls)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Loading")

        refresh_action = QAction("Refresh Cameras", self)
        refresh_action.triggered.connect(self._discover_cameras)
        self.menuBar().addAction(refresh_action)

    def _discover_cameras(self) -> None:
        self.statusBar().showMessage("Scanning cameras")
        self.camera_combo.clear()
        try:
            self.cameras = discover_cameras()
        except Exception as exc:
            self.cameras = []
            self.statusBar().showMessage(f"Camera scan failed: {exc}")
            self._update_controls()
            return

        for camera in self.cameras:
            self.camera_combo.addItem(camera.label, camera.index)
        if not self.cameras:
            self.camera_combo.addItem("No camera found", None)
            self.statusBar().showMessage("No camera found")
        else:
            self.statusBar().showMessage("Loading model classes")
        self._update_controls()

    def _load_model_classes(self) -> None:
        self._model_thread = QThread(self)
        self._model_worker = ModelClassesWorker(self.config.model.path)
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._on_model_classes_loaded)
        self._model_worker.failed.connect(self._on_model_classes_failed)
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.failed.connect(self._model_thread.quit)
        self._model_thread.finished.connect(self._model_worker.deleteLater)
        self._model_thread.finished.connect(self._model_thread.deleteLater)
        self._model_thread.start()

    def _on_model_classes_loaded(self, classes: object) -> None:
        self.available_classes = tuple(classes)
        defaults = default_selected_classes(self.available_classes)
        self.selected_classes = defaults or tuple(
            item
            for item in self.selected_classes
            if any(candidate.name == item for candidate in self.available_classes)
        )
        self.statusBar().showMessage("Ready")
        self._update_controls()

    def _on_model_classes_failed(self, message: str) -> None:
        self.available_classes = ()
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Model Error", message)
        self._update_controls()

    def _open_objects_dialog(self) -> None:
        dialog = ObjectsDialog(self.available_classes, self.selected_classes, self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_classes = dialog.selected_classes()
            self._update_controls()

    def _toggle_recording(self) -> None:
        if self._running:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            self.statusBar().showMessage("Select a camera")
            return
        if not self.selected_classes:
            self.statusBar().showMessage("Select at least one object")
            return

        runtime_config = with_runtime_selection(
            self.config,
            camera_source=int(camera_index),
            target_classes=self.selected_classes,
        )
        self._stop_event = threading.Event()
        self._recorder_thread = QThread(self)
        self._recorder_worker = RecorderWorker(runtime_config, self._stop_event)
        self._recorder_worker.moveToThread(self._recorder_thread)
        self._recorder_thread.started.connect(self._recorder_worker.run)
        self._recorder_worker.frame.connect(self._on_engine_frame)
        self._recorder_worker.status.connect(self._on_engine_status)
        self._recorder_worker.failed.connect(self._on_engine_error)
        self._recorder_worker.finished.connect(self._on_engine_finished)
        self._recorder_worker.finished.connect(self._recorder_thread.quit)
        self._recorder_worker.finished.connect(self._recorder_worker.deleteLater)
        self._recorder_thread.finished.connect(self._recorder_thread.deleteLater)
        self._running = True
        self.statusBar().showMessage("Waiting")
        self._update_controls()
        self._recorder_thread.start()

    def _stop_recording(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self.statusBar().showMessage("Stopping")
        self.rec_button.setEnabled(False)

    def _on_engine_frame(self, frame: object) -> None:
        engine_frame = frame
        self._last_frame = engine_frame
        self._set_preview_recording(engine_frame.status == RecorderStatus.RECORDING)
        self._render_frame(engine_frame)
        self.statusBar().showMessage(_frame_status_text(engine_frame))

    def _on_engine_status(self, status: str, message: str) -> None:
        text = _status_label(status)
        if message:
            text = f"{text}: {message}"
        self.statusBar().showMessage(text)

    def _on_engine_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Error: {message}")

    def _on_engine_finished(self, code: int) -> None:
        self._running = False
        self._recorder_worker = None
        self._recorder_thread = None
        self._stop_event = None
        self._set_preview_recording(False)
        if code == 0:
            self.statusBar().showMessage("Ready")
        else:
            self.statusBar().showMessage("Error")
        self._update_controls()

    def _render_frame(self, frame: EngineFrame) -> None:
        try:
            import cv2
        except ImportError:
            return

        display = frame.packet.frame.copy()
        for detected in frame.detections:
            x1, y1, x2, y2 = (int(value) for value in detected.xyxy)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{detected.class_name} {detected.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def _update_controls(self) -> None:
        has_camera = bool(self.cameras)
        has_classes = bool(self.available_classes)
        has_selection = bool(self.selected_classes)
        self.camera_combo.setEnabled(not self._running and has_camera)
        self.objects_button.setEnabled(not self._running and has_classes)
        self.rec_button.setText("Stop" if self._running else "Rec")
        self.rec_button.setEnabled(
            self._running or (has_camera and has_classes and has_selection)
        )

    def _set_preview_recording(self, recording: bool) -> None:
        self.preview_label.setStyleSheet(
            PREVIEW_RECORDING_STYLE if recording else PREVIEW_BASE_STYLE
        )

    def closeEvent(self, event) -> None:
        if self._running and self._stop_event is not None:
            self._stop_event.set()
            if self._recorder_thread is not None:
                self._recorder_thread.wait(5000)
        event.accept()


def _frame_status_text(frame: EngineFrame) -> str:
    parts = [_status_label(frame.status.value)]
    if frame.recording_elapsed is not None:
        parts.append(f"{frame.recording_elapsed:.1f}s")
    if frame.camera_fps is not None:
        parts.append(f"camera {frame.camera_fps:.1f} fps")
    if frame.detector_fps is not None:
        parts.append(f"detector {frame.detector_fps:.1f} fps")
    return " | ".join(parts)


def _status_label(status: str) -> str:
    labels = {
        RecorderStatus.IDLE.value: "Ready",
        RecorderStatus.WAITING.value: "Waiting",
        RecorderStatus.RECORDING.value: "Recording",
        RecorderStatus.DEGRADED.value: "Degraded",
    }
    return labels.get(status, status.title())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the event recorder GUI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    try:
        config = load_config(resolve_config_path(args.config or default_config_path()))
    except ConfigError as exc:
        QMessageBox.critical(None, "Configuration Error", str(exc))
        return 2

    configure_logging(config.logging.level)
    window = MainWindow(config)
    window.resize(1024, 680)
    window.show()
    return app.exec_()
