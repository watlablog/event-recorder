from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path


def _preload_model_runtime() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"PyTorch failed to initialize: {exc}") from exc


# Load PyTorch before PyQt so Windows resolves PyTorch's DLL dependencies first.
_preload_model_runtime()

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
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
from event_recorder.audio import MicrophoneCandidate, discover_microphones
from event_recorder.camera_discovery import CameraCandidate, discover_cameras
from event_recorder.config import AppConfig, ConfigError, load_config
from event_recorder.engine import EngineCallbacks, EngineFrame, RecorderEngine
from event_recorder.exclusion import (
    Polygon,
    filter_detection_result_by_exclusion,
    map_display_point_to_frame,
    map_display_point_to_frame_clamped,
    nearest_polygon_vertex,
)
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
VERTEX_PICK_RADIUS_PIXELS = 14.0


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

    def __init__(
        self,
        config: AppConfig,
        stop_event: threading.Event,
        exclusion_polygon: Polygon = (),
    ) -> None:
        super().__init__()
        self.config = config
        self.stop_event = stop_event
        self.exclusion_polygon = exclusion_polygon

    @pyqtSlot()
    def run(self) -> None:
        code = 2
        detection_filter = None
        if self.exclusion_polygon:
            detection_filter = lambda result: filter_detection_result_by_exclusion(
                result, self.exclusion_polygon
            )
        callbacks = EngineCallbacks(
            on_frame=self._on_frame,
            on_status=self._on_status,
            on_error=self.failed.emit,
            detection_filter=detection_filter,
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


class ExclusionPolygonCanvas(QLabel):
    polygon_changed = pyqtSignal()

    def __init__(self, frame, polygon: Polygon, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = frame.copy()
        self._polygon: list[tuple[float, float]] = list(polygon)
        self._drag_vertex_index: int | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 450)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(PREVIEW_BASE_STYLE)
        self.setMouseTracking(True)
        self._refresh()

    def polygon(self) -> Polygon:
        if len(self._polygon) < 3:
            return ()
        return tuple(self._polygon)

    def undo(self) -> None:
        if self._polygon:
            self._polygon.pop()
            self._refresh()
            self.polygon_changed.emit()

    def clear(self) -> None:
        if self._polygon:
            self._polygon.clear()
            self._refresh()
            self.polygon_changed.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = self._event_frame_point(event)
        if point is None:
            return
        vertex_index = nearest_polygon_vertex(
            point,
            tuple(self._polygon),
            self._frame_pick_radius(),
        )
        if vertex_index is not None:
            self._drag_vertex_index = vertex_index
            self.setCursor(Qt.ClosedHandCursor)
            return
        self._polygon.append(point)
        self._refresh()
        self.polygon_changed.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_vertex_index is None:
            point = self._event_frame_point(event)
            if point is not None and nearest_polygon_vertex(
                point,
                tuple(self._polygon),
                self._frame_pick_radius(),
            ) is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            return
        point = self._event_frame_point(event, clamp=True)
        if point is None:
            return
        self._polygon[self._drag_vertex_index] = point
        self._refresh()
        self.polygon_changed.emit()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_vertex_index is not None:
            self._drag_vertex_index = None
            self.unsetCursor()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _event_frame_point(self, event, clamp: bool = False) -> tuple[float, float] | None:
        height, width = self._frame.shape[:2]
        mapper = (
            map_display_point_to_frame_clamped if clamp else map_display_point_to_frame
        )
        return mapper(
            (float(event.x()), float(event.y())),
            (self.width(), self.height()),
            (width, height),
        )

    def _frame_pick_radius(self) -> float:
        height, width = self._frame.shape[:2]
        scale = min(self.width() / width, self.height() / height)
        if scale <= 0:
            return VERTEX_PICK_RADIUS_PIXELS
        return VERTEX_PICK_RADIUS_PIXELS / scale

    def _refresh(self) -> None:
        try:
            import cv2
        except ImportError:
            return

        display = self._frame.copy()
        if self._polygon:
            points = _polygon_to_cv_points(self._polygon)
            if len(points) >= 3:
                overlay = display.copy()
                cv2.fillPoly(overlay, [points], (0, 0, 255))
                cv2.addWeighted(overlay, 0.22, display, 0.78, 0, display)
            if len(points) >= 2:
                cv2.polylines(
                    display,
                    [points],
                    len(points) >= 3,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            for x, y in points:
                cv2.circle(display, (int(x), int(y)), 5, (255, 255, 255), -1)
                cv2.circle(display, (int(x), int(y)), 5, (0, 0, 255), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class ExclusionPolygonDialog(QDialog):
    def __init__(
        self,
        frame,
        polygon: Polygon,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Exclusion Area")
        self._canvas = ExclusionPolygonCanvas(frame, polygon, self)

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self._canvas.undo)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._canvas.clear)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(undo_button)
        buttons.addWidget(clear_button)
        buttons.addStretch(1)
        buttons.addWidget(apply_button)
        buttons.addWidget(cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.resize(960, 620)

    def polygon(self) -> Polygon:
        return self._canvas.polygon()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.available_classes: tuple[DetectionClass, ...] = ()
        self.selected_classes: tuple[str, ...] = DEFAULT_TARGET_CLASSES
        self.cameras: list[CameraCandidate] = []
        self.microphones: list[MicrophoneCandidate] = []
        self._model_thread: QThread | None = None
        self._model_worker: ModelClassesWorker | None = None
        self._recorder_thread: QThread | None = None
        self._recorder_worker: RecorderWorker | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._last_frame: EngineFrame | None = None
        self.exclusion_polygon: Polygon = ()

        self.setWindowTitle("Event Recorder")
        self._build_ui()
        self._refresh_devices()
        self._load_model_classes()
        self._update_controls()

    def _build_ui(self) -> None:
        self.preview_label = QLabel("No preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 450)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(PREVIEW_BASE_STYLE)

        self.camera_combo = QComboBox()
        self.audio_checkbox = QCheckBox("Enable Audio")
        self.audio_checkbox.setChecked(self.config.audio.enabled)
        self.audio_checkbox.stateChanged.connect(lambda _state: self._update_controls())
        self.record_boxes_checkbox = QCheckBox("Record Boxes")
        self.record_boxes_checkbox.setChecked(self.config.recording.draw_boxes)
        self.microphone_combo = QComboBox()
        self.objects_button = QPushButton("Objects to Detect")
        self.objects_button.clicked.connect(self._open_objects_dialog)
        self.exclusion_button = QPushButton("Set Exclusion Area")
        self.exclusion_button.clicked.connect(self._open_exclusion_dialog)
        self.show_exclusion_checkbox = QCheckBox("Show Exclusion Area")
        self.show_exclusion_checkbox.stateChanged.connect(
            lambda _state: self._rerender_last_frame()
        )
        self.rec_button = QPushButton("Rec")
        self.rec_button.clicked.connect(self._toggle_recording)

        device_controls = QHBoxLayout()
        device_controls.addWidget(QLabel("Camera"))
        device_controls.addWidget(self.camera_combo, stretch=1)
        device_controls.addWidget(self.audio_checkbox)
        device_controls.addWidget(QLabel("Microphone"))
        device_controls.addWidget(self.microphone_combo, stretch=1)

        recording_controls = QHBoxLayout()
        recording_controls.addWidget(self.objects_button)
        recording_controls.addWidget(self.exclusion_button)
        recording_controls.addWidget(self.show_exclusion_checkbox)
        recording_controls.addWidget(self.record_boxes_checkbox)
        recording_controls.addStretch(1)
        recording_controls.addWidget(self.rec_button)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, stretch=1)
        layout.addLayout(device_controls)
        layout.addLayout(recording_controls)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Loading")

        refresh_action = QAction("Refresh Devices", self)
        refresh_action.triggered.connect(self._refresh_devices)
        self.menuBar().addAction(refresh_action)

    def _refresh_devices(self) -> None:
        self._discover_cameras()
        self._discover_microphones()
        self._update_controls()

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

    def _discover_microphones(self) -> None:
        self.microphone_combo.clear()
        try:
            self.microphones = discover_microphones()
        except Exception as exc:
            self.microphones = []
            self.microphone_combo.addItem("No microphone found", None)
            self.statusBar().showMessage(f"Microphone scan failed: {exc}")
            self._update_controls()
            return

        for microphone in self.microphones:
            self.microphone_combo.addItem(microphone.label, microphone.index)
        if not self.microphones:
            self.microphone_combo.addItem("No microphone found", None)
            return

        if self.config.audio.device is not None:
            target = self.config.audio.device
            for row in range(self.microphone_combo.count()):
                if self.microphone_combo.itemData(row) == target:
                    self.microphone_combo.setCurrentIndex(row)
                    break

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

    def _open_exclusion_dialog(self) -> None:
        frame = self._capture_exclusion_frame()
        if frame is None:
            self.statusBar().showMessage("Could not capture a frame for exclusion setup")
            return

        dialog = ExclusionPolygonDialog(frame, self.exclusion_polygon, self)
        if dialog.exec_() == QDialog.Accepted:
            self.exclusion_polygon = dialog.polygon()
            if not self.exclusion_polygon:
                self.show_exclusion_checkbox.setChecked(False)
            self._update_controls()
            self._rerender_last_frame()

    def _capture_exclusion_frame(self):
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            return None
        try:
            import cv2
        except ImportError:
            self.statusBar().showMessage("opencv-python is not installed")
            return None

        capture = cv2.VideoCapture(int(camera_index))
        try:
            if not capture.isOpened():
                return None
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.camera.requested_fps)
            for _ in range(8):
                ok, frame = capture.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    return frame
            return None
        finally:
            capture.release()

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
        audio_enabled = self.audio_checkbox.isChecked()
        audio_device = self.microphone_combo.currentData() if audio_enabled else None
        if audio_enabled and audio_device is None:
            self.statusBar().showMessage("Select a microphone or disable audio")
            return

        runtime_config = with_runtime_selection(
            self.config,
            camera_source=int(camera_index),
            target_classes=self.selected_classes,
            audio_enabled=audio_enabled,
            audio_device=audio_device,
            recording_draw_boxes=self.record_boxes_checkbox.isChecked(),
        )
        self._stop_event = threading.Event()
        self._recorder_thread = QThread(self)
        self._recorder_worker = RecorderWorker(
            runtime_config,
            self._stop_event,
            self.exclusion_polygon,
        )
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
        if self.show_exclusion_checkbox.isChecked() and self.exclusion_polygon:
            _draw_exclusion_polygon(cv2, display, self.exclusion_polygon)
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

    def _rerender_last_frame(self) -> None:
        if self._last_frame is not None:
            self._render_frame(self._last_frame)

    def _update_controls(self) -> None:
        has_camera = bool(self.cameras)
        has_microphone = bool(self.microphones)
        has_classes = bool(self.available_classes)
        has_selection = bool(self.selected_classes)
        audio_enabled = self.audio_checkbox.isChecked()
        self.camera_combo.setEnabled(not self._running and has_camera)
        self.audio_checkbox.setEnabled(not self._running)
        self.record_boxes_checkbox.setEnabled(not self._running)
        self.microphone_combo.setEnabled(
            not self._running and audio_enabled and has_microphone
        )
        self.objects_button.setEnabled(not self._running and has_classes)
        self.exclusion_button.setEnabled(not self._running and has_camera)
        self.show_exclusion_checkbox.setEnabled(bool(self.exclusion_polygon))
        self.rec_button.setText("Stop" if self._running else "Rec")
        self.rec_button.setEnabled(
            self._running
            or (
                has_camera
                and has_classes
                and has_selection
                and (not audio_enabled or has_microphone)
            )
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


def _polygon_to_cv_points(polygon: list[tuple[float, float]] | Polygon):
    import numpy as np

    return np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon],
        dtype=np.int32,
    )


def _draw_exclusion_polygon(cv2, frame, polygon: Polygon) -> None:
    if len(polygon) < 3:
        return
    points = _polygon_to_cv_points(polygon)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 0, 255))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [points], True, (0, 0, 255), 3, cv2.LINE_AA)


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
    try:
        _preload_model_runtime()
    except RuntimeError as exc:
        QMessageBox.critical(None, "Model Runtime Error", str(exc))
        return 2

    window = MainWindow(config)
    window.resize(1024, 680)
    window.show()
    return app.exec_()
