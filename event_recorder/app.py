from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from event_recorder.camera import (
    CameraError,
    validate_file_source_exists,
)
from event_recorder.config import AppConfig, ConfigError, load_config
from event_recorder.engine import EngineCallbacks, EngineFrame, RecorderEngine
from event_recorder.logging_utils import configure_logging
from event_recorder.preview import PreviewWindow
from event_recorder.recorder import VideoWriterError

LOGGER = logging.getLogger(__name__)


def run(config_path: Path) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(exc)
        return 2

    configure_logging(config.logging.level)
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    try:
        validate_file_source_exists(config.camera.source)
        return _run_loop(config, stop_event)
    except CameraError as exc:
        LOGGER.error("%s", exc)
        return 2
    except VideoWriterError as exc:
        LOGGER.error("%s", exc)
        return 2


def _run_loop(config: AppConfig, stop_event: threading.Event) -> int:
    preview = PreviewWindow(config.preview)
    try:
        callbacks = EngineCallbacks(on_frame=lambda frame: _show_preview(preview, frame))
        return RecorderEngine(config, stop_event, callbacks).run()
    finally:
        preview.close()


def _show_preview(preview: PreviewWindow, frame: EngineFrame) -> bool:
    return preview.show(
        frame.packet.frame,
        frame.detections,
        frame.status,
        camera_fps=frame.camera_fps,
        detector_fps=frame.detector_fps,
        recording_elapsed=frame.recording_elapsed,
    )


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
