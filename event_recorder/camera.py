from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from event_recorder.config import CameraConfig
from event_recorder.models import FramePacket


class CameraError(RuntimeError):
    pass


class CameraReadError(CameraError):
    pass


class EndOfInput(CameraError):
    pass


class CameraStream:
    def __init__(self, config: CameraConfig, timezone: ZoneInfo | None = None) -> None:
        self.config = config
        self.timezone = timezone
        self.capture = None
        self._cv2 = None

    @property
    def is_file_source(self) -> bool:
        return isinstance(self.config.source, str)

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("opencv-python is not installed.") from exc

        self._cv2 = cv2
        self.release()
        self.capture = cv2.VideoCapture(self.config.source)
        if not self.capture.isOpened():
            raise CameraError(f"Could not open camera source: {self.config.source}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.requested_fps)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def output_fps(self) -> float:
        if self.capture is None or self._cv2 is None:
            return self.config.fallback_fps
        fps = float(self.capture.get(self._cv2.CAP_PROP_FPS))
        if math.isfinite(fps) and 1 <= fps <= 240:
            return fps
        return self.config.fallback_fps

    def read_packet(self, frame_id: int) -> FramePacket:
        if self.capture is None:
            raise CameraReadError("Camera is not open.")
        ok, frame = self.capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            if self.is_file_source:
                raise EndOfInput("End of input file.")
            raise CameraReadError("Failed to read frame from camera.")
        return FramePacket(
            frame_id=frame_id,
            captured_at_monotonic=time.monotonic(),
            captured_at_wall_clock=datetime.now(self.timezone).astimezone(),
            frame=frame,
        )

    def reopen_with_retries(self) -> bool:
        for attempt in range(1, self.config.reconnect_attempts + 1):
            time.sleep(self.config.reconnect_interval_seconds)
            try:
                self.open()
                return True
            except CameraError:
                if attempt >= self.config.reconnect_attempts:
                    return False
        return False


def validate_file_source_exists(source: int | str) -> None:
    if isinstance(source, str) and not Path(source).exists():
        raise CameraError(f"Input video file not found: {source}")
