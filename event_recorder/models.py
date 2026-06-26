from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class StopReason(str, Enum):
    TARGET_ABSENT = "target_absent"
    MAX_CLIP_DURATION = "max_clip_duration"
    USER_SHUTDOWN = "user_shutdown"
    END_OF_FILE = "end_of_file"
    CAMERA_FAILURE = "camera_failure"
    DETECTOR_FAILURE = "detector_failure"


class DetectorHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class RecorderStatus(str, Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    RECORDING = "RECORDING"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    frame_id: int
    captured_at_monotonic: float
    detected: bool
    detections: tuple[DetectedObject, ...]


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_at_monotonic: float
    captured_at_wall_clock: datetime
    frame: np.ndarray


@dataclass(frozen=True)
class EventPaths:
    event_id: str
    part: int
    partial_video_path: Any
    final_video_path: Any
    metadata_path: Any
