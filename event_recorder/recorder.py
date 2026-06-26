from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from event_recorder.config import RecordingConfig
from event_recorder.models import DetectionResult, EventPaths, FramePacket, StopReason
from event_recorder.storage import write_metadata


class VideoWriterError(RuntimeError):
    pass


class VideoClipWriter:
    def __init__(
        self,
        paths: EventPaths,
        recording: RecordingConfig,
        source: int | str,
        model_path: str,
        target_classes: tuple[str, ...],
        fps: float,
        frame_size: tuple[int, int],
        started_at_wall_clock: datetime,
        started_at_monotonic: float,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise VideoWriterError("opencv-python is not installed.") from exc

        self.paths = paths
        self.recording = recording
        self.source = source
        self.model_path = model_path
        self.target_classes = target_classes
        self.fps = fps
        self.frame_width, self.frame_height = frame_size
        self.started_at_wall_clock = started_at_wall_clock
        self.started_at_monotonic = started_at_monotonic
        self.frame_count = 0
        self.last_written_frame_id: int | None = None
        self.detected_classes: set[str] = set()
        self.max_confidence_by_class: dict[str, float] = {}
        fourcc = cv2.VideoWriter_fourcc(*recording.fourcc)
        self._writer = cv2.VideoWriter(
            str(paths.partial_video_path),
            fourcc,
            fps,
            frame_size,
        )
        if not self._writer.isOpened():
            self._writer.release()
            _unlink_if_exists(paths.partial_video_path)
            raise VideoWriterError(f"Could not open VideoWriter: {paths.partial_video_path}")

    def write(self, packet: FramePacket) -> None:
        if (
            self.last_written_frame_id is not None
            and packet.frame_id <= self.last_written_frame_id
        ):
            return
        self._writer.write(packet.frame)
        self.frame_count += 1
        self.last_written_frame_id = packet.frame_id

    def observe_detection(self, result: DetectionResult) -> None:
        for detected in result.detections:
            self.detected_classes.add(detected.class_name)
            current = self.max_confidence_by_class.get(detected.class_name, 0.0)
            if detected.confidence > current:
                self.max_confidence_by_class[detected.class_name] = detected.confidence

    def close(
        self,
        stop_reason: StopReason,
        ended_at_wall_clock: datetime,
        ended_at_monotonic: float,
    ) -> None:
        self._writer.release()
        Path(self.paths.partial_video_path).rename(self.paths.final_video_path)
        write_metadata(
            Path(self.paths.metadata_path),
            self._metadata(stop_reason, ended_at_wall_clock, ended_at_monotonic),
        )

    def _metadata(
        self,
        stop_reason: StopReason,
        ended_at_wall_clock: datetime,
        ended_at_monotonic: float,
    ) -> dict[str, Any]:
        return {
            "event_id": self.paths.event_id,
            "source": self.source,
            "started_at": self.started_at_wall_clock.isoformat(),
            "ended_at": ended_at_wall_clock.isoformat(),
            "duration_seconds": round(
                max(0.0, ended_at_monotonic - self.started_at_monotonic), 3
            ),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "model_path": self.model_path,
            "target_classes": list(self.target_classes),
            "detected_classes": sorted(self.detected_classes),
            "max_confidence_by_class": self.max_confidence_by_class,
            "stop_reason": stop_reason.value,
        }


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
