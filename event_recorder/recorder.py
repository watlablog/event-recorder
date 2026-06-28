from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from event_recorder.audio import AudioMuxError, mux_audio_video
from event_recorder.config import RecordingConfig
from event_recorder.models import (
    DetectedObject,
    DetectionResult,
    EventPaths,
    FramePacket,
    StopReason,
)
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

        self._cv2 = cv2
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

    def write(
        self,
        packet: FramePacket,
        detections: tuple[DetectedObject, ...] = (),
    ) -> None:
        if (
            self.last_written_frame_id is not None
            and packet.frame_id <= self.last_written_frame_id
        ):
            return
        frame = packet.frame
        if self.recording.draw_boxes and detections:
            frame = _draw_detections(self._cv2, frame, detections)
        self._writer.write(frame)
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
        audio_path: Path | None = None,
        audio_requested: bool = False,
        audio_status: str = "disabled",
        audio_device: int | str | None = None,
        audio_sample_rate: int | None = None,
        audio_channels: int | None = None,
    ) -> None:
        self._writer.release()
        audio_recorded = False
        final_audio_status = audio_status
        partial_video_path = Path(self.paths.partial_video_path)
        final_video_path = Path(self.paths.final_video_path)
        muxed_partial_path = Path(self.paths.muxed_partial_video_path)

        if audio_path is not None and audio_path.exists() and audio_status == "captured":
            try:
                mux_audio_video(partial_video_path, audio_path, muxed_partial_path)
                muxed_partial_path.rename(final_video_path)
                _unlink_if_exists(partial_video_path)
                _unlink_if_exists(audio_path)
                audio_recorded = True
                final_audio_status = "muxed"
            except AudioMuxError:
                partial_video_path.rename(final_video_path)
                _unlink_if_exists(muxed_partial_path)
                final_audio_status = "mux_failed"
        else:
            partial_video_path.rename(final_video_path)

        write_metadata(
            Path(self.paths.metadata_path),
            self._metadata(
                stop_reason,
                ended_at_wall_clock,
                ended_at_monotonic,
                audio_requested=audio_requested,
                audio_recorded=audio_recorded,
                audio_status=final_audio_status,
                audio_device=audio_device,
                audio_sample_rate=audio_sample_rate,
                audio_channels=audio_channels,
            ),
        )

    def _metadata(
        self,
        stop_reason: StopReason,
        ended_at_wall_clock: datetime,
        ended_at_monotonic: float,
        audio_requested: bool,
        audio_recorded: bool,
        audio_status: str,
        audio_device: int | str | None,
        audio_sample_rate: int | None,
        audio_channels: int | None,
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
            "draw_boxes": self.recording.draw_boxes,
            "detected_classes": sorted(self.detected_classes),
            "max_confidence_by_class": self.max_confidence_by_class,
            "stop_reason": stop_reason.value,
            "audio_requested": audio_requested,
            "audio_recorded": audio_recorded,
            "audio_status": audio_status,
            "audio_device": audio_device,
            "audio_sample_rate": audio_sample_rate,
            "audio_channels": audio_channels,
        }


def _draw_detections(cv2, frame, detections: tuple[DetectedObject, ...]):
    display = frame.copy()
    for detected in detections:
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
    return display


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
