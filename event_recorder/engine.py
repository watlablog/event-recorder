from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from event_recorder.camera import (
    CameraReadError,
    CameraStream,
    EndOfInput,
)
from event_recorder.config import AppConfig
from event_recorder.detector import (
    DetectorFailure,
    DetectorWorker,
    put_latest,
)
from event_recorder.event_state import EventState
from event_recorder.models import (
    DetectedObject,
    DetectionResult,
    DetectorHealth,
    FramePacket,
    RecorderStatus,
    StopReason,
)
from event_recorder.prebuffer import PreEventBuffer
from event_recorder.recorder import VideoClipWriter, VideoWriterError
from event_recorder.storage import create_event_paths, has_minimum_free_disk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineFrame:
    packet: FramePacket
    detections: tuple[DetectedObject, ...]
    status: RecorderStatus
    camera_fps: float | None
    detector_fps: float | None
    recording_elapsed: float | None


@dataclass(frozen=True)
class EngineCallbacks:
    on_frame: Callable[[EngineFrame], bool | None] | None = None
    on_status: Callable[[RecorderStatus, str], None] | None = None
    on_error: Callable[[str], None] | None = None


class RecorderEngine:
    def __init__(
        self,
        config: AppConfig,
        stop_event: threading.Event,
        callbacks: EngineCallbacks | None = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event
        self.callbacks = callbacks or EngineCallbacks()

    def run(self) -> int:
        camera = CameraStream(self.config.camera)
        detector_input: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        detector_results: queue.Queue[DetectionResult] = queue.Queue(maxsize=8)
        detector_failures: queue.Queue[DetectorFailure] = queue.Queue(maxsize=1)
        detector = DetectorWorker(
            self.config.model,
            detector_input,
            detector_results,
            detector_failures,
            self.stop_event,
        )
        detector_started = False
        state = EventState(
            self.config.start_confirmation,
            self.config.recording,
            self.config.health,
            detector_started_at_monotonic=time.monotonic(),
        )
        writer: VideoClipWriter | None = None
        latest_detections: tuple[DetectedObject, ...] = ()
        part = 1
        current_event_id: str | None = None
        pending_continuation = False
        frame_times: deque[float] = deque(maxlen=60)
        detector_times: deque[float] = deque(maxlen=30)
        packet: FramePacket | None = None

        try:
            self._status(RecorderStatus.WAITING, "Opening camera")
            camera.open()
            output_fps = camera.output_fps()
            packet = camera.read_packet(1)
            prebuffer = PreEventBuffer(
                output_fps=output_fps,
                pre_event_seconds=self.config.recording.pre_event_seconds,
                max_prebuffer_mb=self.config.recording.max_prebuffer_mb,
                first_frame_nbytes=packet.frame.nbytes,
            )
            if prebuffer.actual_seconds_capacity < self.config.recording.pre_event_seconds:
                LOGGER.warning(
                    "Prebuffer capacity is %.2fs, below requested %.2fs",
                    prebuffer.actual_seconds_capacity,
                    self.config.recording.pre_event_seconds,
                )
            detector.start()
            detector_started = True
            self._status(RecorderStatus.WAITING, "Waiting for target detection")

            frame_id = 1
            while packet is not None and not self.stop_event.is_set():
                now = packet.captured_at_monotonic
                frame_times.append(now)
                prebuffer.add(packet)
                put_latest(detector_input, packet)

                failure_code = self._handle_detector_failures(
                    detector_failures, writer, packet
                )
                if failure_code is not None:
                    return failure_code

                for result in _drain_results(detector_results):
                    detector_times.append(time.monotonic())
                    latest_detections = result.detections
                    start_decision = state.observe_detection(result, time.monotonic())
                    if writer is not None:
                        writer.observe_detection(result)
                    if start_decision.should_start and writer is None:
                        event_start_monotonic = (
                            start_decision.event_start_monotonic
                            if start_decision.event_start_monotonic is not None
                            else packet.captured_at_monotonic
                        )
                        writer = _start_writer(
                            config=self.config,
                            packet=packet,
                            output_fps=output_fps,
                            event_start_monotonic=event_start_monotonic,
                            event_id=None,
                            part=1,
                            prebuffer=prebuffer,
                        )
                        current_event_id = writer.paths.event_id
                        part = 1
                        writer.observe_detection(result)
                        self._status(RecorderStatus.RECORDING, "Recording event")

                health = state.detector_health(time.monotonic())
                if health == DetectorHealth.FAILED:
                    _close_writer(writer, StopReason.DETECTOR_FAILURE, packet)
                    self._error("Detector failed to return results.")
                    return 2

                if pending_continuation and writer is None:
                    part += 1
                    state.mark_next_clip_started(packet.captured_at_monotonic)
                    writer = _start_writer(
                        config=self.config,
                        packet=packet,
                        output_fps=output_fps,
                        event_start_monotonic=packet.captured_at_monotonic,
                        event_id=current_event_id,
                        part=part,
                        prebuffer=None,
                    )
                    pending_continuation = False
                    self._status(RecorderStatus.RECORDING, "Recording next clip part")

                if writer is not None:
                    writer.write(packet)
                    split_decision = state.max_clip_stop_decision(
                        packet.captured_at_monotonic
                    )
                    absence_decision = state.absence_stop_decision(
                        packet.captured_at_monotonic
                    )
                    if split_decision.should_stop:
                        _close_writer(writer, StopReason.MAX_CLIP_DURATION, packet)
                        writer = None
                        pending_continuation = True
                    elif absence_decision.should_stop:
                        _close_writer(writer, StopReason.TARGET_ABSENT, packet)
                        writer = None
                        state.mark_recording_stopped()
                        current_event_id = None
                        part = 1
                        self._status(
                            RecorderStatus.WAITING,
                            "Waiting for target detection",
                        )

                status = _recorder_status(writer, health)
                recording_elapsed = (
                    packet.captured_at_monotonic - writer.started_at_monotonic
                    if writer is not None
                    else None
                )
                should_stop = self._frame(
                    EngineFrame(
                        packet=packet,
                        detections=latest_detections,
                        status=status,
                        camera_fps=_fps_from_times(frame_times),
                        detector_fps=_fps_from_times(detector_times),
                        recording_elapsed=recording_elapsed,
                    )
                )
                if should_stop:
                    self.stop_event.set()
                    break

                frame_id += 1
                try:
                    packet = camera.read_packet(frame_id)
                except EndOfInput:
                    _close_writer(writer, StopReason.END_OF_FILE, packet)
                    return 0
                except CameraReadError:
                    _close_writer(writer, StopReason.CAMERA_FAILURE, packet)
                    writer = None
                    state.mark_recording_stopped()
                    if camera.is_file_source or not camera.reopen_with_retries():
                        self._error("Failed to read from camera.")
                        return 2
                    packet = camera.read_packet(frame_id)

            if writer is not None:
                _close_writer(writer, StopReason.USER_SHUTDOWN, packet)
            return 0
        except VideoWriterError as exc:
            self._error(str(exc))
            raise
        finally:
            self.stop_event.set()
            if detector_started:
                detector.join(timeout=5)
            camera.release()

    def _handle_detector_failures(
        self,
        failure_queue: queue.Queue[DetectorFailure],
        writer: VideoClipWriter | None,
        packet: FramePacket,
    ) -> int | None:
        try:
            failure = failure_queue.get_nowait()
        except queue.Empty:
            return None
        LOGGER.error("Detector failed: %s", failure.message)
        self._error(f"Detector failed: {failure.message}")
        _close_writer(writer, StopReason.DETECTOR_FAILURE, packet)
        return 2

    def _frame(self, frame: EngineFrame) -> bool:
        if self.callbacks.on_frame is None:
            return False
        return bool(self.callbacks.on_frame(frame))

    def _status(self, status: RecorderStatus, message: str) -> None:
        if self.callbacks.on_status is not None:
            self.callbacks.on_status(status, message)

    def _error(self, message: str) -> None:
        if self.callbacks.on_error is not None:
            self.callbacks.on_error(message)


def _start_writer(
    config: AppConfig,
    packet: FramePacket,
    output_fps: float,
    event_start_monotonic: float,
    event_id: str | None,
    part: int,
    prebuffer: PreEventBuffer | None,
) -> VideoClipWriter:
    if not has_minimum_free_disk(
        config.recording.output_directory, config.recording.minimum_free_disk_gb
    ):
        raise VideoWriterError(
            f"Free disk space is below {config.recording.minimum_free_disk_gb:.2f} GB."
        )

    paths = create_event_paths(
        config.recording,
        packet.captured_at_wall_clock,
        event_id=event_id,
        part=part,
    )
    started_at_packet = packet
    buffered_frames = []
    if prebuffer is not None:
        buffered_frames = prebuffer.frames_for_event(event_start_monotonic)
        for buffered in buffered_frames:
            if buffered.captured_at_monotonic >= event_start_monotonic:
                started_at_packet = buffered
                break
    frame_height, frame_width = packet.frame.shape[:2]
    writer = VideoClipWriter(
        paths=paths,
        recording=config.recording,
        source=config.camera.source,
        model_path=config.model.path,
        target_classes=config.model.target_classes,
        fps=output_fps,
        frame_size=(frame_width, frame_height),
        started_at_wall_clock=started_at_packet.captured_at_wall_clock,
        started_at_monotonic=event_start_monotonic,
    )
    for buffered in buffered_frames:
        writer.write(buffered)
    return writer


def _close_writer(
    writer: VideoClipWriter | None,
    reason: StopReason,
    packet: FramePacket | None,
) -> None:
    if writer is None or packet is None:
        return
    writer.close(
        reason,
        ended_at_wall_clock=packet.captured_at_wall_clock,
        ended_at_monotonic=packet.captured_at_monotonic,
    )


def _drain_results(
    result_queue: queue.Queue[DetectionResult],
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            return results


def _recorder_status(
    writer: VideoClipWriter | None, health: DetectorHealth
) -> RecorderStatus:
    if health == DetectorHealth.DEGRADED:
        return RecorderStatus.DEGRADED
    if writer is not None:
        return RecorderStatus.RECORDING
    return RecorderStatus.WAITING


def _fps_from_times(times: deque[float]) -> float | None:
    if len(times) < 2:
        return None
    elapsed = times[-1] - times[0]
    if elapsed <= 0:
        return None
    return (len(times) - 1) / elapsed
