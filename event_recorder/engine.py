from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from event_recorder.audio import AudioCapture, AudioClipWriter, AudioError, AudioPreBuffer
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
        audio_capture: AudioCapture | None = None
        audio_prebuffer: AudioPreBuffer | None = None
        audio_writer: AudioClipWriter | None = None
        audio_status = "disabled" if not self.config.audio.enabled else "not_started"
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
            audio_capture, audio_prebuffer, audio_status = self._start_audio_capture()
            self._status(RecorderStatus.WAITING, "Waiting for target detection")

            frame_id = 1
            while packet is not None and not self.stop_event.is_set():
                now = packet.captured_at_monotonic
                frame_times.append(now)
                prebuffer.add(packet)
                audio_capture, audio_status = self._drain_audio(
                    audio_capture,
                    audio_prebuffer,
                    audio_writer,
                    audio_status,
                )
                put_latest(detector_input, packet)

                failure_code = self._handle_detector_failures(
                    detector_failures,
                    writer,
                    audio_writer,
                    packet,
                    audio_status,
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
                        audio_writer, audio_status = self._start_audio_clip(
                            writer,
                            audio_prebuffer,
                            event_start_monotonic,
                            audio_capture,
                            audio_status,
                        )
                        self._status(RecorderStatus.RECORDING, "Recording event")

                health = state.detector_health(time.monotonic())
                if health == DetectorHealth.FAILED:
                    _close_clip(
                        writer,
                        audio_writer,
                        self.config,
                        StopReason.DETECTOR_FAILURE,
                        packet,
                        audio_status,
                    )
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
                    audio_writer, audio_status = self._start_audio_clip(
                        writer,
                        None,
                        packet.captured_at_monotonic,
                        audio_capture,
                        audio_status,
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
                        _close_clip(
                            writer,
                            audio_writer,
                            self.config,
                            StopReason.MAX_CLIP_DURATION,
                            packet,
                            audio_status,
                        )
                        writer = None
                        audio_writer = None
                        if audio_capture is not None:
                            audio_status = "capturing"
                        pending_continuation = True
                    elif absence_decision.should_stop:
                        _close_clip(
                            writer,
                            audio_writer,
                            self.config,
                            StopReason.TARGET_ABSENT,
                            packet,
                            audio_status,
                        )
                        writer = None
                        audio_writer = None
                        if audio_capture is not None:
                            audio_status = "capturing"
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
                    _close_clip(
                        writer,
                        audio_writer,
                        self.config,
                        StopReason.END_OF_FILE,
                        packet,
                        audio_status,
                    )
                    return 0
                except CameraReadError:
                    _close_clip(
                        writer,
                        audio_writer,
                        self.config,
                        StopReason.CAMERA_FAILURE,
                        packet,
                        audio_status,
                    )
                    writer = None
                    audio_writer = None
                    if audio_capture is not None:
                        audio_status = "capturing"
                    state.mark_recording_stopped()
                    if camera.is_file_source or not camera.reopen_with_retries():
                        self._error("Failed to read from camera.")
                        return 2
                    packet = camera.read_packet(frame_id)

            if writer is not None:
                _close_clip(
                    writer,
                    audio_writer,
                    self.config,
                    StopReason.USER_SHUTDOWN,
                    packet,
                    audio_status,
                )
            return 0
        except VideoWriterError as exc:
            self._error(str(exc))
            raise
        finally:
            self.stop_event.set()
            if detector_started:
                detector.join(timeout=5)
            if audio_capture is not None:
                audio_capture.stop()
            camera.release()

    def _handle_detector_failures(
        self,
        failure_queue: queue.Queue[DetectorFailure],
        writer: VideoClipWriter | None,
        audio_writer: AudioClipWriter | None,
        packet: FramePacket,
        audio_status: str,
    ) -> int | None:
        try:
            failure = failure_queue.get_nowait()
        except queue.Empty:
            return None
        LOGGER.error("Detector failed: %s", failure.message)
        self._error(f"Detector failed: {failure.message}")
        _close_clip(
            writer,
            audio_writer,
            self.config,
            StopReason.DETECTOR_FAILURE,
            packet,
            audio_status,
        )
        return 2

    def _start_audio_capture(
        self,
    ) -> tuple[AudioCapture | None, AudioPreBuffer | None, str]:
        if not self.config.audio.enabled:
            return None, None, "disabled"
        try:
            audio_capture = AudioCapture(self.config.audio)
            audio_capture.start()
            audio_prebuffer = AudioPreBuffer(
                self.config.recording.pre_event_seconds,
                self.config.audio.sample_rate,
            )
            return audio_capture, audio_prebuffer, "capturing"
        except Exception as exc:
            self._error(f"Audio disabled: {exc}")
            if not self.config.audio.fallback_to_video_only:
                raise AudioError(str(exc)) from exc
            return None, None, "capture_failed"

    def _drain_audio(
        self,
        audio_capture: AudioCapture | None,
        audio_prebuffer: AudioPreBuffer | None,
        audio_writer: AudioClipWriter | None,
        audio_status: str,
    ) -> tuple[AudioCapture | None, str]:
        if audio_capture is None:
            return None, audio_status
        try:
            blocks = audio_capture.drain()
        except AudioError as exc:
            self._error(f"Audio disabled: {exc}")
            audio_capture.stop()
            return None, "capture_failed"
        for block in blocks:
            if audio_prebuffer is not None:
                audio_prebuffer.add(block)
            if audio_writer is not None:
                audio_writer.write(block)
        return audio_capture, audio_status

    def _start_audio_clip(
        self,
        writer: VideoClipWriter,
        audio_prebuffer: AudioPreBuffer | None,
        event_start_monotonic: float,
        audio_capture: AudioCapture | None,
        audio_status: str,
    ) -> tuple[AudioClipWriter | None, str]:
        if not self.config.audio.enabled:
            return None, "disabled"
        if audio_capture is None:
            return None, audio_status
        try:
            audio_writer = AudioClipWriter(
                writer.paths.partial_audio_path,
                self.config.audio.sample_rate,
                self.config.audio.channels,
            )
            if audio_prebuffer is not None:
                for block in audio_prebuffer.blocks_for_event(event_start_monotonic):
                    audio_writer.write(block)
            return audio_writer, "captured"
        except Exception as exc:
            self._error(f"Audio disabled for this clip: {exc}")
            return None, "write_failed"

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


def _close_clip(
    writer: VideoClipWriter | None,
    audio_writer: AudioClipWriter | None,
    config: AppConfig,
    reason: StopReason,
    packet: FramePacket | None,
    audio_status: str,
) -> None:
    if writer is None or packet is None:
        return
    audio_path = None
    final_audio_status = audio_status
    if audio_writer is not None:
        audio_writer.close()
        if audio_writer.frames_written > 0:
            audio_path = audio_writer.path
            final_audio_status = "captured"
        else:
            final_audio_status = "no_audio_samples"
    elif config.audio.enabled and audio_status in {"capturing", "captured"}:
        final_audio_status = "no_audio_samples"

    writer.close(
        reason,
        ended_at_wall_clock=packet.captured_at_wall_clock,
        ended_at_monotonic=packet.captured_at_monotonic,
        audio_path=audio_path,
        audio_requested=config.audio.enabled,
        audio_status=final_audio_status,
        audio_device=config.audio.device,
        audio_sample_rate=config.audio.sample_rate if config.audio.enabled else None,
        audio_channels=config.audio.channels if config.audio.enabled else None,
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
