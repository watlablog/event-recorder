from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from event_recorder.config import HealthConfig, RecordingConfig, StartConfirmationConfig
from event_recorder.models import DetectionResult, DetectorHealth, StopReason


@dataclass(frozen=True)
class StartDecision:
    should_start: bool
    event_start_monotonic: float | None = None
    event_start_frame_id: int | None = None


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: StopReason | None = None


class EventState:
    def __init__(
        self,
        start_config: StartConfirmationConfig,
        recording_config: RecordingConfig,
        health_config: HealthConfig,
        detector_started_at_monotonic: float = 0.0,
    ) -> None:
        self.start_config = start_config
        self.recording_config = recording_config
        self.health_config = health_config
        self.detector_started_at_monotonic = detector_started_at_monotonic
        self._recent_results: deque[DetectionResult] = deque(
            maxlen=start_config.window_results
        )
        self.is_recording = False
        self.recording_started_at_monotonic: float | None = None
        self.last_positive_at_monotonic: float | None = None
        self.last_detector_result_at_monotonic: float | None = None

    def observe_detection(
        self, result: DetectionResult, now_monotonic: float
    ) -> StartDecision:
        self.last_detector_result_at_monotonic = now_monotonic
        self._recent_results.append(result)
        if result.detected:
            self.last_positive_at_monotonic = result.captured_at_monotonic

        if self.is_recording or self.detector_health(now_monotonic) != DetectorHealth.HEALTHY:
            return StartDecision(False)

        window = [
            item
            for item in self._recent_results
            if now_monotonic - item.captured_at_monotonic
            <= self.start_config.expire_seconds
        ]
        if len(window) < self.start_config.window_results:
            return StartDecision(False)

        positives = [item for item in window if item.detected]
        if len(positives) < self.start_config.required_positive_results:
            return StartDecision(False)

        first_positive = min(positives, key=lambda item: item.captured_at_monotonic)
        self.is_recording = True
        self.recording_started_at_monotonic = first_positive.captured_at_monotonic
        return StartDecision(
            True,
            event_start_monotonic=first_positive.captured_at_monotonic,
            event_start_frame_id=first_positive.frame_id,
        )

    def detector_health(self, now_monotonic: float) -> DetectorHealth:
        last_result_at = self.last_detector_result_at_monotonic
        reference = (
            last_result_at
            if last_result_at is not None
            else self.detector_started_at_monotonic
        )
        elapsed = now_monotonic - reference
        if elapsed >= self.health_config.detector_failure_seconds:
            return DetectorHealth.FAILED
        if elapsed >= self.health_config.detector_stale_seconds:
            return DetectorHealth.DEGRADED
        return DetectorHealth.HEALTHY

    def absence_stop_decision(self, now_monotonic: float) -> StopDecision:
        if not self.is_recording or self.last_positive_at_monotonic is None:
            return StopDecision(False)
        if self.detector_health(now_monotonic) != DetectorHealth.HEALTHY:
            return StopDecision(False)
        elapsed = now_monotonic - self.last_positive_at_monotonic
        if elapsed >= self.recording_config.post_event_seconds:
            return StopDecision(True, StopReason.TARGET_ABSENT)
        return StopDecision(False)

    def max_clip_stop_decision(self, now_monotonic: float) -> StopDecision:
        if not self.is_recording or self.recording_started_at_monotonic is None:
            return StopDecision(False)
        elapsed = now_monotonic - self.recording_started_at_monotonic
        if elapsed >= self.recording_config.max_clip_seconds:
            return StopDecision(True, StopReason.MAX_CLIP_DURATION)
        return StopDecision(False)

    def mark_recording_stopped(self) -> None:
        self.is_recording = False
        self.recording_started_at_monotonic = None
        self.last_positive_at_monotonic = None
        self._recent_results.clear()

    def mark_next_clip_started(self, now_monotonic: float) -> None:
        self.is_recording = True
        self.recording_started_at_monotonic = now_monotonic
