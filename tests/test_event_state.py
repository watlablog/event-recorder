from __future__ import annotations

from pathlib import Path

from event_recorder.config import HealthConfig, RecordingConfig, StartConfirmationConfig
from event_recorder.event_state import EventState
from event_recorder.models import DetectionResult, DetectorHealth, StopReason


def _recording_config(post_event_seconds: float = 3.0, max_clip_seconds: float = 600.0):
    return RecordingConfig(
        output_directory=Path("recordings"),
        extension="mp4",
        fourcc="mp4v",
        pre_event_seconds=1.0,
        post_event_seconds=post_event_seconds,
        max_clip_seconds=max_clip_seconds,
        max_prebuffer_mb=256,
        minimum_free_disk_gb=0,
    )


def _state(post_event_seconds: float = 3.0, max_clip_seconds: float = 600.0):
    return EventState(
        StartConfirmationConfig(
            window_results=3,
            required_positive_results=2,
            expire_seconds=2.0,
        ),
        _recording_config(post_event_seconds, max_clip_seconds),
        HealthConfig(detector_stale_seconds=5.0, detector_failure_seconds=30.0),
        detector_started_at_monotonic=0.0,
    )


def _result(frame_id: int, at: float, detected: bool) -> DetectionResult:
    return DetectionResult(
        frame_id=frame_id,
        captured_at_monotonic=at,
        detected=detected,
        detections=(),
    )


def test_single_positive_in_three_results_does_not_start_recording():
    state = _state()

    assert not state.observe_detection(_result(1, 0.0, False), 0.0).should_start
    assert not state.observe_detection(_result(2, 0.5, True), 0.5).should_start
    decision = state.observe_detection(_result(3, 1.0, False), 1.0)

    assert not decision.should_start
    assert not state.is_recording


def test_two_positives_in_three_recent_results_start_recording():
    state = _state()

    assert not state.observe_detection(_result(1, 0.0, True), 0.0).should_start
    assert not state.observe_detection(_result(2, 0.5, False), 0.5).should_start
    decision = state.observe_detection(_result(3, 1.0, True), 1.0)

    assert decision.should_start
    assert decision.event_start_monotonic == 0.0
    assert decision.event_start_frame_id == 1
    assert state.is_recording


def test_recording_continues_within_post_event_window_and_stops_after_it():
    state = _state(post_event_seconds=3.0)
    state.observe_detection(_result(1, 0.0, True), 0.0)
    state.observe_detection(_result(2, 0.5, False), 0.5)
    state.observe_detection(_result(3, 1.0, True), 1.0)

    state.observe_detection(_result(4, 2.0, False), 2.0)
    assert not state.absence_stop_decision(3.9).should_stop

    state.observe_detection(_result(5, 4.0, False), 4.0)
    decision = state.absence_stop_decision(4.0)
    assert decision.should_stop
    assert decision.reason == StopReason.TARGET_ABSENT


def test_max_clip_duration_requests_split():
    state = _state(max_clip_seconds=10.0)
    state.observe_detection(_result(1, 0.0, True), 0.0)
    state.observe_detection(_result(2, 0.5, False), 0.5)
    state.observe_detection(_result(3, 1.0, True), 1.0)

    decision = state.max_clip_stop_decision(10.0)

    assert decision.should_stop
    assert decision.reason == StopReason.MAX_CLIP_DURATION


def test_detector_health_degrades_and_fails_without_results():
    state = _state()

    assert state.detector_health(4.9) == DetectorHealth.HEALTHY
    assert state.detector_health(5.0) == DetectorHealth.DEGRADED
    assert state.detector_health(30.0) == DetectorHealth.FAILED
