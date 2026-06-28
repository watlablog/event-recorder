from __future__ import annotations

import threading
from datetime import datetime, timezone

import numpy as np

import event_recorder.engine as engine
from event_recorder.config import parse_config
from event_recorder.models import DetectedObject, DetectionResult, FramePacket


class FakeCameraStream:
    last_instance: "FakeCameraStream | None" = None

    def __init__(self, config):
        self.config = config
        self.released = False
        FakeCameraStream.last_instance = self

    @property
    def is_file_source(self) -> bool:
        return False

    def open(self) -> None:
        pass

    def output_fps(self) -> float:
        return 30.0

    def read_packet(self, frame_id: int) -> FramePacket:
        return FramePacket(
            frame_id=frame_id,
            captured_at_monotonic=float(frame_id),
            captured_at_wall_clock=datetime.now(timezone.utc),
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
        )

    def release(self) -> None:
        self.released = True

    def reopen_with_retries(self) -> bool:
        return False


class FakeDetectorWorker:
    last_instance: "FakeDetectorWorker | None" = None

    def __init__(self, *args, **kwargs):
        self.started = False
        self.joined = False
        FakeDetectorWorker.last_instance = self

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.joined = True


def test_recorder_engine_releases_camera_and_detector_on_stop(monkeypatch):
    config = parse_config({"model": {"target_classes": ["person", "car"]}})
    stop_event = threading.Event()
    stop_event.set()
    monkeypatch.setattr(engine, "CameraStream", FakeCameraStream)
    monkeypatch.setattr(engine, "DetectorWorker", FakeDetectorWorker)

    code = engine.RecorderEngine(config, stop_event).run()

    assert code == 0
    assert FakeCameraStream.last_instance is not None
    assert FakeCameraStream.last_instance.released
    assert FakeDetectorWorker.last_instance is not None
    assert FakeDetectorWorker.last_instance.started
    assert FakeDetectorWorker.last_instance.joined


class FailingAudioCapture:
    def __init__(self, config):
        self.config = config

    def start(self) -> None:
        raise RuntimeError("no microphone permission")

    def stop(self) -> None:
        pass


def test_recorder_engine_continues_video_when_audio_start_fails(monkeypatch):
    config = parse_config(
        {
            "model": {"target_classes": ["person", "car"]},
            "audio": {"enabled": True, "fallback_to_video_only": True},
        }
    )
    stop_event = threading.Event()
    stop_event.set()
    errors: list[str] = []
    monkeypatch.setattr(engine, "CameraStream", FakeCameraStream)
    monkeypatch.setattr(engine, "DetectorWorker", FakeDetectorWorker)
    monkeypatch.setattr(engine, "AudioCapture", FailingAudioCapture)

    code = engine.RecorderEngine(
        config,
        stop_event,
        engine.EngineCallbacks(on_error=errors.append),
    ).run()

    assert code == 0
    assert any("Audio disabled" in error for error in errors)


def _detection(frame_id: int, detections: tuple[DetectedObject, ...]) -> DetectionResult:
    return DetectionResult(
        frame_id=frame_id,
        captured_at_monotonic=float(frame_id),
        detected=bool(detections),
        detections=detections,
    )


def _detected(name: str, xyxy: tuple[float, float, float, float]) -> DetectedObject:
    return DetectedObject(
        class_id=0,
        class_name=name,
        confidence=0.9,
        xyxy=xyxy,
    )


def test_recorder_engine_filter_runs_before_start_decision(monkeypatch):
    config = parse_config(
        {
            "model": {"target_classes": ["person"]},
            "start_confirmation": {
                "window_results": 1,
                "required_positive_results": 1,
                "expire_seconds": 1_000_000_000.0,
            },
        }
    )
    stop_event = threading.Event()
    frames: list[engine.EngineFrame] = []
    started: list[bool] = []
    positive = _detection(1, (_detected("person", (1.0, 1.0, 2.0, 2.0)),))
    pending_results = [positive]
    monkeypatch.setattr(engine, "CameraStream", FakeCameraStream)
    monkeypatch.setattr(engine, "DetectorWorker", FakeDetectorWorker)
    monkeypatch.setattr(
        engine,
        "_drain_results",
        lambda _queue: [pending_results.pop(0)] if pending_results else [],
    )
    monkeypatch.setattr(
        engine,
        "_start_writer",
        lambda *args, **kwargs: started.append(True),
    )

    def stop_after_frame(frame: engine.EngineFrame) -> bool:
        frames.append(frame)
        return True

    code = engine.RecorderEngine(
        config,
        stop_event,
        engine.EngineCallbacks(
            on_frame=stop_after_frame,
            detection_filter=lambda result: DetectionResult(
                frame_id=result.frame_id,
                captured_at_monotonic=result.captured_at_monotonic,
                detected=False,
                detections=(),
            ),
        ),
    ).run()

    assert code == 0
    assert started == []
    assert frames[0].detections == ()


def test_recorder_engine_passes_filtered_detections_to_writer(monkeypatch):
    config = parse_config(
        {
            "model": {"target_classes": ["person", "car"]},
            "start_confirmation": {
                "window_results": 1,
                "required_positive_results": 1,
                "expire_seconds": 1_000_000_000.0,
            },
        }
    )
    stop_event = threading.Event()
    inside = _detected("person", (1.0, 1.0, 2.0, 2.0))
    outside = _detected("car", (5.0, 5.0, 8.0, 8.0))
    pending_results = [_detection(1, (inside, outside))]
    starts: list[tuple[DetectedObject, ...]] = []

    class FakeWriter:
        def __init__(self) -> None:
            self.started_at_monotonic = 1.0
            self.paths = type("Paths", (), {"event_id": "event"})()
            self.observed: list[tuple[DetectedObject, ...]] = []
            self.written: list[tuple[DetectedObject, ...]] = []

        def observe_detection(self, result: DetectionResult) -> None:
            self.observed.append(result.detections)

        def write(self, packet: FramePacket, detections=()) -> None:
            self.written.append(detections)

        def close(self, *args, **kwargs) -> None:
            pass

    writer = FakeWriter()
    monkeypatch.setattr(engine, "CameraStream", FakeCameraStream)
    monkeypatch.setattr(engine, "DetectorWorker", FakeDetectorWorker)
    monkeypatch.setattr(
        engine,
        "_drain_results",
        lambda _queue: [pending_results.pop(0)] if pending_results else [],
    )

    def fake_start_writer(*args, **kwargs):
        starts.append(kwargs["current_detections"])
        return writer

    monkeypatch.setattr(engine, "_start_writer", fake_start_writer)

    code = engine.RecorderEngine(
        config,
        stop_event,
        engine.EngineCallbacks(
            on_frame=lambda _frame: True,
            detection_filter=lambda result: DetectionResult(
                frame_id=result.frame_id,
                captured_at_monotonic=result.captured_at_monotonic,
                detected=True,
                detections=(outside,),
            ),
        ),
    ).run()

    assert code == 0
    assert starts == [(outside,)]
    assert writer.observed == [(outside,)]
    assert writer.written == [(outside,)]
