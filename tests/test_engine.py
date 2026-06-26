from __future__ import annotations

import threading
from datetime import datetime, timezone

import numpy as np

import event_recorder.engine as engine
from event_recorder.config import parse_config
from event_recorder.models import FramePacket


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
