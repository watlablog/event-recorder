from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from event_recorder.detector import (
    ClassMappingError,
    _to_detection_result,
    resolve_target_class_ids,
)
from event_recorder.models import FramePacket


def test_resolves_target_classes_by_normalized_name():
    names = {0: " person ", 1: "BICYCLE", 2: "Car"}

    assert resolve_target_class_ids(names, ["PERSON", " car "]) == (0, 2)


def test_resolves_sequence_model_names():
    names = ["person", "bicycle", "car"]

    assert resolve_target_class_ids(names, ["car", "person"]) == (2, 0)


def test_missing_target_class_reports_available_classes():
    names = {0: "person", 2: "car"}

    with pytest.raises(ClassMappingError) as exc_info:
        resolve_target_class_ids(names, ["truck"])

    message = str(exc_info.value)
    assert "truck" in message
    assert "person" in message
    assert "car" in message


def test_detection_result_filters_non_target_class_ids():
    packet = FramePacket(
        frame_id=1,
        captured_at_monotonic=10.0,
        captured_at_wall_clock=datetime.now(timezone.utc),
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    results = [_FakeResult(_FakeBoxes(cls=[0, 2], conf=[0.9, 0.8]))]

    result = _to_detection_result(
        packet,
        results,
        {0: "person", 2: "car"},
        allowed_class_ids=(2,),
    )

    assert result.detected is True
    assert [item.class_name for item in result.detections] == ["car"]


def test_detection_result_is_negative_when_only_non_targets_are_returned():
    packet = FramePacket(
        frame_id=1,
        captured_at_monotonic=10.0,
        captured_at_wall_clock=datetime.now(timezone.utc),
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    results = [_FakeResult(_FakeBoxes(cls=[0], conf=[0.9]))]

    result = _to_detection_result(
        packet,
        results,
        {0: "person", 2: "car"},
        allowed_class_ids=(2,),
    )

    assert result.detected is False
    assert result.detections == ()


class _FakeBoxes:
    def __init__(self, cls, conf):
        self.cls = np.array(cls)
        self.conf = np.array(conf)
        self.xyxy = np.array([[1.0, 2.0, 3.0, 4.0] for _ in cls])

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
