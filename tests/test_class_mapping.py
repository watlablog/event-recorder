from __future__ import annotations

import pytest

from event_recorder.detector import ClassMappingError, resolve_target_class_ids


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
