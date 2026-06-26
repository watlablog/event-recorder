from __future__ import annotations

from event_recorder.model_metadata import (
    default_selected_classes,
    model_classes_from_names,
)


def test_model_classes_are_sorted_by_class_id():
    classes = model_classes_from_names({2: "car", 0: "person", 1: "bicycle"})

    assert [item.name for item in classes] == ["person", "bicycle", "car"]


def test_default_selected_classes_prefers_person_and_car():
    classes = model_classes_from_names({0: " person ", 2: "Car", 7: "truck"})

    assert default_selected_classes(classes) == (" person ", "Car")
