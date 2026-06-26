from __future__ import annotations

from event_recorder.config import parse_config
from event_recorder.runtime_config import with_runtime_selection


def test_runtime_config_replaces_camera_and_target_classes_only():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    updated = with_runtime_selection(
        config,
        camera_source=3,
        target_classes=("person", "truck"),
    )

    assert updated.camera.source == 3
    assert updated.model.target_classes == ("person", "truck")
    assert updated.camera.width == config.camera.width
    assert updated.recording.output_directory == config.recording.output_directory
    assert config.camera.source == 0
    assert config.model.target_classes == ("person", "car")
