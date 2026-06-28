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


def test_runtime_config_replaces_audio_selection():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    updated = with_runtime_selection(
        config,
        camera_source=1,
        target_classes=("person",),
        audio_enabled=True,
        audio_device=4,
    )

    assert updated.audio.enabled is True
    assert updated.audio.device == 4
    assert config.audio.enabled is False
    assert config.audio.device is None


def test_runtime_config_replaces_recording_draw_boxes():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    updated = with_runtime_selection(
        config,
        camera_source=1,
        target_classes=("person",),
        recording_draw_boxes=True,
    )

    assert updated.recording.draw_boxes is True
    assert config.recording.draw_boxes is False


def test_config_parses_recording_draw_boxes():
    config = parse_config(
        {
            "model": {"target_classes": ["person", "car"]},
            "recording": {"draw_boxes": True},
        }
    )

    assert config.recording.draw_boxes is True
