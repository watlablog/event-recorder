from __future__ import annotations

import pytest

from event_recorder.config import ConfigError, parse_config
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


def test_config_parses_night_enhancement_defaults():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    assert config.night_enhancement.enabled is False
    assert config.night_enhancement.contrast == 1.35
    assert config.night_enhancement.brightness == 12.0
    assert config.night_enhancement.gamma == 1.20


def test_config_parses_night_enhancement_values():
    config = parse_config(
        {
            "model": {"target_classes": ["person", "car"]},
            "night_enhancement": {
                "enabled": True,
                "contrast": 1.5,
                "brightness": 20,
                "gamma": 1.4,
            },
        }
    )

    assert config.night_enhancement.enabled is True
    assert config.night_enhancement.contrast == 1.5
    assert config.night_enhancement.brightness == 20.0
    assert config.night_enhancement.gamma == 1.4


def test_config_rejects_invalid_night_enhancement_values():
    with pytest.raises(ConfigError):
        parse_config(
            {
                "model": {"target_classes": ["person", "car"]},
                "night_enhancement": {"gamma": 0.1},
            }
        )


def test_runtime_config_replaces_night_enhancement_selection():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    updated = with_runtime_selection(
        config,
        camera_source=1,
        target_classes=("person",),
        night_enhancement_enabled=True,
        night_enhancement_contrast=1.6,
        night_enhancement_brightness=18.0,
        night_enhancement_gamma=1.3,
    )

    assert updated.night_enhancement.enabled is True
    assert updated.night_enhancement.contrast == 1.6
    assert updated.night_enhancement.brightness == 18.0
    assert updated.night_enhancement.gamma == 1.3
    assert config.night_enhancement.enabled is False
