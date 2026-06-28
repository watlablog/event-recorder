from __future__ import annotations

from event_recorder.config import parse_config


def test_audio_config_defaults_to_disabled():
    config = parse_config({"model": {"target_classes": ["person", "car"]}})

    assert config.audio.enabled is False
    assert config.audio.device is None
    assert config.audio.sample_rate == 48000
    assert config.audio.channels == 1
    assert config.audio.fallback_to_video_only is True


def test_audio_config_parses_enabled_device_and_format():
    config = parse_config(
        {
            "model": {"target_classes": ["person", "car"]},
            "audio": {
                "enabled": True,
                "device": "3",
                "sample_rate": 44100,
                "channels": 2,
                "fallback_to_video_only": False,
            },
        }
    )

    assert config.audio.enabled is True
    assert config.audio.device == 3
    assert config.audio.sample_rate == 44100
    assert config.audio.channels == 2
    assert config.audio.fallback_to_video_only is False
