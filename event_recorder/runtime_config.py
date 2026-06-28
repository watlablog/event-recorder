from __future__ import annotations

from dataclasses import replace

from event_recorder.config import AppConfig


def with_runtime_selection(
    config: AppConfig,
    camera_source: int,
    target_classes: tuple[str, ...],
    audio_enabled: bool | None = None,
    audio_device: int | str | None = None,
) -> AppConfig:
    audio = config.audio
    if audio_enabled is not None:
        audio = replace(audio, enabled=audio_enabled, device=audio_device)
    elif audio_device is not None:
        audio = replace(audio, device=audio_device)

    return replace(
        config,
        camera=replace(config.camera, source=camera_source),
        model=replace(config.model, target_classes=target_classes),
        audio=audio,
    )
