from __future__ import annotations

from dataclasses import replace

from event_recorder.config import AppConfig


def with_runtime_selection(
    config: AppConfig,
    camera_source: int,
    target_classes: tuple[str, ...],
    audio_enabled: bool | None = None,
    audio_device: int | str | None = None,
    recording_draw_boxes: bool | None = None,
) -> AppConfig:
    audio = config.audio
    if audio_enabled is not None:
        audio = replace(audio, enabled=audio_enabled, device=audio_device)
    elif audio_device is not None:
        audio = replace(audio, device=audio_device)

    recording = config.recording
    if recording_draw_boxes is not None:
        recording = replace(recording, draw_boxes=recording_draw_boxes)

    return replace(
        config,
        camera=replace(config.camera, source=camera_source),
        model=replace(config.model, target_classes=target_classes),
        recording=recording,
        audio=audio,
    )
