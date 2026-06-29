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
    recording_draw_timestamp: bool | None = None,
    night_enhancement_enabled: bool | None = None,
    night_enhancement_contrast: float | None = None,
    night_enhancement_brightness: float | None = None,
    night_enhancement_gamma: float | None = None,
) -> AppConfig:
    audio = config.audio
    if audio_enabled is not None:
        audio = replace(audio, enabled=audio_enabled, device=audio_device)
    elif audio_device is not None:
        audio = replace(audio, device=audio_device)

    recording = config.recording
    if recording_draw_boxes is not None:
        recording = replace(recording, draw_boxes=recording_draw_boxes)
    if recording_draw_timestamp is not None:
        recording = replace(recording, draw_timestamp=recording_draw_timestamp)

    night_enhancement = config.night_enhancement
    if night_enhancement_enabled is not None:
        night_enhancement = replace(
            night_enhancement,
            enabled=night_enhancement_enabled,
        )
    if night_enhancement_contrast is not None:
        night_enhancement = replace(
            night_enhancement,
            contrast=night_enhancement_contrast,
        )
    if night_enhancement_brightness is not None:
        night_enhancement = replace(
            night_enhancement,
            brightness=night_enhancement_brightness,
        )
    if night_enhancement_gamma is not None:
        night_enhancement = replace(
            night_enhancement,
            gamma=night_enhancement_gamma,
        )

    return replace(
        config,
        camera=replace(config.camera, source=camera_source),
        model=replace(config.model, target_classes=target_classes),
        recording=recording,
        audio=audio,
        night_enhancement=night_enhancement,
    )
