from __future__ import annotations

from dataclasses import replace

from event_recorder.config import AppConfig


def with_runtime_selection(
    config: AppConfig,
    camera_source: int,
    target_classes: tuple[str, ...],
) -> AppConfig:
    return replace(
        config,
        camera=replace(config.camera, source=camera_source),
        model=replace(config.model, target_classes=target_classes),
    )
