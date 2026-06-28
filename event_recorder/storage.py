from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from event_recorder.config import RecordingConfig
from event_recorder.models import EventPaths


def has_minimum_free_disk(path: Path, minimum_free_disk_gb: float) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    required_bytes = minimum_free_disk_gb * 1024 * 1024 * 1024
    return free_bytes >= required_bytes


def create_event_paths(
    recording: RecordingConfig,
    started_at: datetime,
    event_id: str | None = None,
    part: int = 1,
) -> EventPaths:
    event_id = event_id or uuid.uuid4().hex[:8]
    day_dir = recording.output_directory / started_at.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S.%f")[:-3]
    offset = started_at.strftime("%z")
    part_suffix = f"_part{part:03d}" if part > 1 else ""
    stem = f"event_{timestamp}{offset}_{event_id}{part_suffix}"
    extension = recording.extension.lstrip(".")
    return EventPaths(
        event_id=event_id,
        part=part,
        partial_video_path=day_dir / f"{stem}_partial.{extension}",
        partial_audio_path=day_dir / f"{stem}_partial.wav",
        muxed_partial_video_path=day_dir / f"{stem}_muxed_partial.{extension}",
        final_video_path=day_dir / f"{stem}.{extension}",
        metadata_path=day_dir / f"{stem}.json",
    )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
