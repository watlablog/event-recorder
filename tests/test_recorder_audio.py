from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from event_recorder.audio import AudioMuxError
from event_recorder.config import RecordingConfig
from event_recorder.models import EventPaths, StopReason
from event_recorder.recorder import VideoClipWriter


class FakeVideoWriter:
    def __init__(self, *args, **kwargs):
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        pass

    def release(self) -> None:
        self.released = True


class FakeCv2:
    VideoWriter = FakeVideoWriter

    @staticmethod
    def VideoWriter_fourcc(*args):
        return 0


def test_video_writer_falls_back_to_video_only_when_mux_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2)

    def fail_mux(*args, **kwargs):
        raise AudioMuxError("mux failed")

    monkeypatch.setattr("event_recorder.recorder.mux_audio_video", fail_mux)
    paths = EventPaths(
        event_id="event",
        part=1,
        partial_video_path=tmp_path / "event_partial.mp4",
        partial_audio_path=tmp_path / "event_partial.wav",
        muxed_partial_video_path=tmp_path / "event_muxed_partial.mp4",
        final_video_path=tmp_path / "event.mp4",
        metadata_path=tmp_path / "event.json",
    )
    writer = VideoClipWriter(
        paths=paths,
        recording=RecordingConfig(
            output_directory=tmp_path,
            extension="mp4",
            fourcc="mp4v",
            pre_event_seconds=1.0,
            post_event_seconds=3.0,
            max_clip_seconds=600,
            max_prebuffer_mb=256,
            minimum_free_disk_gb=0,
        ),
        source=0,
        model_path="model.pt",
        target_classes=("person",),
        fps=30.0,
        frame_size=(2, 2),
        started_at_wall_clock=datetime.now(timezone.utc),
        started_at_monotonic=1.0,
    )
    Path(paths.partial_video_path).write_bytes(b"video")
    Path(paths.partial_audio_path).write_bytes(b"audio")

    writer.close(
        StopReason.USER_SHUTDOWN,
        ended_at_wall_clock=datetime.now(timezone.utc),
        ended_at_monotonic=2.0,
        audio_path=Path(paths.partial_audio_path),
        audio_requested=True,
        audio_status="captured",
        audio_device=1,
        audio_sample_rate=48000,
        audio_channels=1,
    )

    metadata = json.loads(Path(paths.metadata_path).read_text(encoding="utf-8"))
    assert Path(paths.final_video_path).read_bytes() == b"video"
    assert metadata["audio_requested"] is True
    assert metadata["audio_recorded"] is False
    assert metadata["audio_status"] == "mux_failed"
