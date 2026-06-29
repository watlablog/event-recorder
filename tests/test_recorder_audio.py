from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from event_recorder.audio import AudioMuxError
from event_recorder.config import NightEnhancementConfig, RecordingConfig
from event_recorder.models import DetectedObject, EventPaths, FramePacket, StopReason
from event_recorder.recorder import VideoClipWriter


def _night_config(enabled: bool = False) -> NightEnhancementConfig:
    return NightEnhancementConfig(
        enabled=enabled,
        contrast=1.35,
        brightness=12.0,
        gamma=1.20,
    )


class FakeVideoWriter:
    written_frames = []

    def __init__(self, *args, **kwargs):
        self.released = False
        FakeVideoWriter.written_frames = []

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        FakeVideoWriter.written_frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class FakeCv2:
    VideoWriter = FakeVideoWriter
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    rectangle_calls = []
    text_calls = []

    @staticmethod
    def VideoWriter_fourcc(*args):
        return 0

    @staticmethod
    def rectangle(frame, pt1, pt2, color, thickness):
        FakeCv2.rectangle_calls.append((pt1, pt2, color, thickness))
        frame[:, :, 1] = 255

    @staticmethod
    def getTextSize(text, font, scale, thickness):
        return (len(text) * 10, 20), 5

    @staticmethod
    def putText(frame, text, org, font, scale, color, thickness, line_type):
        FakeCv2.text_calls.append((text, org, color, thickness))
        frame[:, :, 2] = 255


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
        night_enhancement=_night_config(enabled=True),
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
    assert metadata["draw_timestamp"] is False
    assert metadata["night_enhancement_enabled"] is True
    assert metadata["night_enhancement_contrast"] == 1.35
    assert metadata["night_enhancement_brightness"] == 12.0
    assert metadata["night_enhancement_gamma"] == 1.20


def test_video_writer_draws_boxes_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2)
    FakeCv2.rectangle_calls = []
    FakeCv2.text_calls = []
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
            draw_boxes=True,
        ),
        source=0,
        model_path="model.pt",
        target_classes=("person",),
        night_enhancement=_night_config(),
        fps=30.0,
        frame_size=(4, 4),
        started_at_wall_clock=datetime.now(timezone.utc),
        started_at_monotonic=1.0,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    writer.write(
        FramePacket(
            frame_id=1,
            captured_at_monotonic=1.0,
            captured_at_wall_clock=datetime.now(timezone.utc),
            frame=frame,
        ),
        (
            DetectedObject(
                class_id=0,
                class_name="person",
                confidence=0.9,
                xyxy=(0.0, 0.0, 3.0, 3.0),
            ),
        ),
    )

    assert FakeCv2.rectangle_calls
    assert FakeVideoWriter.written_frames[0][:, :, 1].max() == 255
    assert frame[:, :, 1].max() == 0


def test_video_writer_draws_timestamp_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2)
    FakeCv2.rectangle_calls = []
    FakeCv2.text_calls = []
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
            draw_timestamp=True,
        ),
        source=0,
        model_path="model.pt",
        target_classes=("person",),
        night_enhancement=_night_config(),
        fps=30.0,
        frame_size=(640, 480),
        started_at_wall_clock=datetime(2026, 6, 29, tzinfo=timezone.utc),
        started_at_monotonic=1.0,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    writer.write(
        FramePacket(
            frame_id=1,
            captured_at_monotonic=1.0,
            captured_at_wall_clock=datetime(2026, 6, 29, 12, 34, 56, tzinfo=timezone.utc),
            frame=frame,
        )
    )

    assert [call[2] for call in FakeCv2.text_calls] == [
        (0, 0, 0),
        (255, 255, 255),
    ]
    assert FakeCv2.text_calls[0][0] == "2026-06-29 12:34:56"
    assert FakeVideoWriter.written_frames[0][:, :, 2].max() == 255
    assert frame[:, :, 2].max() == 0
