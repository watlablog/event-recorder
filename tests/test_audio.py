from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from event_recorder.audio import (
    AudioBlock,
    AudioPreBuffer,
    build_ffmpeg_mux_command,
    discover_microphones,
    mux_audio_video,
)


def test_discover_microphones_returns_input_devices_only():
    devices = [
        {"name": "Speaker", "max_input_channels": 0},
        {"name": "Built-in Mic", "max_input_channels": 2},
        {"name": "USB Mic", "max_input_channels": 1},
    ]

    microphones = discover_microphones(devices)

    assert [(item.index, item.label, item.name) for item in microphones] == [
        (1, "Microphone 1 - Built-in Mic", "Built-in Mic"),
        (2, "Microphone 2 - USB Mic", "USB Mic"),
    ]


def test_audio_prebuffer_returns_blocks_for_event_window():
    buffer = AudioPreBuffer(pre_event_seconds=1.0, sample_rate=48000)
    for index, at in enumerate([0.0, 0.4, 1.0, 1.4], start=1):
        samples = np.full((2, 1), index, dtype=np.int16)
        buffer.add(AudioBlock(captured_at_monotonic=at, samples=samples))

    blocks = buffer.blocks_for_event(event_start_monotonic=1.4)

    assert [int(block.samples[0, 0]) for block in blocks] == [2, 3, 4]


def test_build_ffmpeg_mux_command_uses_video_copy_and_aac():
    cmd = build_ffmpeg_mux_command(
        "ffmpeg",
        Path("video.mp4"),
        Path("audio.wav"),
        Path("out.mp4"),
    )

    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "video.mp4",
        "-i",
        "audio.wav",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "out.mp4",
    ]


def test_mux_audio_video_raises_on_ffmpeg_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "imageio_ffmpeg.get_ffmpeg_exe",
        lambda: "ffmpeg",
    )

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stderr="bad mux")

    try:
        mux_audio_video(
            tmp_path / "video.mp4",
            tmp_path / "audio.wav",
            tmp_path / "out.mp4",
            runner=runner,
        )
    except Exception as exc:
        assert "bad mux" in str(exc)
    else:
        raise AssertionError("Expected mux failure")
