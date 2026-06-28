from __future__ import annotations

import queue
import subprocess
import time
import wave
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from event_recorder.config import AudioConfig


class AudioError(RuntimeError):
    pass


class AudioMuxError(AudioError):
    pass


@dataclass(frozen=True)
class MicrophoneCandidate:
    index: int
    label: str
    name: str


@dataclass(frozen=True)
class AudioBlock:
    captured_at_monotonic: float
    samples: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])


def discover_microphones(
    devices: Sequence[Mapping[str, Any]] | None = None,
) -> list[MicrophoneCandidate]:
    if devices is None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("sounddevice is required for microphone discovery.") from exc
        devices = sd.query_devices()

    microphones: list[MicrophoneCandidate] = []
    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0) or 0)
        if max_input_channels <= 0:
            continue
        name = str(device.get("name", f"Microphone {index}"))
        microphones.append(
            MicrophoneCandidate(
                index=index,
                label=f"Microphone {index} - {name}",
                name=name,
            )
        )
    return microphones


class AudioPreBuffer:
    def __init__(self, pre_event_seconds: float, sample_rate: int) -> None:
        if pre_event_seconds < 0:
            raise ValueError("pre_event_seconds cannot be negative.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self.pre_event_seconds = pre_event_seconds
        self.sample_rate = sample_rate
        self._blocks: deque[AudioBlock] = deque()

    def add(self, block: AudioBlock) -> None:
        self._blocks.append(block)
        self._trim(block.captured_at_monotonic)

    def blocks_for_event(self, event_start_monotonic: float) -> list[AudioBlock]:
        earliest = event_start_monotonic - self.pre_event_seconds
        return [
            block
            for block in self._blocks
            if block.captured_at_monotonic >= earliest
        ]

    def _trim(self, now_monotonic: float) -> None:
        earliest = now_monotonic - self.pre_event_seconds
        while self._blocks and self._blocks[0].captured_at_monotonic < earliest:
            self._blocks.popleft()

    def __len__(self) -> int:
        return len(self._blocks)


class AudioCapture:
    def __init__(
        self,
        config: AudioConfig,
        block_queue_size: int = 128,
    ) -> None:
        self.config = config
        self._queue: queue.Queue[AudioBlock] = queue.Queue(maxsize=block_queue_size)
        self._stream = None
        self._failure: str | None = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("sounddevice is required for audio capture.") from exc

        self._stream = sd.InputStream(
            device=self.config.device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def drain(self) -> list[AudioBlock]:
        if self._failure is not None:
            raise AudioError(self._failure)
        blocks: list[AudioBlock] = []
        while True:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                return blocks

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._failure = str(status)
        block = AudioBlock(
            captured_at_monotonic=time.monotonic(),
            samples=np.array(indata, dtype=np.int16, copy=True),
        )
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(block)


class AudioClipWriter:
    def __init__(self, path: Path, sample_rate: int, channels: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_written = 0
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = wave.open(str(path), "wb")
        self._writer.setnchannels(channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(sample_rate)

    def write(self, block: AudioBlock) -> None:
        if self._closed:
            return
        samples = _coerce_samples(block.samples, self.channels)
        self._writer.writeframes(samples.tobytes())
        self.frames_written += int(samples.shape[0])

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._closed = True


def build_ffmpeg_mux_command(
    ffmpeg_exe: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise AudioMuxError("imageio-ffmpeg is required for audio muxing.") from exc

    runner = runner or subprocess.run
    cmd = build_ffmpeg_mux_command(
        imageio_ffmpeg.get_ffmpeg_exe(),
        video_path,
        audio_path,
        output_path,
    )
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise AudioMuxError(stderr or "ffmpeg mux failed.")


def _coerce_samples(samples: np.ndarray, channels: int) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] > channels:
        array = array[:, :channels]
    elif array.shape[1] < channels:
        padding = np.zeros(
            (array.shape[0], channels - array.shape[1]),
            dtype=array.dtype,
        )
        array = np.concatenate([array, padding], axis=1)
    if array.dtype != np.int16:
        array = array.astype(np.int16)
    return np.ascontiguousarray(array)
