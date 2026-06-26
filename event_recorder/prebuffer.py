from __future__ import annotations

from collections import deque
from math import ceil, floor

from event_recorder.models import FramePacket


class PreEventBuffer:
    def __init__(
        self,
        output_fps: float,
        pre_event_seconds: float,
        max_prebuffer_mb: int,
        first_frame_nbytes: int,
    ) -> None:
        if output_fps <= 0:
            raise ValueError("output_fps must be positive.")
        if pre_event_seconds < 0:
            raise ValueError("pre_event_seconds cannot be negative.")
        if max_prebuffer_mb <= 0:
            raise ValueError("max_prebuffer_mb must be positive.")
        if first_frame_nbytes <= 0:
            raise ValueError("first_frame_nbytes must be positive.")

        max_by_seconds = max(1, ceil(output_fps * pre_event_seconds) + 2)
        max_bytes = max_prebuffer_mb * 1024 * 1024
        max_by_memory = max(1, floor(max_bytes / first_frame_nbytes))
        self.max_frames = min(max_by_seconds, max_by_memory)
        self.requested_seconds = pre_event_seconds
        self.output_fps = output_fps
        self._frames: deque[FramePacket] = deque(maxlen=self.max_frames)

    @property
    def actual_seconds_capacity(self) -> float:
        return self.max_frames / self.output_fps

    def add(self, packet: FramePacket) -> None:
        self._frames.append(packet)

    def frames_for_event(self, event_start_monotonic: float) -> list[FramePacket]:
        earliest = event_start_monotonic - self.requested_seconds
        return sorted(
            (
                packet
                for packet in self._frames
                if packet.captured_at_monotonic >= earliest
            ),
            key=lambda packet: packet.frame_id,
        )

    def frames_after(self, last_written_frame_id: int | None) -> list[FramePacket]:
        if last_written_frame_id is None:
            return sorted(self._frames, key=lambda packet: packet.frame_id)
        return sorted(
            (
                packet
                for packet in self._frames
                if packet.frame_id > last_written_frame_id
            ),
            key=lambda packet: packet.frame_id,
        )

    def __len__(self) -> int:
        return len(self._frames)
