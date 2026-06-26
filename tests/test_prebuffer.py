from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from event_recorder.models import FramePacket
from event_recorder.prebuffer import PreEventBuffer


def _packet(frame_id: int, at: float) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic=at,
        captured_at_wall_clock=datetime.now(timezone.utc),
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_prebuffer_keeps_frames_in_frame_id_order_for_event():
    buffer = PreEventBuffer(
        output_fps=2.0,
        pre_event_seconds=1.0,
        max_prebuffer_mb=1,
        first_frame_nbytes=12,
    )
    for frame_id, at in [(1, 0.0), (2, 0.5), (3, 1.0), (4, 1.5)]:
        buffer.add(_packet(frame_id, at))

    frames = buffer.frames_for_event(event_start_monotonic=1.5)

    assert [frame.frame_id for frame in frames] == [2, 3, 4]


def test_prebuffer_applies_memory_limit():
    buffer = PreEventBuffer(
        output_fps=30.0,
        pre_event_seconds=10.0,
        max_prebuffer_mb=1,
        first_frame_nbytes=512 * 1024,
    )

    assert buffer.max_frames == 2
    for frame_id in range(1, 5):
        buffer.add(_packet(frame_id, float(frame_id)))

    assert [frame.frame_id for frame in buffer.frames_after(None)] == [3, 4]


def test_frames_after_prevents_duplicate_writes():
    buffer = PreEventBuffer(
        output_fps=30.0,
        pre_event_seconds=1.0,
        max_prebuffer_mb=1,
        first_frame_nbytes=12,
    )
    for frame_id in range(1, 5):
        buffer.add(_packet(frame_id, float(frame_id)))

    assert [frame.frame_id for frame in buffer.frames_after(2)] == [3, 4]
