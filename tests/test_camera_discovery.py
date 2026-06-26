from __future__ import annotations

from event_recorder.camera_discovery import discover_cameras


class FakeCapture:
    released_indices: list[int] = []

    def __init__(self, index: int, opened_indices: set[int]) -> None:
        self.index = index
        self.opened_indices = opened_indices

    def isOpened(self) -> bool:
        return self.index in self.opened_indices

    def release(self) -> None:
        self.released_indices.append(self.index)


def test_discover_cameras_returns_opened_indices():
    FakeCapture.released_indices = []
    opened = {0, 2}

    cameras = discover_cameras(
        max_index=4,
        video_capture_factory=lambda index: FakeCapture(index, opened),
    )

    assert [(camera.index, camera.label) for camera in cameras] == [
        (0, "Camera 0"),
        (2, "Camera 2"),
    ]
    assert FakeCapture.released_indices == [0, 1, 2, 3]
