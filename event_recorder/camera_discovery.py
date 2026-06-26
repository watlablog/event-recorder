from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraCandidate:
    index: int
    label: str


def discover_cameras(
    max_index: int = 10,
    video_capture_factory: Callable[[int], CaptureLike] | None = None,
) -> list[CameraCandidate]:
    if max_index <= 0:
        return []
    if video_capture_factory is None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for camera discovery.") from exc

        video_capture_factory = cv2.VideoCapture

    candidates: list[CameraCandidate] = []
    for index in range(max_index):
        capture = video_capture_factory(index)
        try:
            if capture.isOpened():
                candidates.append(CameraCandidate(index=index, label=f"Camera {index}"))
        finally:
            capture.release()
    return candidates
