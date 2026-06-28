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
    use_default_factory = video_capture_factory is None
    if video_capture_factory is None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for camera discovery.") from exc

        video_capture_factory = cv2.VideoCapture

    camera_names = _camera_names_by_index() if use_default_factory else []
    candidates: list[CameraCandidate] = []
    for index in range(max_index):
        capture = video_capture_factory(index)
        try:
            if capture.isOpened():
                label = f"Camera {index}"
                if index < len(camera_names) and camera_names[index]:
                    label = f"{label} - {camera_names[index]}"
                candidates.append(CameraCandidate(index=index, label=label))
        finally:
            capture.release()
    return candidates


def _camera_names_by_index() -> list[str]:
    try:
        from PyQt5.QtMultimedia import QCameraInfo
    except ImportError:
        return []
    names: list[str] = []
    for camera in QCameraInfo.availableCameras():
        names.append(camera.description() or camera.deviceName())
    return names
