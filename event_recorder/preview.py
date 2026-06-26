from __future__ import annotations

from event_recorder.config import PreviewConfig
from event_recorder.models import DetectedObject, RecorderStatus


class PreviewWindow:
    def __init__(self, config: PreviewConfig) -> None:
        self.config = config
        self._cv2 = None

    def show(
        self,
        frame,
        detections: tuple[DetectedObject, ...],
        status: RecorderStatus,
        camera_fps: float | None = None,
        detector_fps: float | None = None,
        recording_elapsed: float | None = None,
    ) -> bool:
        if not self.config.enabled:
            return False
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is not installed.") from exc

        self._cv2 = cv2
        display = frame.copy()
        if self.config.draw_boxes:
            for detected in detections:
                x1, y1, x2, y2 = (int(value) for value in detected.xyxy)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"{detected.class_name} {detected.confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        lines = [status.value]
        if recording_elapsed is not None:
            lines.append(f"REC {recording_elapsed:.1f}s")
        if self.config.show_fps:
            if camera_fps is not None:
                lines.append(f"camera {camera_fps:.1f} fps")
            if detector_fps is not None:
                lines.append(f"detector {detector_fps:.1f} fps")
        y = 28
        for line in lines:
            cv2.putText(
                display,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26

        cv2.imshow("event-recorder", display)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
