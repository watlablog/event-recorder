from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from event_recorder.config import ModelConfig
from event_recorder.models import DetectedObject, DetectionResult, FramePacket

LOGGER = logging.getLogger(__name__)


class ClassMappingError(ValueError):
    pass


class DetectorWorkerError(RuntimeError):
    pass


def normalize_class_name(name: str) -> str:
    return name.strip().casefold()


def resolve_target_class_ids(
    model_names: Mapping[int, str] | Sequence[str],
    target_classes: Iterable[str],
) -> tuple[int, ...]:
    if isinstance(model_names, Mapping):
        id_to_name = {int(class_id): str(name) for class_id, name in model_names.items()}
    else:
        id_to_name = {class_id: str(name) for class_id, name in enumerate(model_names)}

    name_to_id: dict[str, int] = {}
    for class_id, name in id_to_name.items():
        normalized = normalize_class_name(name)
        if normalized and normalized not in name_to_id:
            name_to_id[normalized] = class_id

    resolved: list[int] = []
    missing: list[str] = []
    for target in target_classes:
        normalized = normalize_class_name(str(target))
        if normalized in name_to_id:
            resolved.append(name_to_id[normalized])
        else:
            missing.append(str(target))

    if missing:
        available = ", ".join(
            name for _, name in sorted(id_to_name.items(), key=lambda item: item[0])
        )
        missing_text = ", ".join(missing)
        raise ClassMappingError(
            f"Target class not found: {missing_text}. Available classes: {available}"
        )

    return tuple(resolved)


def put_latest(input_queue: queue.Queue[FramePacket], packet: FramePacket) -> None:
    try:
        input_queue.put_nowait(packet)
        return
    except queue.Full:
        pass

    try:
        input_queue.get_nowait()
    except queue.Empty:
        pass
    input_queue.put_nowait(packet)


def put_bounded_drop_oldest(output_queue: queue.Queue[Any], item: Any) -> None:
    try:
        output_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    output_queue.put_nowait(item)


@dataclass(frozen=True)
class DetectorFailure:
    message: str
    exception: BaseException | None = None


class DetectorWorker(threading.Thread):
    def __init__(
        self,
        config: ModelConfig,
        input_queue: queue.Queue[FramePacket],
        result_queue: queue.Queue[DetectionResult],
        failure_queue: queue.Queue[DetectorFailure],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="yolo-detector", daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.failure_queue = failure_queue
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            LOGGER.exception("Detector worker failed")
            put_bounded_drop_oldest(
                self.failure_queue,
                DetectorFailure(message=str(exc), exception=exc),
            )

    def _run(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorWorkerError("ultralytics is not installed.") from exc

        model = YOLO(self.config.path)
        target_class_ids = resolve_target_class_ids(
            model.names, self.config.target_classes
        )
        LOGGER.info("Resolved target class IDs: %s", target_class_ids)
        min_interval = (
            1.0 / self.config.max_detection_fps
            if self.config.max_detection_fps > 0
            else 0.0
        )
        last_started_at = 0.0

        while not self.stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if min_interval:
                elapsed = time.monotonic() - last_started_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            last_started_at = time.monotonic()

            results = model.predict(
                source=packet.frame,
                classes=list(target_class_ids),
                conf=self.config.confidence,
                iou=self.config.iou,
                imgsz=self.config.image_size,
                device=self.config.device,
                half=self.config.half,
                verbose=False,
            )
            result = _to_detection_result(
                packet,
                results,
                model.names,
                allowed_class_ids=target_class_ids,
            )
            put_bounded_drop_oldest(self.result_queue, result)


def _to_detection_result(
    packet: FramePacket,
    results: Any,
    model_names: Mapping[int, str] | Sequence[str],
    allowed_class_ids: Iterable[int] | None = None,
) -> DetectionResult:
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return DetectionResult(
            frame_id=packet.frame_id,
            captured_at_monotonic=packet.captured_at_monotonic,
            detected=False,
            detections=(),
        )

    boxes = results[0].boxes
    class_ids = _to_list(boxes.cls)
    confidences = _to_list(boxes.conf)
    coordinates = boxes.xyxy.tolist()
    allowed = set(allowed_class_ids) if allowed_class_ids is not None else None
    detections: list[DetectedObject] = []
    for class_id_raw, confidence_raw, xyxy_raw in zip(
        class_ids, confidences, coordinates
    ):
        class_id = int(class_id_raw)
        if allowed is not None and class_id not in allowed:
            continue
        detections.append(
            DetectedObject(
                class_id=class_id,
                class_name=_class_name(model_names, class_id),
                confidence=float(confidence_raw),
                xyxy=tuple(float(value) for value in xyxy_raw),
            )
        )

    return DetectionResult(
        frame_id=packet.frame_id,
        captured_at_monotonic=packet.captured_at_monotonic,
        detected=bool(detections),
        detections=tuple(detections),
    )


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _class_name(model_names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    if isinstance(model_names, Mapping):
        return str(model_names.get(class_id, class_id))
    try:
        return str(model_names[class_id])
    except IndexError:
        return str(class_id)
