from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from event_recorder.detector import normalize_class_name

DEFAULT_TARGET_CLASSES = ("person", "car")


@dataclass(frozen=True)
class DetectionClass:
    class_id: int
    name: str


def model_classes_from_names(
    model_names: Mapping[int, str] | Sequence[str],
) -> tuple[DetectionClass, ...]:
    if isinstance(model_names, Mapping):
        items = ((int(class_id), str(name)) for class_id, name in model_names.items())
    else:
        items = ((class_id, str(name)) for class_id, name in enumerate(model_names))
    return tuple(
        DetectionClass(class_id=class_id, name=name)
        for class_id, name in sorted(items, key=lambda item: item[0])
    )


def default_selected_classes(
    classes: tuple[DetectionClass, ...],
    defaults: tuple[str, ...] = DEFAULT_TARGET_CLASSES,
) -> tuple[str, ...]:
    by_normalized = {normalize_class_name(item.name): item.name for item in classes}
    selected = [
        by_normalized[normalize_class_name(default)]
        for default in defaults
        if normalize_class_name(default) in by_normalized
    ]
    return tuple(selected)


def load_model_classes(model_path: str) -> tuple[DetectionClass, ...]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to load model metadata.") from exc

    model = YOLO(model_path)
    return model_classes_from_names(model.names)
