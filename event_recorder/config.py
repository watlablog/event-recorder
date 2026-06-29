from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    source: int | str
    width: int
    height: int
    requested_fps: int
    fallback_fps: float
    reconnect_attempts: int
    reconnect_interval_seconds: float


@dataclass(frozen=True)
class ModelConfig:
    path: str
    target_classes: tuple[str, ...]
    confidence: float
    iou: float
    image_size: int
    device: str | None
    half: bool
    max_detection_fps: float


@dataclass(frozen=True)
class StartConfirmationConfig:
    window_results: int
    required_positive_results: int
    expire_seconds: float


@dataclass(frozen=True)
class RecordingConfig:
    output_directory: Path
    extension: str
    fourcc: str
    pre_event_seconds: float
    post_event_seconds: float
    max_clip_seconds: float
    max_prebuffer_mb: int
    minimum_free_disk_gb: float
    draw_boxes: bool = False
    draw_timestamp: bool = False


@dataclass(frozen=True)
class HealthConfig:
    detector_stale_seconds: float
    detector_failure_seconds: float


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool
    draw_boxes: bool
    show_fps: bool


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool
    device: int | str | None
    sample_rate: int
    channels: int
    fallback_to_video_only: bool


@dataclass(frozen=True)
class NightEnhancementConfig:
    enabled: bool
    contrast: float
    brightness: float
    gamma: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    start_confirmation: StartConfirmationConfig
    recording: RecordingConfig
    health: HealthConfig
    preview: PreviewConfig
    audio: AudioConfig
    night_enhancement: NightEnhancementConfig
    logging: LoggingConfig


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Copy config.example.yaml to config.yaml first."
        )

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load configuration files.") from exc

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> AppConfig:
    camera = _mapping(raw, "camera")
    model = _mapping(raw, "model")
    start = _mapping(raw, "start_confirmation")
    recording = _mapping(raw, "recording")
    health = _mapping(raw, "health")
    preview = _mapping(raw, "preview")
    audio = _mapping(raw, "audio")
    night_enhancement = _mapping(raw, "night_enhancement")
    logging = _mapping(raw, "logging")

    config = AppConfig(
        camera=CameraConfig(
            source=_camera_source(camera.get("source", 0)),
            width=_int(camera, "width", 1280),
            height=_int(camera, "height", 720),
            requested_fps=_int(camera, "requested_fps", 30),
            fallback_fps=_float(camera, "fallback_fps", 30.0),
            reconnect_attempts=_int(camera, "reconnect_attempts", 10),
            reconnect_interval_seconds=_float(
                camera, "reconnect_interval_seconds", 0.5
            ),
        ),
        model=ModelConfig(
            path=str(model.get("path", "yolo26n.pt")),
            target_classes=tuple(str(item) for item in model.get("target_classes", [])),
            confidence=_float(model, "confidence", 0.40),
            iou=_float(model, "iou", 0.70),
            image_size=_int(model, "image_size", 640),
            device=model.get("device"),
            half=bool(model.get("half", False)),
            max_detection_fps=_float(model, "max_detection_fps", 0),
        ),
        start_confirmation=StartConfirmationConfig(
            window_results=_int(start, "window_results", 3),
            required_positive_results=_int(start, "required_positive_results", 2),
            expire_seconds=_float(start, "expire_seconds", 2.0),
        ),
        recording=RecordingConfig(
            output_directory=Path(str(recording.get("output_directory", "recordings"))),
            extension=str(recording.get("extension", "mp4")).lstrip("."),
            fourcc=str(recording.get("fourcc", "mp4v")),
            pre_event_seconds=_float(recording, "pre_event_seconds", 1.0),
            post_event_seconds=_float(recording, "post_event_seconds", 3.0),
            max_clip_seconds=_float(recording, "max_clip_seconds", 600),
            max_prebuffer_mb=_int(recording, "max_prebuffer_mb", 256),
            minimum_free_disk_gb=_float(recording, "minimum_free_disk_gb", 2.0),
            draw_boxes=bool(recording.get("draw_boxes", False)),
            draw_timestamp=bool(recording.get("draw_timestamp", False)),
        ),
        health=HealthConfig(
            detector_stale_seconds=_float(health, "detector_stale_seconds", 5.0),
            detector_failure_seconds=_float(health, "detector_failure_seconds", 30.0),
        ),
        preview=PreviewConfig(
            enabled=bool(preview.get("enabled", True)),
            draw_boxes=bool(preview.get("draw_boxes", True)),
            show_fps=bool(preview.get("show_fps", True)),
        ),
        audio=AudioConfig(
            enabled=bool(audio.get("enabled", False)),
            device=_optional_device(audio.get("device")),
            sample_rate=_int(audio, "sample_rate", 48000),
            channels=_int(audio, "channels", 1),
            fallback_to_video_only=bool(audio.get("fallback_to_video_only", True)),
        ),
        night_enhancement=NightEnhancementConfig(
            enabled=bool(night_enhancement.get("enabled", False)),
            contrast=_float(night_enhancement, "contrast", 1.35),
            brightness=_float(night_enhancement, "brightness", 12.0),
            gamma=_float(night_enhancement, "gamma", 1.20),
        ),
        logging=LoggingConfig(level=str(logging.get("level", "INFO"))),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ConfigError("camera.width and camera.height must be positive.")
    if config.camera.requested_fps <= 0:
        raise ConfigError("camera.requested_fps must be positive.")
    if config.camera.fallback_fps <= 0:
        raise ConfigError("camera.fallback_fps must be positive.")
    if config.camera.reconnect_attempts < 0:
        raise ConfigError("camera.reconnect_attempts must be zero or greater.")
    if config.camera.reconnect_interval_seconds < 0:
        raise ConfigError("camera.reconnect_interval_seconds must be zero or greater.")

    if not config.model.path:
        raise ConfigError("model.path must not be empty.")
    if not config.model.target_classes:
        raise ConfigError("model.target_classes must contain at least one class.")
    if not 0 <= config.model.confidence <= 1:
        raise ConfigError("model.confidence must be between 0 and 1.")
    if not 0 <= config.model.iou <= 1:
        raise ConfigError("model.iou must be between 0 and 1.")
    if config.model.image_size <= 0:
        raise ConfigError("model.image_size must be positive.")
    if config.model.max_detection_fps < 0:
        raise ConfigError("model.max_detection_fps must be zero or greater.")

    start = config.start_confirmation
    if start.window_results <= 0:
        raise ConfigError("start_confirmation.window_results must be positive.")
    if start.required_positive_results <= 0:
        raise ConfigError(
            "start_confirmation.required_positive_results must be positive."
        )
    if start.required_positive_results > start.window_results:
        raise ConfigError(
            "start_confirmation.required_positive_results cannot exceed window_results."
        )
    if start.expire_seconds <= 0:
        raise ConfigError("start_confirmation.expire_seconds must be positive.")

    rec = config.recording
    if not rec.extension:
        raise ConfigError("recording.extension must not be empty.")
    if len(rec.fourcc) != 4:
        raise ConfigError("recording.fourcc must be exactly four characters.")
    if rec.pre_event_seconds < 0 or rec.post_event_seconds < 0:
        raise ConfigError("recording pre/post event seconds cannot be negative.")
    if rec.max_clip_seconds <= 0:
        raise ConfigError("recording.max_clip_seconds must be positive.")
    if rec.max_prebuffer_mb <= 0:
        raise ConfigError("recording.max_prebuffer_mb must be positive.")
    if rec.minimum_free_disk_gb < 0:
        raise ConfigError("recording.minimum_free_disk_gb cannot be negative.")

    if config.health.detector_stale_seconds <= 0:
        raise ConfigError("health.detector_stale_seconds must be positive.")
    if (
        config.health.detector_failure_seconds
        <= config.health.detector_stale_seconds
    ):
        raise ConfigError(
            "health.detector_failure_seconds must be greater than detector_stale_seconds."
        )

    if config.audio.sample_rate <= 0:
        raise ConfigError("audio.sample_rate must be positive.")
    if config.audio.channels <= 0:
        raise ConfigError("audio.channels must be positive.")

    night = config.night_enhancement
    if not 0.5 <= night.contrast <= 3.0:
        raise ConfigError("night_enhancement.contrast must be between 0.5 and 3.0.")
    if not -100.0 <= night.brightness <= 100.0:
        raise ConfigError(
            "night_enhancement.brightness must be between -100 and 100."
        )
    if not 0.5 <= night.gamma <= 3.0:
        raise ConfigError("night_enhancement.gamma must be between 0.5 and 3.0.")


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping.")
    return value


def _int(mapping: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer.") from exc


def _float(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number.") from exc


def _camera_source(value: Any) -> int | str:
    if isinstance(value, bool):
        raise ConfigError("camera.source must be an integer device index or path.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ConfigError("camera.source must not be empty.")
        return value
    raise ConfigError("camera.source must be an integer device index or path.")


def _optional_device(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError("audio.device must be an integer device index, name, or null.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return value
    raise ConfigError("audio.device must be an integer device index, name, or null.")
