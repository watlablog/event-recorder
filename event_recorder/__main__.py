from __future__ import annotations

import argparse
from pathlib import Path

from event_recorder.app import run

DEFAULT_CONFIG_NAME = "config.yaml"
EXAMPLE_CONFIG_NAME = "config.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record event clips when configured YOLO classes are detected."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to YAML configuration file. Defaults to project config.yaml, "
            "or config.example.yaml if config.yaml does not exist."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(resolve_config_path(args.config))


def resolve_config_path(config_path: Path | None) -> Path:
    if config_path is None:
        return default_config_path()
    if config_path.is_absolute() or config_path.exists():
        return config_path

    project_relative = project_root() / config_path
    if project_relative.exists():
        return project_relative
    return config_path


def default_config_path() -> Path:
    root = project_root()
    local_config = root / DEFAULT_CONFIG_NAME
    if local_config.exists():
        return local_config
    return root / EXAMPLE_CONFIG_NAME


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
