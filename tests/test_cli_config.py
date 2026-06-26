from __future__ import annotations

from pathlib import Path

import event_recorder.__main__ as cli


def test_default_config_uses_project_config_when_present(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    example = tmp_path / "config.example.yaml"
    config.write_text("camera: {}\n", encoding="utf-8")
    example.write_text("camera: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    assert cli.resolve_config_path(None) == config


def test_default_config_falls_back_to_example(monkeypatch, tmp_path):
    example = tmp_path / "config.example.yaml"
    example.write_text("camera: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    assert cli.resolve_config_path(None) == example


def test_relative_explicit_config_can_resolve_from_project_root(
    monkeypatch, tmp_path
):
    config = tmp_path / "custom.yaml"
    config.write_text("camera: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    assert cli.resolve_config_path(Path("custom.yaml")) == config
