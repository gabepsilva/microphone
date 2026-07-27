"""Dependency-free startup configuration handling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

STARTUP_CONFIG_KEYS = (
    "microphone",
    "tts",
    "tts_provider",
    "them_output",
    "playback_output",
    "codex_after",
)


def load_startup_config(filename: str | Path) -> dict[str, object]:
    """Load the flat YAML subset emitted by :func:`save_startup_config`."""
    settings: dict[str, object] = {}
    try:
        with Path(filename).open(encoding="utf-8") as config_file:
            lines = config_file.readlines()
    except OSError as error:
        raise RuntimeError(
            f"Could not read startup config {str(filename)!r}: {error}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise RuntimeError(
                f"Invalid startup config line {line_number}: expected key: value"
            )
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in STARTUP_CONFIG_KEYS:
            raise RuntimeError(
                f"Unknown startup config key {key!r} on line {line_number}."
            )
        if not value:
            settings[key] = None
            continue
        try:
            settings[key] = json.loads(value)
        except json.JSONDecodeError:
            settings[key] = value
    return settings


def save_startup_config(filename: str | Path, settings: Mapping[str, object]) -> None:
    """Save prompt answers as dependency-free, human-editable YAML."""
    lines = [
        "# Voice Codex startup choices",
        "# Command-line options override these values.",
    ]
    for key in STARTUP_CONFIG_KEYS:
        value = settings.get(key)
        encoded = "null" if value is None else json.dumps(value)
        lines.append(f"{key}: {encoded}")
    try:
        Path(filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Could not save startup config {str(filename)!r}: {error}"
        ) from error
