"""Dependency-free startup configuration handling.

The file is both what a session starts from and where it records what the
sidebar changed, so the two lists have to stay the same list: a setting the
interface can change but the file cannot hold would be lost at exit, and a key
the file holds but nothing resolves would be read and ignored.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

STARTUP_CONFIG_KEYS = (
    "microphone",
    "tts_provider",
    "audio_stream",
    "tts_output",
    "taga_after",
    "turn_silence",
    "codex_model",
    "codex_reasoning",
    "codex_fast",
    "codex_prefire",
)


def load_startup_config(
    filename: str | Path, *, missing_ok: bool = False
) -> dict[str, object]:
    """Load the flat YAML subset emitted by :func:`save_startup_config`.

    A first run has no file yet. Callers that supply ``missing_ok`` get an
    empty layer in that one case; an existing file that cannot be read still
    fails loudly rather than looking like an intentional empty config.
    """
    settings: dict[str, object] = {}
    try:
        with Path(filename).open(encoding="utf-8") as config_file:
            lines = config_file.readlines()
    except FileNotFoundError:
        if missing_ok:
            return settings
        raise RuntimeError(
            f"Could not read startup config {str(filename)!r}: file not found"
        ) from None
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
        "# TagAlong startup choices",
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


class StartupConfigFile:
    """Keep the config file matching the choices currently in force.

    Every sidebar change is written straight through, because there is no
    later moment to rely on: the session ends on Ctrl-C or a closed terminal
    as often as it ends on a clean quit, and a write deferred to shutdown is a
    write that does not happen. The file is a handful of short lines, so the
    cost of rewriting it on each change is not worth batching.

    A file that cannot be written is reported once. Repeating it on every
    keystroke would bury the transcript under the same sentence.
    """

    def __init__(self, path, settings, save=None, stream=None):
        self.path = path
        self.settings = dict(settings)
        self._save = save_startup_config if save is None else save
        self._stream = stream
        self._reported = False
        self._lock = threading.Lock()

    def record(self, key, value):
        """Store a changed setting; report whether the file was rewritten."""
        if key not in STARTUP_CONFIG_KEYS:
            raise RuntimeError(f"{key!r} is not a startup config key.")
        # Applied device selections arrive from reconciler threads while the
        # TUI can save another setting. Keep update plus whole-file rewrite one
        # transaction so concurrent records cannot interleave or lose a value.
        with self._lock:
            if self.settings.get(key) == value:
                return False
            self.settings[key] = value
            return self._write()

    def _write(self):
        try:
            self._save(self.path, self.settings)
        except RuntimeError as error:
            if not self._reported:
                self._reported = True
                print(
                    f"\nStartup config will not be updated: {error}",
                    file=sys.stderr if self._stream is None else self._stream,
                    flush=True,
                )
            return False
        return True
