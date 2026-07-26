from __future__ import annotations

from pathlib import Path

import pytest

from voice_codex.config import load_startup_config, save_startup_config


def test_startup_config_round_trip(tmp_path) -> None:
    settings = {
        "microphone": "USB microphone",
        "tts": "on",
        "them_output": "meeting-monitor",
        "playback_output": None,
        "codex_after": "both",
    }
    config = tmp_path / "voice.yaml"

    save_startup_config(config, settings)

    assert load_startup_config(config) == settings


def test_example_config_is_valid_for_an_interactive_first_run() -> None:
    config = Path(__file__).resolve().parents[1] / "voice.example.yaml"

    assert load_startup_config(config) == {
        "microphone": None,
        "tts": None,
        "them_output": None,
        "playback_output": None,
        "codex_after": None,
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("missing delimiter\n", "expected key: value"),
        ("unknown: value\n", "Unknown startup config key"),
    ],
)
def test_startup_config_rejects_invalid_input(tmp_path, contents, message) -> None:
    config = tmp_path / "voice.yaml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_startup_config(config)
