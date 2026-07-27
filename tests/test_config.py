from __future__ import annotations

import pytest

from voice_codex.config import load_startup_config, save_startup_config


def test_startup_config_round_trip(tmp_path) -> None:
    settings = {
        "microphone": "USB microphone",
        "tts": "on",
        "tts_provider": "piper",
        "them_output": "meeting-monitor",
        "playback_output": None,
        "codex_after": "both",
    }
    config = tmp_path / "voice.yaml"

    save_startup_config(config, settings)

    assert load_startup_config(config) == settings


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


def test_a_config_value_that_is_not_json_is_kept_as_text(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text('microphone: Blue Yeti\ntts: "on"\n', encoding="utf-8")

    assert load_startup_config(path) == {"microphone": "Blue Yeti", "tts": "on"}


def test_an_empty_config_value_reads_as_unset(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text("microphone:\n", encoding="utf-8")

    assert load_startup_config(path) == {"microphone": None}


def test_comments_and_blank_lines_are_ignored(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text('# a comment\n\ntts: "off"\n', encoding="utf-8")

    assert load_startup_config(path) == {"tts": "off"}


def test_a_line_without_a_separator_is_rejected(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text("microphone\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="line 1"):
        load_startup_config(path)


def test_an_unknown_config_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text("volume: 11\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unknown startup config key 'volume'"):
        load_startup_config(path)


def test_a_config_that_cannot_be_written_is_reported(tmp_path) -> None:
    unwritable = tmp_path / "missing-directory" / "voice.yaml"

    with pytest.raises(RuntimeError, match="Could not save startup config"):
        save_startup_config(unwritable, {"microphone": "Yeti"})


def test_saved_settings_reload_to_the_same_values(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    settings = {
        "microphone": "Blue Yeti",
        "tts": "on",
        "tts_provider": "edge",
        "them_output": "isolated",
        "playback_output": None,
        "codex_after": "both",
    }

    save_startup_config(path, settings)

    assert load_startup_config(path) == settings


def test_a_device_name_containing_a_colon_survives_the_round_trip(tmp_path) -> None:
    """Only the first colon separates key from value; the rest is the value."""
    path = tmp_path / "voice.yaml"
    path.write_text("microphone: Scarlett 2i2: USB Audio\n", encoding="utf-8")

    assert load_startup_config(path) == {"microphone": "Scarlett 2i2: USB Audio"}


def test_parsing_continues_past_an_unset_value(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text('microphone:\ntts: "on"\n', encoding="utf-8")

    assert load_startup_config(path) == {"microphone": None, "tts": "on"}


def test_parsing_continues_past_comments_between_settings(tmp_path) -> None:
    path = tmp_path / "voice.yaml"
    path.write_text(
        "microphone: Yeti\n\n# which turns get a reply\ncodex_after: both\n",
        encoding="utf-8",
    )

    assert load_startup_config(path) == {"microphone": "Yeti", "codex_after": "both"}


def test_an_unreadable_config_names_the_file_it_could_not_read(tmp_path) -> None:
    missing = tmp_path / "absent.yaml"

    with pytest.raises(
        RuntimeError, match=r"Could not read startup config '[^']*absent\.yaml'"
    ):
        load_startup_config(missing)


def test_an_unwritable_config_names_the_file_it_could_not_write(tmp_path) -> None:
    unwritable = tmp_path / "missing-directory" / "voice.yaml"

    with pytest.raises(
        RuntimeError, match=r"Could not save startup config '[^']*voice\.yaml'"
    ):
        save_startup_config(unwritable, {"microphone": "Yeti"})


def test_a_saved_config_explains_itself_to_whoever_edits_it(tmp_path) -> None:
    path = tmp_path / "voice.yaml"

    save_startup_config(path, {"microphone": "Yeti"})
    header = path.read_text(encoding="utf-8").splitlines()[:2]

    assert header == [
        "# Voice Codex startup choices",
        "# Command-line options override these values.",
    ]
