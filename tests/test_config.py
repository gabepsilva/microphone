from __future__ import annotations

import pytest

from voice_codex.config import (
    STARTUP_CONFIG_KEYS,
    StartupConfigFile,
    load_startup_config,
    save_startup_config,
)


def test_startup_config_round_trip(tmp_path) -> None:
    settings = {
        "microphone": "USB microphone",
        "tts": "on",
        "tts_provider": "piper",
        "them_output": "meeting-monitor",
        "playback_output": None,
        "codex_after": "both",
        "turn_silence": 3.0,
        "codex_model": "gpt-5.6-luna",
        "codex_reasoning": "low",
        # A boolean has to survive the round trip as a boolean: the loader
        # reads it back through JSON, and "true" would be a nine-tenths-right
        # config key that silently reads as truthy either way.
        "codex_fast": False,
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
        "turn_silence": 3.0,
        "codex_model": "gpt-5.6-luna",
        "codex_reasoning": "low",
        "codex_fast": True,
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


class RecordingSave:
    """Capture what would be written, and optionally refuse to write it."""

    def __init__(self, error=None):
        self.writes: list[tuple[object, dict]] = []
        self.error = error

    def __call__(self, path, settings):
        if self.error is not None:
            raise RuntimeError(self.error)
        self.writes.append((path, dict(settings)))


def stored(**overrides):
    settings = dict.fromkeys(STARTUP_CONFIG_KEYS)
    settings.update(overrides)
    return settings


def test_a_changed_setting_is_written_through(tmp_path) -> None:
    save = RecordingSave()
    config = StartupConfigFile(tmp_path / "voice.yaml", stored(tts="on"), save=save)

    assert config.record("tts", "off") is True
    assert save.writes[0][0] == tmp_path / "voice.yaml"
    assert save.writes[0][1]["tts"] == "off"


def test_a_setting_changed_to_what_it_already_was_is_not_rewritten() -> None:
    save = RecordingSave()
    config = StartupConfigFile("voice.yaml", stored(tts="on"), save=save)

    assert config.record("tts", "on") is False
    assert save.writes == []


def test_every_recorded_change_accumulates_in_the_file() -> None:
    """The file describes the whole session, not only the last thing touched."""
    save = RecordingSave()
    config = StartupConfigFile("voice.yaml", stored(), save=save)

    config.record("tts_provider", "edge")
    config.record("turn_silence", 1.25)

    assert save.writes[-1][1]["tts_provider"] == "edge"
    assert save.writes[-1][1]["turn_silence"] == 1.25


def test_the_store_does_not_share_the_mapping_it_was_given() -> None:
    original = stored(tts="on")
    config = StartupConfigFile("voice.yaml", original, save=RecordingSave())

    config.record("tts", "off")

    assert original["tts"] == "on"


def test_a_key_the_file_cannot_hold_is_refused() -> None:
    """A setting the file cannot store would be silently lost at exit."""
    config = StartupConfigFile("voice.yaml", stored(), save=RecordingSave())

    with pytest.raises(RuntimeError, match="'muted' is not a startup config key"):
        config.record("muted", True)


def test_a_file_that_cannot_be_written_is_reported_once(capsys) -> None:
    save = RecordingSave(error="disk full")
    config = StartupConfigFile("voice.yaml", stored(), save=save, stream=None)

    assert config.record("tts", "off") is False
    assert config.record("tts", "on") is False

    assert capsys.readouterr().err.count("Startup config will not be updated") == 1


def test_a_change_that_could_not_be_saved_is_still_the_setting_in_force() -> None:
    """The session already applied it; the file is what failed, not the change."""
    config = StartupConfigFile(
        "voice.yaml", stored(tts="on"), save=RecordingSave(error="read-only")
    )

    config.record("tts", "off")

    assert config.settings["tts"] == "off"


def test_the_saved_keys_are_exactly_the_keys_that_can_be_loaded(tmp_path) -> None:
    """A key the file writes but nothing reads would be saved and ignored."""
    path = tmp_path / "voice.yaml"
    save_startup_config(path, dict.fromkeys(STARTUP_CONFIG_KEYS, "x"))

    assert set(load_startup_config(path)) == set(STARTUP_CONFIG_KEYS)
