"""Entry points, catalog parsing, and startup-config error paths."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ENTRYPOINTS = ["voice-codex.py", "voice-codex-tui.py"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_each_entrypoint_launches_the_configured_application(
    monkeypatch, script
) -> None:
    called: list[bool] = []
    monkeypatch.setattr("voice_codex.cli.main", lambda: called.append(True))

    runpy.run_path(str(ROOT / script), run_name="__main__")

    assert called == [True]


def test_a_keyboard_interrupt_stops_without_a_traceback(
    voice, monkeypatch, capsys
) -> None:
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(voice, "main", interrupted)

    voice.run_entrypoint()

    assert "Stopped." in capsys.readouterr().out


def test_a_startup_failure_is_reported_and_exits_nonzero(
    voice, monkeypatch, capsys
) -> None:
    def failed():
        raise RuntimeError("No audio input devices were found.")

    monkeypatch.setattr(voice, "main", failed)

    with pytest.raises(SystemExit, match="1"):
        voice.run_entrypoint()

    assert "Error: No audio input devices were found." in capsys.readouterr().err


def catalog(*models):
    return {"models": list(models)}


def model(**overrides):
    entry = {
        "slug": "gpt-5.6-luna",
        "display_name": "Luna",
        "visibility": "list",
        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
        "default_reasoning_level": "high",
        "priority": 1,
    }
    entry.update(overrides)
    return entry


def test_a_usable_model_is_parsed_with_its_efforts(voice) -> None:
    (option,) = voice._parse_codex_model_catalog(catalog(model()))

    assert option.slug == "gpt-5.6-luna"
    assert option.label == "Luna"
    assert option.efforts == ("low", "high")
    assert option.default_effort == "high"


def test_models_are_ordered_by_priority_then_label(voice) -> None:
    options = voice._parse_codex_model_catalog(
        catalog(
            model(slug="c", display_name="Third", priority=9),
            model(slug="b", display_name="Beta", priority=1),
            model(slug="a", display_name="Alpha", priority=1),
        )
    )

    assert [option.label for option in options] == ["Alpha", "Beta", "Third"]


def test_a_model_without_a_priority_sorts_last(voice) -> None:
    options = voice._parse_codex_model_catalog(
        catalog(
            model(slug="b", display_name="No priority", priority=None),
            model(slug="a", display_name="Has priority", priority=5),
        )
    )

    assert [option.label for option in options] == ["Has priority", "No priority"]


def test_a_missing_display_name_falls_back_to_the_slug(voice) -> None:
    (option,) = voice._parse_codex_model_catalog(catalog(model(display_name=None)))

    assert option.label == "gpt-5.6-luna"


def test_an_unlisted_default_effort_falls_back_to_the_first(voice) -> None:
    (option,) = voice._parse_codex_model_catalog(
        catalog(model(default_reasoning_level="extreme"))
    )

    assert option.default_effort == "low"


@pytest.mark.parametrize(
    "overrides",
    [
        {"visibility": "hidden"},
        {"supported_in_api": False},
        {"slug": ""},
        {"slug": None},
        {"supported_reasoning_levels": []},
        {"supported_reasoning_levels": "not a list"},
        {"supported_reasoning_levels": [{"effort": ""}, {"not": "an effort"}, 7]},
    ],
)
def test_an_unusable_model_is_skipped(voice, overrides) -> None:
    assert voice._parse_codex_model_catalog(catalog(model(**overrides))) == []


@pytest.mark.parametrize(
    "payload",
    ["not a dict", None, 7, {}, {"models": "not a list"}, {"models": ["not a dict"]}],
)
def test_an_unusable_catalog_yields_no_models(voice, payload) -> None:
    assert voice._parse_codex_model_catalog(payload) == []


def test_a_usable_model_survives_an_unusable_neighbour(voice) -> None:
    options = voice._parse_codex_model_catalog(
        catalog("not a dict", model(visibility="hidden"), model())
    )

    assert [option.slug for option in options] == ["gpt-5.6-luna"]


def test_a_config_value_that_is_not_json_is_kept_as_text(tmp_path) -> None:
    from voice_codex.config import load_startup_config

    path = tmp_path / "voice.yaml"
    path.write_text('microphone: Blue Yeti\ntts: "on"\n', encoding="utf-8")

    assert load_startup_config(path) == {"microphone": "Blue Yeti", "tts": "on"}


def test_an_empty_config_value_reads_as_unset(tmp_path) -> None:
    from voice_codex.config import load_startup_config

    path = tmp_path / "voice.yaml"
    path.write_text("microphone:\n", encoding="utf-8")

    assert load_startup_config(path) == {"microphone": None}


def test_comments_and_blank_lines_are_ignored(tmp_path) -> None:
    from voice_codex.config import load_startup_config

    path = tmp_path / "voice.yaml"
    path.write_text('# a comment\n\ntts: "off"\n', encoding="utf-8")

    assert load_startup_config(path) == {"tts": "off"}


def test_a_line_without_a_separator_is_rejected(tmp_path) -> None:
    from voice_codex.config import load_startup_config

    path = tmp_path / "voice.yaml"
    path.write_text("microphone\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="line 1"):
        load_startup_config(path)


def test_an_unknown_config_key_is_rejected(tmp_path) -> None:
    from voice_codex.config import load_startup_config

    path = tmp_path / "voice.yaml"
    path.write_text("volume: 11\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unknown startup config key 'volume'"):
        load_startup_config(path)


def test_a_config_that_cannot_be_written_is_reported(tmp_path) -> None:
    from voice_codex.config import save_startup_config

    unwritable = tmp_path / "missing-directory" / "voice.yaml"

    with pytest.raises(RuntimeError, match="Could not save startup config"):
        save_startup_config(unwritable, {"microphone": "Yeti"})


def test_saved_settings_reload_to_the_same_values(tmp_path) -> None:
    from voice_codex.config import load_startup_config, save_startup_config

    path = tmp_path / "voice.yaml"
    settings = {
        "microphone": "Blue Yeti",
        "tts": "on",
        "them_output": "isolated",
        "playback_output": None,
        "codex_after": "both",
    }

    save_startup_config(path, settings)

    assert load_startup_config(path) == settings
