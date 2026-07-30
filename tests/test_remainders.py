"""Entry points, catalog parsing, and startup-config error paths."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from tagalong import cli
from tagalong.catalog import _parse_codex_model_catalog

ENTRYPOINTS = ["tagalong.py"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_each_entrypoint_launches_the_configured_application(
    monkeypatch, script
) -> None:
    called: list[bool] = []
    monkeypatch.setattr("tagalong.cli.main", lambda: called.append(True))

    runpy.run_path(str(ROOT / script), run_name="__main__")

    assert called == [True]


def test_a_keyboard_interrupt_stops_without_a_traceback(monkeypatch, capsys) -> None:
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "main", interrupted)

    cli.run_entrypoint()

    assert "Stopped." in capsys.readouterr().out


def test_a_startup_failure_is_reported_and_exits_nonzero(monkeypatch, capsys) -> None:
    def failed():
        raise RuntimeError("No audio input devices were found.")

    monkeypatch.setattr(cli, "main", failed)

    with pytest.raises(SystemExit, match="1"):
        cli.run_entrypoint()

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


def test_a_usable_model_is_parsed_with_its_efforts() -> None:
    (option,) = _parse_codex_model_catalog(catalog(model()))

    assert option.slug == "gpt-5.6-luna"
    assert option.label == "Luna"
    # ``none`` is offered on top of the catalog, which lists no model as
    # accepting it; the catalog's own default is untouched by that.
    assert option.efforts == ("none", "low", "high")
    assert option.default_effort == "high"


def test_models_are_ordered_by_priority_then_label() -> None:
    options = _parse_codex_model_catalog(
        catalog(
            model(slug="c", display_name="Third", priority=9),
            model(slug="b", display_name="Beta", priority=1),
            model(slug="a", display_name="Alpha", priority=1),
        )
    )

    assert [option.label for option in options] == ["Alpha", "Beta", "Third"]


def test_model_priority_outranks_alphabetical_label_order() -> None:
    options = _parse_codex_model_catalog(
        catalog(
            model(slug="slow", display_name="Alpha", priority=9),
            model(slug="fast", display_name="Zulu", priority=1),
        )
    )

    assert [option.slug for option in options] == ["fast", "slow"]


def test_a_model_without_a_priority_sorts_last() -> None:
    options = _parse_codex_model_catalog(
        catalog(
            model(slug="b", display_name="No priority", priority=None),
            model(slug="a", display_name="Has priority", priority=5),
        )
    )

    assert [option.label for option in options] == ["Has priority", "No priority"]


def test_a_missing_display_name_falls_back_to_the_slug() -> None:
    (option,) = _parse_codex_model_catalog(catalog(model(display_name=None)))

    assert option.label == "gpt-5.6-luna"


def test_a_non_text_display_name_falls_back_to_the_slug() -> None:
    (option,) = _parse_codex_model_catalog(catalog(model(display_name=7)))

    assert option.label == "gpt-5.6-luna"


def test_an_invalid_reasoning_level_does_not_hide_a_later_valid_one() -> None:
    (option,) = _parse_codex_model_catalog(
        catalog(
            model(
                supported_reasoning_levels=[7, {"effort": "high"}],
                default_reasoning_level="high",
            )
        )
    )

    assert option.efforts == ("none", "high")
    assert option.default_effort == "high"


def test_an_unlisted_default_effort_falls_back_to_the_first() -> None:
    (option,) = _parse_codex_model_catalog(
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
def test_an_unusable_model_is_skipped(overrides) -> None:
    assert _parse_codex_model_catalog(catalog(model(**overrides))) == []


@pytest.mark.parametrize(
    "payload",
    ["not a dict", None, 7, {}, {"models": "not a list"}, {"models": ["not a dict"]}],
)
def test_an_unusable_catalog_yields_no_models(payload) -> None:
    assert _parse_codex_model_catalog(payload) == []


def test_a_usable_model_survives_an_unusable_neighbour() -> None:
    options = _parse_codex_model_catalog(
        catalog("not a dict", model(visibility="hidden"), model())
    )

    assert [option.slug for option in options] == ["gpt-5.6-luna"]


def test_the_committed_example_config_is_valid_for_a_first_run() -> None:
    """The example ships every key unset, so a first run resolves them all.

    Derived from the key list rather than restating it: a key added to the
    config without a line in the example is one a new machine cannot discover,
    and a second hand-written list here would not notice.
    """
    from tagalong.config import STARTUP_CONFIG_KEYS, load_startup_config

    assert load_startup_config(ROOT / "tagalong.example.yaml") == dict.fromkeys(
        STARTUP_CONFIG_KEYS
    )


def test_a_model_already_listing_the_unlisted_effort_is_not_given_it_twice() -> None:
    """No catalog lists it today, but one that starts to must not be doubled."""
    (option,) = _parse_codex_model_catalog(
        catalog(
            model(
                supported_reasoning_levels=[{"effort": "none"}, {"effort": "low"}],
                default_reasoning_level="low",
            )
        )
    )

    assert option.efforts == ("none", "low")
