from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_cli_no_longer_offers_the_ansi_ui_switch(voice, monkeypatch, capsys) -> None:
    monkeypatch.setattr(voice.sys, "argv", ["voice-codex.py", "--help"])

    with pytest.raises(SystemExit, match="0"):
        voice.main()

    assert "--ui" not in capsys.readouterr().out


def test_listener_filters_low_confidence_words_and_flushes_one_turn(voice) -> None:
    submitted: list[tuple[str, str]] = []
    display = SimpleNamespace(
        update=lambda speaker, text: None,
        commit=lambda speaker, text: None,
        finish_turn=lambda speaker: None,
        close_speaker=lambda speaker: None,
    )
    listener = voice.ConversationListener(
        0.6,
        3.0,
        "User Voice",
        lambda speaker, text: submitted.append((speaker, text)),
        display,
    )
    line = SimpleNamespace(
        words=[
            SimpleNamespace(word=" keep ", confidence=0.8),
            SimpleNamespace(word="discard", confidence=0.2),
        ],
        text="fallback",
    )

    assert listener._text(line) == "keep"
    listener.pending = ["one", "two"]
    listener.timer_generation = 4
    listener._flush(4)

    assert submitted == [("User Voice", "one two")]
    assert listener.pending == []


def test_listener_discards_completed_audio_while_muted(voice) -> None:
    submitted: list[tuple[str, str]] = []
    display = SimpleNamespace(
        update=lambda speaker, text: None,
        commit=lambda speaker, text: None,
        finish_turn=lambda speaker: None,
        close_speaker=lambda speaker: None,
    )
    listener = voice.ConversationListener(
        0.6,
        3.0,
        "User Voice",
        lambda speaker, text: submitted.append((speaker, text)),
        display,
    )
    listener.set_muted(True)
    line = SimpleNamespace(words=[], text="do not submit")

    listener.on_line_completed(SimpleNamespace(line=line))

    assert submitted == []
    assert listener.pending == []


def test_audio_outputs_parses_pactl_json(voice, monkeypatch) -> None:
    monkeypatch.setattr(voice.shutil, "which", lambda name: "/usr/bin/pactl")
    payload = [
        {
            "name": "sink-name",
            "monitor_source": "sink-name.monitor",
            "description": "Headphones",
        },
        {"name": "incomplete"},
    ]
    monkeypatch.setattr(
        voice.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    assert voice.audio_outputs() == [
        {
            "name": "sink-name",
            "monitor": "sink-name.monitor",
            "description": "Headphones",
        }
    ]


def test_audio_level_reporter_publishes_normalized_rms_levels(voice) -> None:
    levels: list[tuple[str, float]] = []
    display = SimpleNamespace(
        set_audio=lambda channel, level: levels.append((channel, level))
    )
    reporter = voice.AudioLevelReporter(display, "mic", interval=0)

    reporter.update(voice.np.array([-0.1, 0.1], dtype=voice.np.float32))
    reporter.update(voice.np.array([], dtype=voice.np.float32))

    assert levels[0] == ("mic", pytest.approx(0.5))
    assert levels[1] == ("mic", 0.0)


def test_codex_context_entries_include_timestamps(voice) -> None:
    router = voice.TranscriptRouter()
    request = router.ingest(
        "User Voice", "What time is it?", "2026-07-26T12:30:00-04:00", True
    )

    assert request is not None
    assert voice.CodexConversation.context_entries(request) == [
        {
            "timestamp": "2026-07-26T12:30:00-04:00",
            "source": "User Voice",
            "text": "What time is it?",
        }
    ]


def test_probe_codex_models_uses_visible_catalog_entries(voice, monkeypatch) -> None:
    catalog = {
        "models": [
            {
                "slug": "hidden-model",
                "display_name": "Hidden",
                "visibility": "hidden",
                "supported_reasoning_levels": [{"effort": "low"}],
            },
            {
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6 Sol",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 2,
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low"},
                    {"effort": "medium"},
                ],
            },
            {
                "slug": "gpt-5.6-luna",
                "display_name": "GPT-5.6 Luna",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [{"effort": "low"}],
            },
        ]
    }
    monkeypatch.setattr(
        voice.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(catalog)),
    )

    options = voice.probe_codex_models()

    assert options == [
        voice.CodexModelOption("gpt-5.6-luna", "GPT-5.6 Luna", ("low",), "low"),
        voice.CodexModelOption(
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            ("low", "medium"),
            "medium",
        ),
    ]


def test_model_switch_forks_the_current_codex_thread(voice) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    updates: list[dict[str, object]] = []
    conversation = object.__new__(voice.CodexConversation)
    conversation.model = "gpt-5.6-luna"
    conversation.reasoning_effort = "low"
    conversation.service_tier = None
    conversation.sandbox = voice.Sandbox("full-access")
    conversation.thread = SimpleNamespace(id="old-thread")
    conversation.codex = SimpleNamespace(
        thread_fork=lambda thread_id, **kwargs: (
            calls.append((thread_id, kwargs)) or SimpleNamespace(id="new-thread")
        )
    )
    conversation.transcript_display = SimpleNamespace(
        set_codex=lambda **fields: updates.append(fields),
        note=lambda text: None,
        error=lambda text: None,
    )
    conversation.settings_lock = voice.threading.Lock()
    conversation.requested_model = "gpt-5.6-sol"
    conversation.requested_reasoning_effort = "high"

    conversation._apply_pending_settings()

    assert calls[0][0] == "old-thread"
    assert calls[0][1]["model"] == "gpt-5.6-sol"
    assert conversation.thread.id == "new-thread"
    assert conversation.model == "gpt-5.6-sol"
    assert conversation.reasoning_effort == "high"
    assert updates == [
        {"model": "gpt-5.6-sol", "thread": "new-thread"},
        {"effort": "high"},
    ]
