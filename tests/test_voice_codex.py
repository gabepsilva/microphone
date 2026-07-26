from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_startup_config_round_trip(voice, tmp_path) -> None:
    settings = {
        "microphone": "USB microphone",
        "tts": "on",
        "them_output": "meeting-monitor",
        "playback_output": None,
        "codex_after": "both",
    }
    config = tmp_path / "voice.yaml"

    voice.save_startup_config(config, settings)

    assert voice.load_startup_config(config) == settings


def test_example_config_is_valid_for_an_interactive_first_run(voice) -> None:
    config = Path(__file__).resolve().parents[1] / "voice.example.yaml"

    assert voice.load_startup_config(config) == {
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
def test_startup_config_rejects_invalid_input(
    voice, tmp_path, contents, message
) -> None:
    config = tmp_path / "voice.yaml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        voice.load_startup_config(config)


@pytest.mark.parametrize(
    ("requested", "label", "speakers"),
    [
        ("them", "Them", frozenset({"Them"})),
        ("both", "User Voice and Them", frozenset({"User Voice", "Them"})),
        ("user", "User Voice", frozenset({"User Voice"})),
        ("quiet", "Codex will be quiet for voice", frozenset()),
    ],
)
def test_codex_response_policy_mapping(voice, requested, label, speakers) -> None:
    assert voice.choose_codex_after(requested) == (label, speakers)


def test_sentence_chunker_preserves_sentence_order_and_size(voice) -> None:
    emitted: list[str] = []
    chunker = voice.SentenceChunker(emitted.append, max_chars=12)

    chunker.feed("First line. A longer second sentence")
    chunker.flush()

    assert emitted == ["First line.", "A longer", "second", "sentence"]
    assert all(0 < len(chunk) <= 12 for chunk in emitted)


def test_echo_matching_is_case_and_punctuation_insensitive(voice) -> None:
    assert voice.EdgeSentenceTTS._normalize_speech("Hello, WORLD!") == "hello world"
    assert voice.EdgeSentenceTTS._speech_matches(
        "please open the settings panel",
        "open the settings panel",
    )
    assert not voice.EdgeSentenceTTS._speech_matches("first unrelated", "second text")


def test_listener_filters_low_confidence_words_and_flushes_one_turn(voice) -> None:
    submitted: list[tuple[str, str]] = []
    display = SimpleNamespace(
        update=lambda text: None,
        commit=lambda text: None,
        finish_turn=lambda: None,
        close=lambda: None,
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


def test_live_transcript_is_truncated_to_terminal_width(voice, monkeypatch) -> None:
    monkeypatch.setattr(
        voice.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((20, 24)),
    )

    assert (
        voice.TranscriptDisplay._fit_live_line("Them", "abcdefghijklmno")
        == "…cdefghijklmno"
    )
