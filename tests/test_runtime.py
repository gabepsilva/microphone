"""Runtime behavior of the classes ``main`` wires together.

Each test here names the module it exercises in its imports. The file is
kept whole because these are the behaviors that matter once the parts are
assembled, not because the parts share a module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from openai_codex import Sandbox

from voice_codex.capture import SoundActivityReporter
from voice_codex.catalog import CodexModelOption, probe_codex_models
from voice_codex.choosers import audio_outputs
from voice_codex.cli import main
from voice_codex.codex import CodexConversation, load_codex_sdk
from voice_codex.domain import TranscriptRouter


def test_cli_no_longer_offers_the_ansi_ui_switch(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["voice_codex.py", "--help"])

    with pytest.raises(SystemExit, match="0"):
        main()

    assert "--ui" not in capsys.readouterr().out


def test_audio_outputs_parses_pactl_json(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pactl")
    payload = [
        {
            "name": "sink-name",
            "monitor_source": "sink-name.monitor",
            "description": "Headphones",
        },
        {"name": "incomplete"},
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    assert audio_outputs() == [
        {
            "name": "sink-name",
            "monitor": "sink-name.monitor",
            "description": "Headphones",
        }
    ]


def test_sound_activity_reporter_names_the_channel_it_speaks_for() -> None:
    reports: list[tuple[str, bool]] = []
    display = SimpleNamespace(
        set_audio=lambda channel, active: reports.append((channel, active))
    )
    clock = iter([0.0, 1.0]).__next__
    reporter = SoundActivityReporter(display, "mic", release=0.35, clock=clock)

    reporter.update(np.array([-0.1, 0.1], dtype=np.float32))
    reporter.update(np.array([], dtype=np.float32))

    assert reports == [("mic", True), ("mic", False)]


def test_codex_context_entries_include_timestamps() -> None:
    router = TranscriptRouter()
    request = router.ingest(
        "User Voice", "What time is it?", "2026-07-26T12:30:00-04:00", True
    )

    assert request is not None
    assert CodexConversation.context_entries(request) == [
        {
            "timestamp": "2026-07-26T12:30:00-04:00",
            "source": "User Voice",
            "text": "What time is it?",
        }
    ]


def test_probe_codex_models_uses_visible_catalog_entries(monkeypatch) -> None:
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
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(catalog)),
    )

    options = probe_codex_models()

    assert options == [
        CodexModelOption("gpt-5.6-luna", "GPT-5.6 Luna", ("none", "low"), "low"),
        CodexModelOption(
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            ("none", "low", "medium"),
            "medium",
        ),
    ]


def test_model_probe_uses_the_exact_cli_contract_then_its_bundled_fallback(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    outputs = iter(
        [
            "not json",
            json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-5.6-luna",
                            "display_name": "Luna",
                            "visibility": "list",
                            "supported_reasoning_levels": [{"effort": "low"}],
                            "default_reasoning_level": "low",
                        }
                    ]
                }
            ),
        ]
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(subprocess, "run", run)

    assert probe_codex_models() == [
        CodexModelOption("gpt-5.6-luna", "Luna", ("none", "low"), "low")
    ]
    assert calls == [
        (
            ["codex", "debug", "models"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 3,
            },
        ),
        (
            ["codex", "debug", "models", "--bundled"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 3,
            },
        ),
    ]


def test_model_switch_forks_the_current_codex_thread() -> None:
    # Built without ``__init__``, so the SDK names the fork call reads are
    # bound here rather than by whichever test happened to run first.
    load_codex_sdk()
    calls: list[tuple[str, dict[str, object]]] = []
    updates: list[dict[str, object]] = []
    conversation = object.__new__(CodexConversation)
    conversation.model = "gpt-5.6-luna"
    conversation.reasoning_effort = "low"
    conversation.service_tier = None
    conversation.sandbox = Sandbox("full-access")
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
    conversation.settings_lock = threading.Lock()
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
