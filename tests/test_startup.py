"""Startup argument, selection, and session-lifecycle behavior.

These cover the logic extracted from ``main`` so it can be exercised without a
microphone, a PipeWire server, or a Codex account.
"""

from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tagalong import catalog, startup
from tagalong.domain import POLICY_NAMES, RESPONSE_POLICIES, SpeakerGate
from tagalong.listener import TranscriptSubmitter, tts_switch
from tagalong.startup import (
    StartupSelection,
    build_session_state,
    parse_startup_args,
    print_startup_summary,
    resolve_startup_selection,
    run_session,
    startup_settings,
    validate_codex_reasoning,
)


def write_config(tmp_path, body):
    path = tmp_path / "tagalong.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def empty_config(tmp_path):
    return write_config(tmp_path, "")


BASE_SELECTION = StartupSelection(
    device_index=3,
    device={"name": "Yeti"},
    tts_enabled=False,
    tts_provider="piper",
    them_stream=None,
    tts_output=None,
    policy=RESPONSE_POLICIES["them"],
)


def saved_args(**overrides):
    """The parsed options ``startup_settings`` reads alongside a selection."""
    return SimpleNamespace(
        **{
            "turn_silence": 3.0,
            "codex_model": "gpt-5.6-luna",
            "codex_reasoning": "low",
            "codex_fast": True,
            "codex_prefire": True,
            **overrides,
        }
    )


def selection(**overrides):
    """A resolved selection: a plain User-Voice-only session unless overridden."""
    return replace(BASE_SELECTION, **overrides)


def test_parser_defaults_match_the_documented_startup_choices(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    assert (args.model, args.language, args.confidence) == (
        "medium-streaming",
        "en",
        0.60,
    )
    assert (args.sandbox, args.codex_model, args.codex_reasoning) == (
        "full-access",
        "gpt-5.6-luna",
        "low",
    )
    assert (args.turn_silence, args.tts_provider, args.tts_voice) == (
        3.0,
        "piper",
        "en_US-lessac-medium",
    )
    assert args.codex_fast is True
    assert (args.tts, args.codex_after, args.microphone) == ("on", "both", None)
    assert args.them_stream == "none"


def test_command_line_overrides_the_startup_config_file(tmp_path) -> None:
    config = write_config(
        tmp_path,
        'microphone: "Config Mic"\ntts: "on"\ncodex_after: "user"\n',
    )

    _, args = parse_startup_args(["--config", config, "--microphone", "Flag Mic"])

    assert args.microphone == "Flag Mic"
    assert (args.tts, args.codex_after) == ("on", "user")


def test_a_missing_startup_config_uses_defaults(tmp_path) -> None:
    missing = str(tmp_path / "absent.yaml")

    _, args = parse_startup_args(["--config", missing])

    assert (args.tts, args.them_stream, args.codex_after) == (
        "on",
        "none",
        "both",
    )


def test_an_unreadable_startup_config_exits_instead_of_prompting(tmp_path) -> None:
    unreadable = tmp_path / "tagalong.yaml"
    unreadable.mkdir()

    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(["--config", str(unreadable)])


@pytest.mark.parametrize(
    ("body", "argv"),
    [
        ('tts: "maybe"\n', []),
        ('codex_after: "sometimes"\n', []),
        ("", ["--confidence", "1.5"]),
        ("", ["--confidence", "-0.1"]),
        ("", ["--turn-silence", "0"]),
        ("", ["--turn-silence", "-2"]),
    ],
)
def test_out_of_range_startup_values_are_rejected(tmp_path, body, argv) -> None:
    config = write_config(tmp_path, body)

    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(["--config", config, *argv])


def test_saved_settings_round_trip_back_to_the_same_selection() -> None:
    chosen = selection(
        tts_enabled=True,
        them_stream="Chromium",
        tts_output={"name": "alsa_output.pci", "description": "Speakers"},
        policy=RESPONSE_POLICIES["both"],
    )

    assert startup_settings(chosen, saved_args()) == {
        "microphone": "Yeti",
        "tts": "on",
        "tts_provider": "piper",
        "them_stream": "Chromium",
        "tts_output": "alsa_output.pci",
        "codex_after": "both",
        "turn_silence": 3.0,
        "codex_model": "gpt-5.6-luna",
        "codex_reasoning": "low",
        "codex_fast": True,
        "codex_prefire": True,
    }


def test_a_session_without_them_saves_the_word_the_parser_reads_back() -> None:
    """The saved name has to be one ``--them-stream`` accepts, not ``None``."""
    saved = startup_settings(selection(), saved_args())

    assert saved["them_stream"] == "none"
    assert saved["tts_output"] is None


def test_the_startup_summary_reports_the_resolved_choices(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path), "--codex-fast"])
    stream = io.StringIO()

    print_startup_summary(
        args,
        selection(
            tts_enabled=True,
            them_stream="ZOOM VoiceEngine",
            tts_output={"name": "alsa", "description": "Speakers"},
        ),
        stream=stream,
    )
    summary = stream.getvalue()

    assert "User microphone: Yeti" in summary
    assert "Them application: ZOOM VoiceEngine" in summary
    assert "Codex response policy: Them" in summary
    assert "Voice turn silence: 3.0s" in summary
    assert "Codex speed: Fast" in summary
    assert "Codex command access: full-access" in summary
    assert "Codex audio: Piper (local) (en_US-lessac-medium)" in summary
    assert "Codex speaks through: Speakers" in summary


def test_the_startup_summary_names_a_session_with_no_them_application(
    tmp_path,
) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])
    stream = io.StringIO()

    print_startup_summary(args, selection(tts_enabled=True), stream=stream)
    summary = stream.getvalue()

    assert "Them application: None" in summary
    assert "Codex speaks through" not in summary


def test_the_sidebar_state_reflects_the_resolved_startup_choices(tmp_path) -> None:
    _, args = parse_startup_args(
        [
            "--config",
            empty_config(tmp_path),
            "--codex-fast",
            "--codex-reasoning",
            "high",
        ]
    )

    state = build_session_state(
        args,
        selection(tts_enabled=True, policy=RESPONSE_POLICIES["user"]),
    )

    assert (state.policy, state.codex_tier, state.codex_effort) == (
        "user",
        "fast",
        "high",
    )
    assert state.tts_enabled is True
    assert (state.turn_silence, state.confidence) == (3.0, 0.60)
    assert (state.moonshine, state.language) == ("medium-streaming", "en")


def test_the_sidebar_state_uses_each_models_catalog_efforts(tmp_path) -> None:
    _, args = parse_startup_args(
        [
            "--config",
            empty_config(tmp_path),
            "--codex-model",
            "sol",
            "--codex-reasoning",
            "ultra",
        ]
    )
    models = [
        catalog.CodexModelOption("sol", "Sol", ("none", "low", "ultra"), "low"),
        catalog.CodexModelOption("luna", "Luna", ("none", "low", "medium"), "medium"),
    ]

    state = build_session_state(args, selection(), models)

    assert state.codex_models == [("Sol", "sol"), ("Luna", "luna")]
    assert state.codex_efforts == ["none", "low", "ultra"]
    assert state.codex_efforts_by_model == {
        "sol": ["none", "low", "ultra"],
        "luna": ["none", "low", "medium"],
    }
    assert state.codex_default_effort_by_model == {
        "sol": "low",
        "luna": "medium",
    }


def test_a_reasoning_effort_the_selected_model_does_not_offer_is_rejected(
    tmp_path, capsys
) -> None:
    parser, args = parse_startup_args(
        [
            "--config",
            empty_config(tmp_path),
            "--codex-model",
            "luna",
            "--codex-reasoning",
            "sometimes",
        ]
    )
    models = [
        catalog.CodexModelOption("luna", "Luna", ("none", "low", "medium"), "medium")
    ]

    with pytest.raises(SystemExit, match="2"):
        validate_codex_reasoning(parser, args, models)

    assert (
        "startup config 'codex_reasoning' for model 'luna' must be one of "
        "'none', 'low', 'medium'"
    ) in capsys.readouterr().err


def test_a_configured_model_the_catalog_omits_keeps_its_effort(tmp_path) -> None:
    parser, args = parse_startup_args(
        [
            "--config",
            empty_config(tmp_path),
            "--codex-model",
            "private-model",
            "--codex-reasoning",
            "private-effort",
        ]
    )

    assert validate_codex_reasoning(parser, args, []) is None


def test_the_chosen_application_and_speech_output_reach_the_selection(
    monkeypatch, tmp_path
) -> None:
    speakers = {"name": "alsa", "description": "Speakers"}
    monkeypatch.setattr(startup, "input_devices", lambda: [(2, {"name": "M"})])
    monkeypatch.setattr(startup, "choose_tts", lambda requested: False)
    monkeypatch.setattr(startup, "choose_them_stream", lambda requested: "Chromium")
    monkeypatch.setattr(startup, "choose_tts_output", lambda requested: speakers)
    monkeypatch.setattr(
        startup, "choose_codex_after", lambda requested: ("Them", {"Them"})
    )
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    chosen = resolve_startup_selection(args)

    assert chosen.them_stream == "Chromium"
    assert chosen.tts_output is speakers
    assert (chosen.device_index, chosen.tts_enabled) == (2, False)


def test_a_session_with_no_them_application_resolves_to_nothing(
    monkeypatch, tmp_path
) -> None:
    """Speech no longer constrains the Them choice, so None must survive it."""
    monkeypatch.setattr(startup, "input_devices", lambda: [(0, {"name": "M"})])
    monkeypatch.setattr(startup, "choose_tts", lambda requested: True)
    monkeypatch.setattr(startup, "choose_them_stream", lambda requested: None)
    monkeypatch.setattr(
        startup, "choose_codex_after", lambda requested: ("User Voice", {"User Voice"})
    )
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    chosen = resolve_startup_selection(args)

    assert chosen.them_stream is None
    assert chosen.tts_output is None
    assert chosen.tts_enabled is True


def test_startup_keeps_running_without_a_microphone(monkeypatch, tmp_path) -> None:
    # Fake the adapter where both the old startup chooser and the new direct
    # resolver reach it, so the regression fails on behavior rather than on a
    # refactor-specific missing attribute.
    monkeypatch.setattr("tagalong.choosers.input_devices", list)
    if hasattr(startup, "input_devices"):
        monkeypatch.setattr(startup, "input_devices", list)
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    chosen = resolve_startup_selection(args)

    assert chosen.device_index is None
    assert chosen.device is None


def test_startup_keeps_running_when_microphone_discovery_is_unavailable(
    monkeypatch, tmp_path, capsys
) -> None:
    def unavailable():
        raise RuntimeError("PortAudio is unavailable")

    monkeypatch.setattr(startup, "input_devices", unavailable)
    _, args = parse_startup_args(["--config", str(tmp_path / "missing.yaml")])

    chosen = resolve_startup_selection(args)

    assert chosen.device is None
    assert (
        "Microphone discovery unavailable: PortAudio is unavailable"
        in capsys.readouterr().err
    )


def test_a_saved_microphone_can_be_unavailable_at_startup(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(startup, "input_devices", list)
    config = write_config(tmp_path, 'microphone: "Yeti"\n')
    _, args = parse_startup_args(["--config", config])

    chosen = resolve_startup_selection(args)

    assert chosen.device is None
    assert startup_settings(chosen, saved_args(microphone="Yeti"))["microphone"] == (
        "Yeti"
    )
    assert "select one from the sidebar" in capsys.readouterr().err


class FakeTTS:
    def __init__(self, echoes=()):
        self.echoes = set(echoes)
        self.interrupted = 0

    def is_likely_echo(self, text):
        return text in self.echoes

    def interrupt(self):
        self.interrupted += 1


class FakeConversation:
    def __init__(self):
        self.ingested = []

    def ingest(self, speaker, text, respond):
        self.ingested.append((speaker, text, respond))


def test_the_submitter_answers_only_the_speakers_the_policy_allows() -> None:
    conversation = FakeConversation()
    gate = SpeakerGate({"Them"}, {"User Voice", "Them"})
    submitter = TranscriptSubmitter(conversation, gate, None)

    submitter.submit("Them", "a question")
    submitter.submit("User Voice", "thinking aloud")

    assert conversation.ingested == [
        ("Them", "a question", True),
        ("User Voice", "thinking aloud", False),
    ]


class FakeChannel:
    """A channel that records whether its buffer was swept into a reply."""

    def __init__(self, speaker, conversation, buffered=None):
        self.speaker = speaker
        self.conversation = conversation
        self.buffered = buffered
        self.flushes = 0

    def flush_now(self):
        self.flushes += 1
        if self.buffered is not None:
            self.conversation.ingest(self.speaker, self.buffered, respond=False)
            self.buffered = None


def test_a_reply_carries_context_still_buffered_on_the_other_channel() -> None:
    """Them's answer must not wait out User Voice's own silence timer."""
    conversation = FakeConversation()
    gate = SpeakerGate({"Them"}, {"User Voice", "Them"})
    submitter = TranscriptSubmitter(conversation, gate, None)
    user = FakeChannel("User Voice", conversation, buffered="what about latency")
    submitter.add_listener(user)
    submitter.add_listener(FakeChannel("Them", conversation))

    submitter.submit("Them", "so what do you think")

    assert user.flushes == 1
    assert conversation.ingested == [
        ("User Voice", "what about latency", False),
        ("Them", "so what do you think", True),
    ]


def test_a_context_only_channel_is_never_swept_by_its_own_turn() -> None:
    conversation = FakeConversation()
    gate = SpeakerGate({"Them"}, {"User Voice", "Them"})
    submitter = TranscriptSubmitter(conversation, gate, None)
    them = FakeChannel("Them", conversation)
    user = FakeChannel("User Voice", conversation)
    submitter.add_listener(them)
    submitter.add_listener(user)

    submitter.submit("User Voice", "thinking aloud")

    assert (them.flushes, user.flushes) == (0, 0)
    assert conversation.ingested == [("User Voice", "thinking aloud", False)]


def test_a_channel_the_policy_answers_is_left_to_its_own_silence() -> None:
    """Sweeping it would queue a reply to speech its speaker has not finished."""
    conversation = FakeConversation()
    gate = SpeakerGate({"User Voice", "Them"}, {"User Voice", "Them"})
    submitter = TranscriptSubmitter(conversation, gate, None)
    user = FakeChannel("User Voice", conversation, buffered="mid sentence")
    submitter.add_listener(user)

    submitter.submit("Them", "so what do you think")

    assert user.flushes == 0
    assert conversation.ingested == [("Them", "so what do you think", True)]


def test_an_ignored_echo_never_sweeps_the_other_channel() -> None:
    conversation = FakeConversation()
    gate = SpeakerGate({"Them"}, {"User Voice", "Them"})
    submitter = TranscriptSubmitter(
        conversation, gate, FakeTTS(echoes={"my own reply"}), stream=io.StringIO()
    )
    user = FakeChannel("User Voice", conversation, buffered="still talking")
    submitter.add_listener(user)

    submitter.submit("Them", "my own reply")

    assert user.flushes == 0
    assert conversation.ingested == []


def test_the_submitter_drops_a_transcript_of_codex_speaking() -> None:
    conversation = FakeConversation()
    gate = SpeakerGate({"Them"}, {"Them"})
    stream = io.StringIO()
    submitter = TranscriptSubmitter(
        conversation, gate, FakeTTS(echoes={"my own reply"}), stream=stream
    )

    submitter.submit("Them", "my own reply")
    submitter.submit("Them", "a real question")

    assert conversation.ingested == [("Them", "a real question", True)]
    assert "ignored likely Codex TTS echo from Them" in stream.getvalue()


def test_typed_text_is_never_treated_as_an_echo() -> None:
    conversation = FakeConversation()
    gate = SpeakerGate({"User Text"}, {"User Text"})
    submitter = TranscriptSubmitter(
        conversation, gate, FakeTTS(echoes={"my own reply"})
    )

    submitter.submit("User Text", "my own reply")

    assert conversation.ingested == [("User Text", "my own reply", True)]


def test_real_speech_interrupts_playback_but_an_echo_does_not() -> None:
    tts = FakeTTS(echoes={"codex speaking"})
    submitter = TranscriptSubmitter(FakeConversation(), None, tts)

    assert submitter.handle_speech("codex speaking") is False
    assert tts.interrupted == 0
    assert submitter.handle_speech("a person talking") is True
    assert tts.interrupted == 1


def test_speech_never_interrupts_when_tts_is_off() -> None:
    submitter = TranscriptSubmitter(FakeConversation(), None, None)

    assert submitter.handle_speech("a person talking") is False


class FakeTranscriber:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def start(self):
        self.events.append(f"start {self.name}")

    def stop(self):
        self.events.append(f"stop {self.name}")

    def close(self):
        self.events.append(f"close {self.name}")


class FakeListener:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def close(self):
        self.events.append(f"close listener {self.name}")


def session_parts(monkeypatch, run):
    monkeypatch.setattr(catalog, "probe_codex_models", list)
    events: list[str] = []
    tui = SimpleNamespace(
        run=run,
        note=lambda message: None,
        set_codex_catalog=lambda *args: None,
    )
    channels = [
        (FakeTranscriber("mic", events), FakeListener("mic", events)),
        (FakeTranscriber("them", events), FakeListener("them", events)),
    ]
    conversation = SimpleNamespace(close=lambda: events.append("close conversation"))
    return events, tui, channels, conversation


def test_a_session_tears_every_channel_down_in_a_safe_order(monkeypatch) -> None:
    events, tui, channels, conversation = session_parts(monkeypatch, run=lambda: None)

    run_session(tui, channels, conversation)

    assert events == [
        "start mic",
        "start them",
        "stop mic",
        "stop them",
        "close listener mic",
        "close listener them",
        "close mic",
        "close them",
        "close conversation",
    ]


def test_a_session_closes_the_far_end_before_the_channels_it_started_with(
    monkeypatch,
) -> None:
    """It owns the only reference to whatever it built while the session ran."""
    events, tui, channels, conversation = session_parts(monkeypatch, run=lambda: None)
    them = SimpleNamespace(close=lambda: events.append("close them"))

    run_session(tui, channels, conversation, them=them)

    assert events[:3] == ["start mic", "start them", "close them"]


def test_a_session_closes_dynamic_audio_channels_before_static_ones(
    monkeypatch,
) -> None:
    events, tui, channels, conversation = session_parts(monkeypatch, run=lambda: None)
    them = SimpleNamespace(close=lambda: events.append("close dynamic them"))
    microphone = SimpleNamespace(
        close=lambda: events.append("close dynamic microphone")
    )

    run_session(
        tui,
        channels,
        conversation,
        them=them,
        microphone=microphone,
    )

    assert events[:4] == [
        "start mic",
        "start them",
        "close dynamic them",
        "close dynamic microphone",
    ]


def test_an_interrupted_session_still_closes_everything(monkeypatch, capsys) -> None:
    def interrupt():
        raise KeyboardInterrupt

    events, tui, channels, conversation = session_parts(monkeypatch, run=interrupt)

    run_session(tui, channels, conversation)

    assert "Stopping..." in capsys.readouterr().out
    assert events[-1] == "close conversation"
    assert events.count("close conversation") == 1


def test_the_speech_toggle_reports_no_speech_when_the_session_has_none() -> None:
    assert tts_switch(None)(True) is False


def test_the_speech_toggle_forwards_to_the_engine() -> None:
    settings: list[bool] = []
    toggle = tts_switch(SimpleNamespace(set_enabled=settings.append))

    assert toggle(False) is True
    assert toggle(True) is True
    assert settings == [False, True]


# --------------------------------------------------------------------------
# The policy vocabulary has one definition
#
# These fail if a policy is added to `RESPONSE_POLICIES` and the command line
# is not derived from it, which is how the four copies of this list drifted
# apart in the first place.
# --------------------------------------------------------------------------


def test_the_command_line_offers_exactly_the_defined_policies() -> None:
    parser = startup.build_parser()
    action = next(a for a in parser._actions if a.dest == "codex_after")

    assert tuple(action.choices) == POLICY_NAMES


def test_a_bad_config_policy_is_rejected_by_naming_every_real_one(
    tmp_path, capsys
) -> None:
    config = write_config(tmp_path, 'codex_after: "sometimes"\n')

    with pytest.raises(SystemExit):
        parse_startup_args(["--config", config])

    message = capsys.readouterr().err
    for name in POLICY_NAMES:
        assert repr(name) in message


def test_every_policy_round_trips_through_a_saved_config() -> None:
    for name, policy in RESPONSE_POLICIES.items():
        saved = startup_settings(selection(policy=policy), saved_args())

        assert saved["codex_after"] == name
        # A saved name has to be a name the command line will take back.
        assert name in POLICY_NAMES


def test_an_unknown_speech_provider_in_the_config_is_rejected(tmp_path) -> None:
    config = write_config(tmp_path, 'tts_provider: "festival"\n')

    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(["--config", config])


def test_a_configured_speech_provider_brings_its_own_default_voice(tmp_path) -> None:
    config = write_config(tmp_path, 'tts_provider: "edge"\n')

    _, args = parse_startup_args(["--config", config])

    assert (args.tts_provider, args.tts_voice) == ("edge", "en-US-AndrewNeural")


def test_a_chosen_voice_survives_the_provider_default(tmp_path) -> None:
    _, args = parse_startup_args(
        ["--config", empty_config(tmp_path), "--tts-voice", "en_US-ryan-high"]
    )

    assert (args.tts_provider, args.tts_voice) == ("piper", "en_US-ryan-high")


def test_the_startup_summary_reports_a_session_without_speech(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])
    stream = io.StringIO()

    print_startup_summary(args, selection(tts_enabled=False), stream=stream)

    assert "Codex audio: Off" in stream.getvalue()


@pytest.mark.parametrize("seconds", ["0.1", "31", "0"])
def test_a_turn_silence_outside_the_editable_range_is_rejected(
    tmp_path, seconds
) -> None:
    """The command line and the sidebar field enforce one range, not two."""
    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(
            ["--config", empty_config(tmp_path), "--turn-silence", seconds]
        )


@pytest.mark.parametrize("seconds", ["0.25", "30", "1.5"])
def test_a_turn_silence_inside_the_editable_range_is_accepted(
    tmp_path, seconds
) -> None:
    _, args = parse_startup_args(
        ["--config", empty_config(tmp_path), "--turn-silence", seconds]
    )

    assert args.turn_silence == float(seconds)


@pytest.mark.parametrize(
    "body",
    [
        "codex_reasoning: 7\n",
        'turn_silence: "three"\n',
        "turn_silence: true\n",
        'codex_fast: "yes"\n',
        'codex_prefire: "yes"\n',
    ],
)
def test_a_config_value_the_session_cannot_use_is_rejected(tmp_path, body) -> None:
    """These arrive from a file the interface writes, so they are not typed."""
    config = write_config(tmp_path, body)

    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(["--config", config])


def test_saved_codex_choices_are_loaded_back(tmp_path) -> None:
    config = write_config(
        tmp_path,
        'codex_model: "gpt-5.6-sol"\ncodex_reasoning: "high"\nturn_silence: 1.25\n',
    )

    _, args = parse_startup_args(["--config", config])

    assert (args.codex_model, args.codex_reasoning, args.turn_silence) == (
        "gpt-5.6-sol",
        "high",
        1.25,
    )


def test_codex_fast_is_asked_for_unless_the_session_declines_it(tmp_path) -> None:
    # A spoken reply cannot start before the first token, so the tier is on by
    # default and opting out is explicit.
    _, fast = parse_startup_args(["--config", empty_config(tmp_path)])
    _, standard = parse_startup_args(
        ["--config", empty_config(tmp_path), "--no-codex-fast"]
    )

    assert (fast.codex_fast, standard.codex_fast) == (True, False)


def test_a_saved_standard_tier_survives_the_default(tmp_path) -> None:
    config = write_config(tmp_path, "codex_fast: false\n")

    _, args = parse_startup_args(["--config", config])

    assert args.codex_fast is False


def test_the_command_line_still_outranks_a_saved_tier(tmp_path) -> None:
    config = write_config(tmp_path, "codex_fast: false\n")

    _, args = parse_startup_args(["--config", config, "--codex-fast"])

    assert args.codex_fast is True


def test_a_standard_tier_session_says_so_in_its_summary(tmp_path) -> None:
    _, args = parse_startup_args(
        ["--config", empty_config(tmp_path), "--no-codex-fast"]
    )
    stream = io.StringIO()

    print_startup_summary(args, selection(), stream=stream)

    assert "Codex speed: Standard" in stream.getvalue()


def test_the_command_line_still_outranks_a_saved_codex_choice(tmp_path) -> None:
    config = write_config(tmp_path, 'codex_model: "gpt-5.6-sol"\nturn_silence: 1.25\n')

    _, args = parse_startup_args(
        ["--config", config, "--codex-model", "gpt-5.6-luna", "--turn-silence", "2"]
    )

    assert (args.codex_model, args.turn_silence) == ("gpt-5.6-luna", 2.0)


def test_an_empty_config_resolves_every_default_from_the_code(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    assert args.turn_silence == 3.0
    assert args.codex_model == "gpt-5.6-luna"
    assert args.codex_reasoning == "low"
    assert args.codex_fast is True
    assert args.tts_provider == "piper"


def test_the_codex_sdk_is_not_imported_just_to_start_the_interface() -> None:
    """The SDK costs about half a second to import and draws nothing.

    Importing it eagerly left the session blank for that half second before
    the startup questions appeared. Run in a fresh interpreter because the
    suite has almost certainly imported it already.
    """
    probe = (
        "import sys;"
        "import tagalong.cli;"
        "import tagalong.tui;"
        "print('openai_codex' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.stdout.strip() == "False"


def test_building_a_conversation_loads_the_sdk_it_dispatches_on() -> None:
    """Deferring the import must not leave the dispatch names unbound."""
    from tagalong import codex as codex_module

    codex_module.load_codex_sdk()

    assert codex_module._sdk_loaded is True
    assert isinstance(codex_module.AgentMessageDeltaNotification, type)
    assert isinstance(codex_module.ItemStartedNotification, type)
