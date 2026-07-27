"""Startup argument, selection, and session-lifecycle behavior.

These cover the logic extracted from ``main`` so it can be exercised without a
microphone, a PipeWire server, or a Codex account.
"""

from __future__ import annotations

import io
from dataclasses import replace
from types import SimpleNamespace

import pytest

from voice_codex import catalog, startup
from voice_codex.domain import POLICY_NAMES, RESPONSE_POLICIES, SpeakerGate
from voice_codex.listener import TranscriptSubmitter, tts_switch
from voice_codex.startup import (
    StartupSelection,
    build_session_state,
    parse_startup_args,
    print_startup_summary,
    resolve_startup_selection,
    run_session,
    startup_settings,
    them_output_name,
)


def write_config(tmp_path, body):
    path = tmp_path / "voice.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def empty_config(tmp_path):
    return write_config(tmp_path, "")


BASE_SELECTION = StartupSelection(
    device_index=3,
    device={"name": "Yeti"},
    tts_enabled=False,
    tts_provider="piper",
    them_output=None,
    them_output_setting="none",
    playback_output=None,
    policy=RESPONSE_POLICIES["them"],
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
    assert args.codex_fast is False
    assert (args.tts, args.codex_after, args.microphone) == (None, None, None)


def test_command_line_overrides_the_startup_config_file(tmp_path) -> None:
    config = write_config(
        tmp_path,
        'microphone: "Config Mic"\ntts: "on"\ncodex_after: "user"\n',
    )

    _, args = parse_startup_args(["--config", config, "--microphone", "Flag Mic"])

    assert args.microphone == "Flag Mic"
    assert (args.tts, args.codex_after) == ("on", "user")


def test_an_unreadable_startup_config_exits_instead_of_prompting(tmp_path) -> None:
    missing = str(tmp_path / "absent.yaml")

    with pytest.raises(SystemExit, match="2"):
        parse_startup_args(["--config", missing])


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


@pytest.mark.parametrize(
    ("them_output", "expected"),
    [
        (None, "none"),
        ({"isolated": True}, "isolated"),
        ({"name": "alsa_output.pci", "description": "Speakers"}, "alsa_output.pci"),
    ],
)
def test_them_output_is_named_the_way_a_saved_config_records_it(
    them_output, expected
) -> None:
    assert them_output_name(them_output) == expected


def test_saved_settings_round_trip_back_to_the_same_selection() -> None:
    chosen = selection(
        tts_enabled=True,
        them_output_setting="isolated",
        playback_output={"name": "alsa_output.pci", "description": "Speakers"},
        policy=RESPONSE_POLICIES["both"],
    )

    assert startup_settings(chosen) == {
        "microphone": "Yeti",
        "tts": "on",
        "tts_provider": "piper",
        "them_output": "isolated",
        "playback_output": "alsa_output.pci",
        "codex_after": "both",
    }


def test_saved_settings_record_no_playback_output_when_none_was_chosen() -> None:
    assert startup_settings(selection())["playback_output"] is None


def test_the_startup_summary_reports_the_resolved_choices(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path), "--codex-fast"])
    stream = io.StringIO()

    print_startup_summary(
        args,
        selection(
            tts_enabled=True,
            them_output={"description": "Voice Codex Meeting"},
            playback_output={"name": "alsa", "description": "Speakers"},
        ),
        stream=stream,
    )
    summary = stream.getvalue()

    assert "User microphone: Yeti" in summary
    assert "Them audio output: Voice Codex Meeting" in summary
    assert "Codex response policy: Them" in summary
    assert "Voice turn silence: 3.0s" in summary
    assert "Codex speed: Fast" in summary
    assert "Codex command access: full-access" in summary
    assert "Codex audio: Piper (local) (en_US-lessac-medium)" in summary
    assert "Meeting and TTS playback: Speakers" in summary


def test_the_startup_summary_warns_when_tts_shares_a_direct_them_monitor(
    tmp_path,
) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])
    stream = io.StringIO()

    print_startup_summary(
        args,
        selection(
            tts_enabled=True,
            them_output={"description": "Speakers monitor"},
        ),
        stream=stream,
    )

    assert "may transcribe Codex TTS" in stream.getvalue()


def test_the_startup_summary_omits_the_warning_without_a_them_output(tmp_path) -> None:
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])
    stream = io.StringIO()

    print_startup_summary(args, selection(tts_enabled=True), stream=stream)
    summary = stream.getvalue()

    assert "Them audio output: None" in summary
    assert "may transcribe Codex TTS" not in summary
    assert "Codex speed: Standard" in summary


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


def test_a_direct_them_monitor_is_selected_without_a_virtual_meeting(
    monkeypatch, tmp_path
) -> None:
    monitor = {"name": "sink", "monitor": "sink.monitor", "description": "Speakers"}
    monkeypatch.setattr(
        startup, "choose_microphone", lambda requested: (2, {"name": "M"})
    )
    monkeypatch.setattr(startup, "choose_tts", lambda requested: False)
    monkeypatch.setattr(
        startup, "choose_them_output", lambda requested, require_isolation: monitor
    )
    monkeypatch.setattr(
        startup, "choose_codex_after", lambda requested: ("Them", {"Them"})
    )
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    chosen, virtual_meeting = resolve_startup_selection(args)

    assert virtual_meeting is None
    assert chosen.them_output is monitor
    assert chosen.them_output_setting == "sink"
    assert (chosen.device_index, chosen.tts_enabled) == (2, False)


def test_an_isolated_them_output_builds_a_virtual_meeting_sink(
    monkeypatch, tmp_path, capsys
) -> None:
    playback = {"name": "alsa", "description": "Speakers"}
    transcript_output = {"monitor": "meeting.monitor", "description": "Meeting"}
    monkeypatch.setattr(
        startup, "choose_microphone", lambda requested: (0, {"name": "M"})
    )
    monkeypatch.setattr(startup, "choose_tts", lambda requested: True)
    monkeypatch.setattr(
        startup,
        "choose_them_output",
        lambda requested, require_isolation: {"isolated": True},
    )
    monkeypatch.setattr(startup, "choose_playback_output", lambda requested: playback)
    monkeypatch.setattr(
        startup,
        "VirtualMeetingOutput",
        lambda output: SimpleNamespace(transcript_output=transcript_output),
    )
    monkeypatch.setattr(
        startup, "choose_codex_after", lambda requested: ("User Voice", {"User Voice"})
    )
    _, args = parse_startup_args(["--config", empty_config(tmp_path)])

    chosen, virtual_meeting = resolve_startup_selection(args)

    assert virtual_meeting is not None
    assert chosen.them_output is transcript_output
    assert chosen.them_output_setting == "isolated"
    assert chosen.playback_output is playback
    assert "Voice Codex Meeting" in capsys.readouterr().err


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
    meeting = SimpleNamespace(close=lambda: events.append("close meeting"))

    run_session(tui, channels, conversation, meeting)

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
        "close meeting",
    ]


def test_an_interrupted_session_still_closes_everything(monkeypatch, capsys) -> None:
    def interrupt():
        raise KeyboardInterrupt

    events, tui, channels, conversation = session_parts(monkeypatch, run=interrupt)

    run_session(tui, channels, conversation, None)

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
        saved = startup_settings(selection(policy=policy))

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
