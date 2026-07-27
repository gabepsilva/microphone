"""The composition root: what ``main`` builds and how it connects it.

``main`` is wiring, and wiring is exactly what a refactor breaks silently — a
hook left unassigned or a channel never registered raises nothing and fails no
other test. So every collaborator is faked at ``cli``'s own import boundary
and the assertions are about the connections: which hooks are bound, which
channels reach the session, and what is left out when Them or TTS is off.

Nothing here fakes ``main`` itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voice_codex import cli
from voice_codex.domain import RESPONSE_POLICIES

MIC = {"name": "Yeti"}
THEM_OUTPUT = {
    "name": "meeting",
    "monitor": "meeting.monitor",
    "description": "Voice Codex Meeting",
}
PLAYBACK = {"name": "alsa_output.pci", "description": "Speakers"}


class FakeTUI:
    """Stand in for the Textual interface without mounting one."""

    def __init__(self, session_state, on_policy=None):
        self.session_state = session_state
        self.on_policy = on_policy
        self.hooks = SimpleNamespace()
        self.audio: dict[str, str] = {}
        self.codex_fields: dict[str, object] = {}
        self.output = None

    def set_audio(self, channel, device):
        self.audio[channel] = device

    def set_codex(self, **fields):
        self.codex_fields.update(fields)

    def set_output(self, description):
        self.output = description


class FakeTranscriber:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)


class FakeConversation:
    def __init__(self, settings, display, tts):
        self.settings = settings
        self.display = display
        self.tts = tts
        self.thread = SimpleNamespace(id="thread-1")
        self.ingested = []

    def ingest(self, speaker, text, respond):
        self.ingested.append((speaker, text, respond))

    def interrupt(self):
        pass

    def request_model(self, _model):
        return True

    def request_reasoning_effort(self, _effort):
        return True


class FakeTTS:
    def __init__(self, voice, output_sink=None):
        self.voice = voice
        self.output_sink = output_sink

    def set_enabled(self, _enabled):
        return True


@pytest.fixture
def wiring(monkeypatch, tmp_path):
    """Fake every adapter ``main`` reaches for and record what it built."""
    config = tmp_path / "voice.yaml"
    config.write_text("", encoding="utf-8")
    built: dict[str, object] = {}

    def resolve(_args):
        return (
            SimpleNamespace(
                device_index=3,
                device=MIC,
                tts_enabled=built["tts_enabled"],
                tts_provider="piper",
                them_output=built["them_output"],
                them_output_setting="isolated",
                playback_output=built["playback_output"],
                policy=RESPONSE_POLICIES["them"],
            ),
            built["virtual_meeting"],
        )

    def run_session(tui, channels, conversation, virtual_meeting):
        built["session"] = (tui, channels, conversation, virtual_meeting)

    monkeypatch.setattr(cli, "resolve_startup_selection", resolve)
    monkeypatch.setattr(cli, "run_session", run_session)
    monkeypatch.setattr(cli, "print_startup_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "build_speech",
        lambda selection, args, playback_output: (
            FakeTTS(
                args.tts_voice,
                playback_output["name"] if playback_output is not None else None,
            )
            if selection.tts_enabled
            else None
        ),
    )
    monkeypatch.setattr(cli, "CodexConversation", FakeConversation)
    monkeypatch.setattr(
        cli, "metered_mic_transcriber", lambda **kwargs: FakeTranscriber(**kwargs)
    )
    monkeypatch.setattr(
        cli, "PulseMonitorTranscriber", lambda **kwargs: FakeTranscriber(**kwargs)
    )
    monkeypatch.setattr(
        cli, "get_model_for_language", lambda **kwargs: ("model-path", "arch")
    )
    monkeypatch.setattr("voice_codex.tui.VoiceCodexTUI", FakeTUI)
    monkeypatch.setattr(cli.sys, "argv", ["voice-codex.py", "--config", str(config)])

    built.update(
        tts_enabled=False,
        them_output=None,
        playback_output=None,
        virtual_meeting=None,
    )
    return built


def test_a_user_only_session_registers_one_channel(wiring) -> None:
    cli.main()
    tui, channels, _, virtual_meeting = wiring["session"]

    assert len(channels) == 1
    transcriber, listener = channels[0]
    assert listener.speaker == "User Voice"
    assert transcriber.listeners == [listener]
    assert virtual_meeting is None
    assert tui.audio == {"mic": "Yeti"}


def test_a_them_output_adds_a_second_channel_in_order(wiring) -> None:
    wiring["them_output"] = THEM_OUTPUT

    cli.main()
    tui, channels, _, _ = wiring["session"]

    assert [listener.speaker for _, listener in channels] == ["User Voice", "Them"]
    assert channels[1][0].kwargs["monitor"] == "meeting.monitor"
    assert tui.audio["them"] == "Voice Codex Meeting"


def test_every_interface_hook_is_bound_before_the_session_runs(wiring) -> None:
    cli.main()
    tui, _, _, _ = wiring["session"]

    assert {
        "on_user_text",
        "on_interrupt",
        "on_codex_model",
        "on_codex_effort",
        "on_tts",
        "on_mute",
    } <= set(vars(tui.hooks))


def test_typed_text_always_requests_a_reply(wiring) -> None:
    cli.main()
    tui, _, conversation, _ = wiring["session"]

    tui.hooks.on_user_text("what time is it?")

    assert conversation.ingested == [("User Text", "what time is it?", True)]


def test_a_session_without_speech_reports_that_it_cannot_be_switched_on(
    wiring,
) -> None:
    """The toggle must say a silent session is silent, not appear to enable it."""
    cli.main()
    tui, _, conversation, _ = wiring["session"]

    assert conversation.tts is None
    assert tui.hooks.on_tts(True) is False


def test_speech_is_routed_to_the_chosen_playback_sink(wiring) -> None:
    wiring["tts_enabled"] = True
    wiring["playback_output"] = PLAYBACK

    cli.main()
    tui, _, conversation, _ = wiring["session"]

    assert conversation.tts.output_sink == "alsa_output.pci"
    assert conversation.tts.voice == "en_US-lessac-medium"
    assert tui.output == "Speakers"
    assert tui.hooks.on_tts(True) is True


def test_an_isolated_meeting_is_closed_when_the_interface_quits(wiring) -> None:
    closed: list[bool] = []
    wiring["them_output"] = THEM_OUTPUT
    wiring["virtual_meeting"] = SimpleNamespace(close=lambda: closed.append(True))

    cli.main()
    tui, _, _, virtual_meeting = wiring["session"]

    tui.hooks.on_quit()

    assert closed == [True]
    assert virtual_meeting is wiring["virtual_meeting"]


def test_the_codex_thread_is_shown_in_the_sidebar(wiring) -> None:
    cli.main()
    tui, _, conversation, _ = wiring["session"]

    assert tui.codex_fields["thread"] == conversation.thread.id


def test_the_startup_selection_is_saved_when_asked(wiring, tmp_path) -> None:
    saved = tmp_path / "saved.yaml"
    wiring["them_output"] = THEM_OUTPUT
    cli.sys.argv.extend(["--save-config", str(saved)])

    cli.main()

    written = saved.read_text(encoding="utf-8")
    assert 'microphone: "Yeti"' in written
    assert 'them_output: "isolated"' in written


def test_a_silent_session_builds_no_speech() -> None:
    selection = SimpleNamespace(tts_enabled=False, tts_provider="piper")

    assert cli.build_speech(selection, SimpleNamespace(tts_voice="v"), None) is None


def test_speech_is_built_for_the_chosen_provider_and_output(monkeypatch) -> None:
    started: list[tuple] = []
    monkeypatch.setattr(
        cli.SwitchableSpeech,
        "start",
        classmethod(
            lambda _cls, provider, voice, output_sink=None: started.append(
                (provider, voice, output_sink)
            )
        ),
    )
    selection = SimpleNamespace(tts_enabled=True, tts_provider="edge")

    cli.build_speech(
        selection,
        SimpleNamespace(tts_voice="en-US-AndrewNeural"),
        {"name": "alsa_output.pci"},
    )

    assert started == [("edge", "en-US-AndrewNeural", "alsa_output.pci")]


def test_speech_plays_through_the_default_output_when_none_was_chosen(
    monkeypatch,
) -> None:
    started: list[str | None] = []
    monkeypatch.setattr(
        cli.SwitchableSpeech,
        "start",
        classmethod(
            lambda _cls, provider, voice, output_sink=None: started.append(output_sink)
        ),
    )
    selection = SimpleNamespace(tts_enabled=True, tts_provider="piper")

    cli.build_speech(selection, SimpleNamespace(tts_voice="v"), None)

    assert started == [None]
