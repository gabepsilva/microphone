"""The composition root: what ``main`` builds and how it connects it.

``main`` is wiring, and wiring is exactly what a refactor breaks silently — a
hook left unassigned or a channel never registered raises nothing and fails no
other test. So every collaborator is faked at ``cli``'s own import boundary
and the assertions are about the connections: which hooks are bound, which
channels reach the session, and what is left out when Them or TTS is off.

Nothing here fakes ``main`` itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_codex import cli
from voice_codex.domain import RESPONSE_POLICIES, TurnLatencyEstimator

MIC = {"name": "Yeti"}
THEM_OUTPUT = {
    "name": "meeting",
    "monitor": "meeting.monitor",
    "description": "Voice Codex Meeting",
}
PLAYBACK = {"name": "alsa_output.pci", "description": "Speakers"}


class FakeTUI:
    """Stand in for the Textual interface without mounting one."""

    def __init__(self, session_state, countdown=None, speech=None, **hooks):
        self.session_state = session_state
        self.countdown = countdown
        self.speech = speech
        self.hook_arguments = hooks
        self.hooks = SimpleNamespace(**hooks)
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
        self.latency = TurnLatencyEstimator()
        self.prefired = []

    def ingest(self, speaker, text, respond):
        self.ingested.append((speaker, text, respond))

    def prefire(self, speaker, text):
        self.prefired.append((speaker, text))
        return True

    def commit_prefire(self, _speaker):
        return True

    def cancel_prefire(self, _speaker):
        return True

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

    def is_speaking(self):
        return False


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

    # The real lock is a file in the user's runtime directory, so leaving it
    # unfaked makes every ``main`` test fail whenever a session happens to be
    # running on the machine. The lock has its own tests below.
    monkeypatch.setattr(cli, "acquire_single_instance_lock", lambda *a, **k: None)
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
    monkeypatch.setattr(cli.sys, "argv", ["voice_codex.py", "--config", str(config)])

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


def test_the_speaker_mute_box_stops_the_them_listener(wiring) -> None:
    """The sidebar's speaker checkbox has to reach the channel it names."""
    wiring["them_output"] = THEM_OUTPUT

    cli.main()
    tui, channels, _, _ = wiring["session"]
    them_listener = channels[1][1]

    tui.hooks.on_them_mute(True)

    assert them_listener.muted is True
    assert channels[0][1].muted is False


def test_a_session_without_a_speaker_channel_binds_no_speaker_mute(wiring) -> None:
    cli.main()
    tui, _, _, _ = wiring["session"]

    assert getattr(tui.hooks, "on_them_mute", None) is None


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
    assert tui.hooks.on_mute is not None


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


def test_one_turn_silence_is_shared_by_every_part_that_needs_it(wiring) -> None:
    """Four copies of the window would be four chances to update three."""
    cli.main()
    tui, channels, _, _ = wiring["session"]
    wiring["them_output"] = THEM_OUTPUT

    windows = {id(listener.turn_silence) for _, listener in channels}
    windows.add(id(tui.countdown.window))

    assert len(windows) == 1


def test_editing_the_window_reaches_the_listeners(wiring) -> None:
    wiring["them_output"] = THEM_OUTPUT

    cli.main()
    tui, channels, _, _ = wiring["session"]

    assert tui.hooks.on_turn_silence(1.25) == 1.25
    assert [listener.turn_silence.seconds for _, listener in channels] == [1.25, 1.25]
    assert tui.countdown.window.seconds == 1.25


def test_the_sidebar_can_see_whether_speech_is_still_playing(wiring) -> None:
    """Without this the state field goes idle while audio is still coming out."""
    wiring["tts_enabled"] = True

    cli.main()
    tui, _, conversation, _ = wiring["session"]

    assert tui.speech is conversation.tts


def test_a_silent_session_gives_the_sidebar_no_speech_to_poll(wiring) -> None:
    cli.main()
    tui, _, _, _ = wiring["session"]

    assert tui.speech is None


def saved_config(tmp_path):
    """Read back what the session wrote to the file it started from."""
    from voice_codex.config import load_startup_config

    return load_startup_config(tmp_path / "voice.yaml")


def test_a_sidebar_change_is_written_to_the_file_the_session_started_from(
    wiring, tmp_path
) -> None:
    cli.main()
    tui, _, _, _ = wiring["session"]

    tui.hooks.on_policy("user")

    assert saved_config(tmp_path)["codex_after"] == "user"


def test_every_sidebar_control_is_remembered(wiring, tmp_path) -> None:
    wiring["tts_enabled"] = True

    cli.main()
    tui, _, _, _ = wiring["session"]

    tui.hooks.on_policy("them")
    tui.hooks.on_codex_model("gpt-5.6-sol")
    tui.hooks.on_codex_effort("high")
    tui.hooks.on_turn_silence(1.25)
    tui.hooks.on_tts(False)

    saved = saved_config(tmp_path)
    assert saved["codex_after"] == "them"
    assert saved["codex_model"] == "gpt-5.6-sol"
    assert saved["codex_reasoning"] == "high"
    assert saved["turn_silence"] == 1.25
    assert saved["tts"] == "off"


def test_a_refused_change_is_not_remembered(wiring, tmp_path) -> None:
    """A silent session cannot switch speech on, so it must not save that it did."""
    cli.main()
    tui, _, _, _ = wiring["session"]
    # Something else has to be saved first, or the file is never written at
    # all and "it did not record the refusal" would pass for the wrong reason.
    tui.hooks.on_policy("user")

    assert tui.hooks.on_tts(True) is False

    assert saved_config(tmp_path)["tts"] == "off"


def test_a_clamped_window_is_saved_as_the_value_in_force(wiring, tmp_path) -> None:
    cli.main()
    tui, _, _, _ = wiring["session"]

    tui.hooks.on_turn_silence(0.01)

    assert saved_config(tmp_path)["turn_silence"] == 0.25


def test_a_session_reopens_with_what_the_last_one_left(wiring, tmp_path) -> None:
    """The whole point: the file a session writes is the file it next reads."""
    from voice_codex.startup import parse_startup_args

    cli.main()
    tui, _, _, _ = wiring["session"]
    tui.hooks.on_turn_silence(1.25)
    tui.hooks.on_codex_effort("high")

    _, args = parse_startup_args(["--config", str(tmp_path / "voice.yaml")])

    assert args.turn_silence == 1.25
    assert args.codex_reasoning == "high"


def test_a_single_instance_lock_is_acquired_once(monkeypatch, tmp_path) -> None:
    """The process should only open and lock once."""
    calls: list[tuple[int, int]] = []

    class FakeLockFile:
        def __init__(self):
            self.closed = False

        def fileno(self):
            return 7

        def close(self):
            self.closed = True

    lock_file = FakeLockFile()
    monkeypatch.setattr(cli, "_INSTANCE_LOCK", None)
    monkeypatch.setattr(
        Path,
        "open",
        lambda _path, *_args, **_kwargs: lock_file,
    )
    monkeypatch.setattr(cli.fcntl, "flock", lambda fd, mode: calls.append((fd, mode)))

    cli.acquire_single_instance_lock(tmp_path / "voice.lock")
    cli.acquire_single_instance_lock(tmp_path / "voice.lock")

    assert calls == [(7, cli.fcntl.LOCK_EX | cli.fcntl.LOCK_NB)]
    assert cli._INSTANCE_LOCK is lock_file


def test_a_running_session_blocks_a_second_lock(monkeypatch, tmp_path) -> None:
    """Contention should fail with a clear runtime error."""

    class FakeLockFile:
        def __init__(self):
            self.closed = False

        def fileno(self):
            return 11

        def close(self):
            self.closed = True

    lock_file = FakeLockFile()
    monkeypatch.setattr(cli, "_INSTANCE_LOCK", None)
    monkeypatch.setattr(
        Path,
        "open",
        lambda _path, *_args, **_kwargs: lock_file,
    )

    def blocked_lock(_fd, _mode):
        raise BlockingIOError

    monkeypatch.setattr(cli.fcntl, "flock", blocked_lock)

    with pytest.raises(RuntimeError, match="already running"):
        cli.acquire_single_instance_lock(tmp_path / "voice.lock")

    assert lock_file.closed is True
    assert cli._INSTANCE_LOCK is None


def test_releasing_a_single_instance_lock_closes_the_file(monkeypatch) -> None:
    closed: list[bool] = []

    class FakeLockFile:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "_INSTANCE_LOCK", FakeLockFile())

    cli._release_single_instance_lock()

    assert closed == [True]
    assert cli._INSTANCE_LOCK is None


def test_release_lock_is_a_noop_when_no_lock_is_held(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_INSTANCE_LOCK", None)
    cli._release_single_instance_lock()
    assert cli._INSTANCE_LOCK is None


def test_the_lock_file_is_named_for_the_user_who_owns_it() -> None:
    """A shared fallback directory must not hand one user another's lock."""
    assert cli.LOCK_PATH.name == f"voice-codex-{os.getuid()}.lock"


@pytest.mark.usefixtures("wiring")
def test_asking_for_help_does_not_wait_on_the_single_instance_lock(monkeypatch) -> None:
    """``--help`` describes the program; it starts no session to conflict with.

    Taking the lock before parsing made every argument-only invocation fail
    while a session was running, which is exactly when someone reads the help.
    """

    def already_running(*_args, **_kwargs):
        raise RuntimeError("Another voice_codex session is already running.")

    monkeypatch.setattr(cli, "acquire_single_instance_lock", already_running)
    cli.sys.argv.append("--help")

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0


@pytest.mark.usefixtures("wiring")
def test_the_lock_is_held_before_the_meeting_devices_are_built(monkeypatch) -> None:
    """Sweeping stale sinks is only safe while no other session owns any."""
    order: list[str] = []

    monkeypatch.setattr(
        cli, "acquire_single_instance_lock", lambda *_a, **_k: order.append("lock")
    )
    resolve = cli.resolve_startup_selection
    monkeypatch.setattr(
        cli,
        "resolve_startup_selection",
        lambda args: (order.append("resolve"), resolve(args))[1],
    )

    cli.main()

    assert order == ["lock", "resolve"]


@pytest.mark.usefixtures("wiring")
def test_a_save_config_failure_is_reported_via_parser_error(monkeypatch) -> None:
    target = Path("/tmp/saved-config.yaml")
    cli.sys.argv.extend(["--save-config", str(target)])
    monkeypatch.setattr(
        cli,
        "save_startup_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cannot save")),
    )

    with pytest.raises(SystemExit):
        cli.main()
