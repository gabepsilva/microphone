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
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_codex import cli
from voice_codex.domain import RESPONSE_POLICIES, TurnLatencyEstimator

MIC = {"name": "Yeti"}
THEM_APPLICATION = "ZOOM VoiceEngine"
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
        self.notes: list[str] = []
        self.them_streams: list[tuple[str, str]] = []

    def set_audio(self, channel, device):
        self.audio[channel] = device

    def set_codex(self, **fields):
        self.codex_fields.update(fields)

    def note(self, text):
        self.notes.append(text)

    def close_speaker(self, speaker):
        self.audio.pop(speaker, None)

    def set_them_streams(self, applications):
        self.them_streams = list(applications)


class FakeTranscriber:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.listeners = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.muted = False

    def add_listener(self, listener):
        self.listeners.append(listener)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def set_muted(self, muted):
        self.muted = muted


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
        return SimpleNamespace(
            device_index=3,
            device=MIC,
            tts_enabled=built["tts_enabled"],
            tts_provider="piper",
            them_stream=built["them_stream"],
            tts_output=built["tts_output"],
            policy=RESPONSE_POLICIES["them"],
        )

    def run_session(tui, channels, conversation, them=None):
        built["session"] = (tui, channels, conversation)
        built["them"] = them

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
        lambda selection, args: (
            FakeTTS(
                args.tts_voice,
                (
                    selection.tts_output["name"]
                    if selection.tts_output is not None
                    else None
                ),
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
        cli, "ApplicationStreamTranscriber", lambda **kwargs: FakeTranscriber(**kwargs)
    )
    # The far end is built on a worker thread when one is chosen. Tests drive
    # that choice themselves, through the same reconcile the worker calls, so
    # what they assert on is the real wiring rather than a race with it.
    monkeypatch.setattr(cli.ThemChannel, "start", lambda self: None)
    monkeypatch.setattr(cli.ApplicationRefresher, "start", lambda self: None)
    monkeypatch.setattr(
        cli, "get_model_for_language", lambda **kwargs: ("model-path", "arch")
    )
    monkeypatch.setattr("voice_codex.tui.VoiceCodexTUI", FakeTUI)
    monkeypatch.setattr(cli.sys, "argv", ["voice_codex.py", "--config", str(config)])

    built.update(tts_enabled=False, them_stream=None, tts_output=None)
    return built


def test_a_user_only_session_registers_one_channel(wiring) -> None:
    cli.main()
    tui, channels, _ = wiring["session"]

    assert len(channels) == 1
    transcriber, listener = channels[0]
    assert listener.speaker == "User Voice"
    assert transcriber.listeners == [listener]
    assert tui.audio == {"mic": "Yeti"}


def test_a_chosen_application_opens_the_far_end_channel(wiring) -> None:
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    tui, _, _ = wiring["session"]
    them = wiring["them"]
    them.reconcile()

    assert them.listener.speaker == "Them"
    assert them.transcriber.kwargs["tap"].application == THEM_APPLICATION
    assert them.transcriber.listeners == [them.listener]
    assert tui.audio["them"] == THEM_APPLICATION


def test_a_session_with_no_application_opens_nothing(wiring) -> None:
    """The model this would load is the whole reason it is not built up front."""
    cli.main()
    them = wiring["them"]
    them.reconcile()

    assert them.transcriber is None
    assert "them" not in wiring["session"][0].audio


def test_switching_applications_keeps_the_channel_and_moves_the_tap(wiring) -> None:
    """Rebuilding would reload a speech model to answer a change of name."""
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    them = wiring["them"]
    them.reconcile()
    opened = them.transcriber

    them.select("Brave")
    them.reconcile()

    assert them.transcriber is opened
    assert them.tap.application == "Brave"
    assert wiring["session"][0].audio["them"] == "Brave"


def test_dropping_the_application_closes_the_channel_it_had(wiring) -> None:
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    them = wiring["them"]
    them.reconcile()
    opened = them.transcriber

    them.select(None)
    them.reconcile()

    assert (opened.stopped, opened.closed) == (True, True)
    assert them.transcriber is None
    assert wiring["session"][0].hooks.on_them_mute is None


def test_the_speaker_mute_box_stops_the_them_listener(wiring) -> None:
    """The sidebar's speaker checkbox has to reach the channel it names."""
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["them"]
    them.reconcile()

    tui.hooks.on_them_mute(True)

    assert them.listener.muted is True
    assert channels[0][1].muted is False


def test_the_speaker_mute_box_stops_the_them_capture(wiring) -> None:
    """Muting has to reach the transcriber, or it only discards finished work."""
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["them"]
    them.reconcile()

    tui.hooks.on_them_mute(True)

    assert them.transcriber.muted is True
    assert channels[0][0].muted is False


def test_the_microphone_mute_box_stops_the_microphone_capture(wiring) -> None:
    """The same for the channel the user speaks on."""
    cli.main()
    tui, channels, _ = wiring["session"]

    tui.hooks.on_mute(True)

    assert channels[0][0].muted is True
    assert channels[0][1].muted is True


def test_unmuting_reaches_the_capture_layer_too(wiring) -> None:
    """A gate that only ever closes would end the session in silence."""
    cli.main()
    tui, channels, _ = wiring["session"]

    tui.hooks.on_mute(True)
    tui.hooks.on_mute(False)

    assert channels[0][0].muted is False
    assert channels[0][1].muted is False


def test_a_session_without_a_speaker_channel_binds_no_speaker_mute(wiring) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    assert getattr(tui.hooks, "on_them_mute", None) is None


def test_every_interface_hook_is_bound_before_the_session_runs(wiring) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

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
    tui, _, conversation = wiring["session"]

    tui.hooks.on_user_text("what time is it?")

    assert conversation.ingested == [("User Text", "what time is it?", True)]


def test_a_session_without_speech_reports_that_it_cannot_be_switched_on(
    wiring,
) -> None:
    """The toggle must say a silent session is silent, not appear to enable it."""
    cli.main()
    tui, _, conversation = wiring["session"]

    assert conversation.tts is None
    assert tui.hooks.on_tts(True) is False


def test_speech_is_routed_to_the_chosen_playback_sink(wiring) -> None:
    wiring["tts_enabled"] = True
    wiring["tts_output"] = PLAYBACK

    cli.main()
    tui, _, conversation = wiring["session"]

    assert conversation.tts.output_sink == "alsa_output.pci"
    assert conversation.tts.voice == "en_US-lessac-medium"
    assert tui.hooks.on_tts(True) is True


def test_the_codex_thread_is_shown_in_the_sidebar(wiring) -> None:
    cli.main()
    tui, _, conversation = wiring["session"]

    assert tui.codex_fields["thread"] == conversation.thread.id


def test_the_startup_selection_is_saved_when_asked(wiring, tmp_path) -> None:
    saved = tmp_path / "saved.yaml"
    wiring["them_stream"] = THEM_APPLICATION
    cli.sys.argv.extend(["--save-config", str(saved)])

    cli.main()

    written = saved.read_text(encoding="utf-8")
    assert 'microphone: "Yeti"' in written
    assert 'them_stream: "ZOOM VoiceEngine"' in written


def test_a_silent_session_builds_no_speech() -> None:
    selection = SimpleNamespace(
        tts_enabled=False, tts_provider="piper", tts_output=None
    )

    assert cli.build_speech(selection, SimpleNamespace(tts_voice="v")) is None


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
    selection = SimpleNamespace(
        tts_enabled=True,
        tts_provider="edge",
        tts_output={"name": "alsa_output.pci"},
    )

    cli.build_speech(selection, SimpleNamespace(tts_voice="en-US-AndrewNeural"))

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
    selection = SimpleNamespace(tts_enabled=True, tts_provider="piper", tts_output=None)

    cli.build_speech(selection, SimpleNamespace(tts_voice="v"))

    assert started == [None]


def test_one_turn_silence_is_shared_by_every_part_that_needs_it(wiring) -> None:
    """Four copies of the window would be four chances to update three."""
    cli.main()
    tui, channels, _ = wiring["session"]
    wiring["them_stream"] = THEM_APPLICATION

    windows = {id(listener.turn_silence) for _, listener in channels}
    windows.add(id(tui.countdown.window))

    assert len(windows) == 1


def test_editing_the_window_reaches_the_listeners(wiring) -> None:
    wiring["them_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["them"]
    them.reconcile()

    assert tui.hooks.on_turn_silence(1.25) == 1.25
    windows = [listener.turn_silence.seconds for _, listener in channels]

    assert [*windows, them.listener.turn_silence.seconds] == [1.25, 1.25]
    assert tui.countdown.window.seconds == 1.25


def test_the_sidebar_can_see_whether_speech_is_still_playing(wiring) -> None:
    """Without this the state field goes idle while audio is still coming out."""
    wiring["tts_enabled"] = True

    cli.main()
    tui, _, conversation = wiring["session"]

    assert tui.speech is conversation.tts


def test_a_silent_session_gives_the_sidebar_no_speech_to_poll(wiring) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    assert tui.speech is None


def saved_config(tmp_path):
    """Read back what the session wrote to the file it started from."""
    from voice_codex.config import load_startup_config

    return load_startup_config(tmp_path / "voice.yaml")


def test_a_sidebar_change_is_written_to_the_file_the_session_started_from(
    wiring, tmp_path
) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_policy("user")

    assert saved_config(tmp_path)["codex_after"] == "user"


def test_every_sidebar_control_is_remembered(wiring, tmp_path) -> None:
    wiring["tts_enabled"] = True

    cli.main()
    tui, _, _ = wiring["session"]

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
    tui, _, _ = wiring["session"]
    # Something else has to be saved first, or the file is never written at
    # all and "it did not record the refusal" would pass for the wrong reason.
    tui.hooks.on_policy("user")

    assert tui.hooks.on_tts(True) is False

    assert saved_config(tmp_path)["tts"] == "off"


def test_a_clamped_window_is_saved_as_the_value_in_force(wiring, tmp_path) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_turn_silence(0.01)

    assert saved_config(tmp_path)["turn_silence"] == 0.25


def test_a_session_reopens_with_what_the_last_one_left(wiring, tmp_path) -> None:
    """The whole point: the file a session writes is the file it next reads."""
    from voice_codex.startup import parse_startup_args

    cli.main()
    tui, _, _ = wiring["session"]
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


# --------------------------------------------------------------------------
# The far end's channel, built and retired while the session runs
# --------------------------------------------------------------------------


class FakeGate:
    def __init__(self):
        self.available = frozenset({"User Voice"})

    def set_available(self, available):
        self.available = frozenset(available)


def them_channel(open_channel=None, close_channel=None):
    """A far-end channel over fakes, with its worker thread left unstarted."""
    tui = FakeTUI(SimpleNamespace(), on_them_mute=None)
    opened: list[object] = []
    closed: list[object] = []

    def open_default(tap):
        transcriber = FakeTranscriber(tap=tap)
        listener = SimpleNamespace(speaker="Them", set_muted=lambda muted: None)
        opened.append((transcriber, listener))
        return transcriber, listener

    channel = cli.ThemChannel(
        tui,
        FakeGate(),
        open_channel or open_default,
        close_channel or (lambda t, listener: closed.append((t, listener))),
    )
    return channel, tui, opened, closed


def test_choosing_an_application_makes_them_answerable() -> None:
    """A policy naming Them cannot fire until a far end actually exists."""
    channel, _, _, _ = them_channel()

    channel.select("Brave")
    channel.reconcile()

    assert channel.gate.available == frozenset({"User Voice", "Them"})


def test_dropping_the_application_makes_them_unanswerable_again() -> None:
    channel, _, _, _ = them_channel()
    channel.select("Brave")
    channel.reconcile()

    channel.select(None)
    channel.reconcile()

    assert channel.gate.available == frozenset({"User Voice"})


def test_the_newest_choice_wins_rather_than_each_one_building() -> None:
    """Moving through the picker produces a choice per keystroke."""
    channel, _, opened, _ = them_channel()

    channel.select("Brave")
    channel.select("Chromium")
    channel.select("ZOOM VoiceEngine")
    channel.reconcile()

    assert len(opened) == 1
    assert channel.tap.application == "ZOOM VoiceEngine"


def test_asking_again_for_what_is_already_open_changes_nothing() -> None:
    channel, _, opened, closed = them_channel()
    channel.select("Brave")
    channel.reconcile()

    channel.select("Brave")
    channel.reconcile()

    assert (len(opened), len(closed)) == (1, 0)


def test_a_far_end_that_cannot_be_opened_leaves_the_session_running() -> None:
    """The state a failure lands in is the one the session was already in."""

    def refuse(_tap):
        raise RuntimeError("no pw-record")

    channel, tui, _, _ = them_channel(open_channel=refuse)

    channel.select("Brave")
    channel.reconcile()

    assert channel.transcriber is None
    assert channel.desired is None
    assert "could not listen to Brave: no pw-record" in tui.notes


def test_a_failed_far_end_is_not_retried_on_every_later_choice() -> None:
    """Forgetting the request is what keeps one failure from becoming a loop."""
    attempts: list[str] = []

    def refuse(tap):
        attempts.append(tap.application)
        raise RuntimeError("no")

    channel, _, _, _ = them_channel(open_channel=refuse)
    channel.select("Brave")
    channel.reconcile()

    channel.reconcile()

    assert attempts == ["Brave"]


def test_the_worker_serves_choices_until_the_session_closes() -> None:
    channel, _tui, opened, closed = them_channel()
    channel.start()
    channel.select("Brave")
    deadline = time.monotonic() + 10
    while not opened and time.monotonic() < deadline:
        time.sleep(0.01)

    channel.close()

    assert len(opened) == 1
    assert len(closed) == 1
    assert channel.worker is None


def test_closing_a_channel_that_was_never_opened_is_quiet() -> None:
    channel, _, opened, closed = them_channel()

    channel.close()

    assert (opened, closed) == ([], [])


def test_starting_the_channel_twice_serves_it_once() -> None:
    channel, _, _, _ = them_channel()
    channel.start()
    worker = channel.worker
    try:
        channel.start()

        assert channel.worker is worker
    finally:
        channel.close()


@pytest.mark.parametrize(
    ("application", "recorded"),
    [(None, "none"), ("Brave", "Brave")],
)
def test_a_chosen_application_is_saved_the_way_the_parser_reads_it_back(
    application, recorded
) -> None:
    assert cli.them_stream_setting(application) == recorded
