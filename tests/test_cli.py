"""The composition root: what ``main`` builds and how it connects it.

``main`` is wiring, and wiring is exactly what a refactor breaks silently — a
hook left unassigned or a channel never registered raises nothing and fails no
other test. So every collaborator is faked at ``cli``'s own import boundary
and the assertions are about the connections: which hooks are bound, which
channels reach the session, and what is left out when Audio or TTS is off.

Nothing here fakes ``main`` itself.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tagalong import cli
from tagalong.catalog import CodexModelOption
from tagalong.domain import RESPONSE_POLICIES, TurnLatencyEstimator, UserTextMessage
from tagalong.tui import SessionState

MIC = {"name": "Yeti"}
THEM_APPLICATION = "ZOOM VoiceEngine"
PLAYBACK = {"name": "alsa_output.pci", "description": "Speakers"}
INPUTS = [
    (
        3,
        {
            "name": "Yeti",
            "default_samplerate": 48000.0,
            "max_input_channels": 1,
        },
    ),
    (
        7,
        {
            "name": "Webcam",
            "default_samplerate": 16000.0,
            "max_input_channels": 1,
        },
    ),
]


class FakeTUI:
    """Stand in for the Textual interface without mounting one."""

    def __init__(self, session_state, countdown=None, speech=None, **hooks):
        self.session_state = session_state
        self.state = session_state
        self.countdown = countdown
        self.speech = speech
        self.hook_arguments = hooks
        self.hooks = SimpleNamespace(**hooks)
        self.codex_fields: dict[str, object] = {}
        self.notes: list[str] = []
        self.audio_streams: list[tuple[str, str]] = []
        self.closed_speakers: list[str] = []

    def set_codex(self, **fields):
        self.codex_fields.update(fields)

    def note(self, text):
        self.notes.append(text)

    def close_speaker(self, speaker):
        self.closed_speakers.append(speaker)

    def set_audio_streams(self, applications):
        self.audio_streams = list(applications)

    def set_microphones(self, microphones):
        self.microphones = list(microphones)


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

    def ingest(self, speaker, text, respond, images=()):
        self.ingested.append((speaker, text, respond, images))

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


class SwitchListener:
    def __init__(self):
        self.muted = False
        self.closed = False

    def set_muted(self, muted):
        self.muted = muted

    def close(self):
        self.closed = True


def managed_microphone(devices=(), discover=None, open_error=None):
    """Build a dynamic microphone around recording capture fakes."""
    tui = FakeTUI(SessionState())
    opened: list[tuple[int, FakeTranscriber, SwitchListener]] = []

    def open_channel(index):
        if open_error is not None:
            raise open_error
        transcriber = FakeTranscriber(device=index)
        listener = SwitchListener()
        opened.append((index, transcriber, listener))
        return transcriber, listener

    def close_channel(transcriber, listener):
        transcriber.stop()
        listener.close()
        transcriber.close()

    microphone = cli.MicrophoneChannel(
        tui,
        open_channel,
        close_channel,
        devices=devices,
        discover=(lambda: list(devices)) if discover is None else discover,
    )
    microphone.poll = 0.001
    return microphone, tui, opened


@pytest.fixture
def wiring(monkeypatch, tmp_path):
    """Fake every adapter ``main`` reaches for and record what it built."""
    config = tmp_path / "tagalong.yaml"
    config.write_text("", encoding="utf-8")
    built: dict[str, object] = {}

    def resolve(_args):
        device = built["input_device"]
        return SimpleNamespace(
            device_index=3 if device is not None else None,
            device=device,
            tts_enabled=built["tts_enabled"],
            tts_provider="piper",
            audio_stream=built["audio_stream"],
            tts_output=built["tts_output"],
            policy=RESPONSE_POLICIES["audio"],
        )

    def run_session(tui, channels, conversation, microphone=None, audio=None):
        microphone_channels = (
            [(microphone.transcriber, microphone.listener)]
            if microphone is not None and microphone.transcriber is not None
            else []
        )
        built["session"] = (tui, [*channels, *microphone_channels], conversation)
        built["microphone"] = microphone
        built["audio"] = audio

    # The real lock is a file in the user's runtime directory, so leaving it
    # unfaked makes every ``main`` test fail whenever a session happens to be
    # running on the machine. The lock has its own tests below.
    monkeypatch.setattr(cli, "acquire_single_instance_lock", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "probe_codex_models",
        lambda: [
            CodexModelOption(
                "gpt-5.6-luna",
                "GPT-5.6 Luna",
                ("none", "low", "medium", "high", "xhigh", "max"),
                "medium",
            ),
            CodexModelOption(
                "gpt-5.6-sol",
                "GPT-5.6 Sol",
                ("none", "low", "medium", "high", "xhigh", "max", "ultra"),
                "low",
            ),
        ],
        raising=False,
    )
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
    monkeypatch.setattr(cli.AudioChannel, "start", lambda self: None)
    monkeypatch.setattr(cli.MicrophoneChannel, "start", lambda self: None)
    monkeypatch.setattr(cli.ApplicationRefresher, "start", lambda self: None)
    monkeypatch.setattr(
        cli, "get_model_for_language", lambda **kwargs: ("model-path", "arch")
    )
    monkeypatch.setattr("tagalong.tui.VoiceCodexTUI", FakeTUI)
    monkeypatch.setattr(cli.sys, "argv", ["tagalong.py", "--config", str(config)])

    built.update(
        input_device=MIC,
        tts_enabled=False,
        audio_stream=None,
        tts_output=None,
    )
    return built


def test_an_unavailable_saved_microphone_waits_without_opening_capture() -> None:
    microphone, _, opened = managed_microphone()

    microphone.select("Yeti")
    microphone.reconcile()

    assert opened == []
    assert microphone.desired == "Yeti"


def test_a_saved_microphone_opens_when_it_appears_later() -> None:
    available: list[tuple[int, dict]] = []
    microphone, tui, opened = managed_microphone(discover=lambda: list(available))
    microphone.select("Yeti")

    microphone.refresh()
    microphone.reconcile()
    available.extend(INPUTS)
    microphone.refresh()
    microphone.reconcile()

    assert [index for index, _, _ in opened] == [3]
    assert microphone.current == "Yeti"
    assert tui.microphones == [
        ("Yeti", "Yeti"),
        ("Webcam", "Webcam"),
    ]


def test_switching_microphones_retires_the_old_capture_before_opening_the_new() -> None:
    microphone, _, opened = managed_microphone(INPUTS)
    microphone.select("Yeti")
    microphone.reconcile()
    _, first_transcriber, first_listener = opened[0]

    microphone.select("Webcam")
    microphone.reconcile()

    assert [index for index, _, _ in opened] == [3, 7]
    assert (first_transcriber.stopped, first_transcriber.closed) == (True, True)
    assert first_listener.closed is True
    assert microphone.current == "Webcam"


def test_selecting_no_microphone_closes_capture_and_unbinds_mute() -> None:
    microphone, tui, opened = managed_microphone(INPUTS)
    microphone.select("Yeti")
    microphone.reconcile()
    _, transcriber, listener = opened[0]

    microphone.select(None)
    microphone.reconcile()

    assert (transcriber.stopped, transcriber.closed, listener.closed) == (
        True,
        True,
        True,
    )
    assert microphone.current is None
    assert tui.hooks.on_mute is None


def test_a_microphone_open_failure_leaves_the_session_running() -> None:
    microphone, tui, opened = managed_microphone(
        INPUTS, open_error=RuntimeError("device busy")
    )

    microphone.select("Yeti")
    microphone.reconcile()

    assert opened == []
    assert microphone.current is None
    assert tui.notes == ["could not listen to Yeti: device busy"]


def test_a_microphone_start_failure_is_cleaned_up_and_reported_once() -> None:
    tui = FakeTUI(SessionState())

    class StartFailingTranscriber(FakeTranscriber):
        def start(self):
            raise RuntimeError("device busy")

    transcriber = StartFailingTranscriber()
    listener = SwitchListener()
    closed: list[tuple[FakeTranscriber, SwitchListener]] = []
    microphone = cli.MicrophoneChannel(
        tui,
        lambda _index: (transcriber, listener),
        lambda opened, heard: closed.append((opened, heard)),
        devices=INPUTS,
        discover=lambda: INPUTS,
    )
    microphone.select("Yeti")

    microphone.reconcile()
    microphone.reconcile()

    assert closed == [(transcriber, listener), (transcriber, listener)]
    assert tui.notes == ["could not listen to Yeti: device busy"]


def test_building_a_microphone_closes_capture_if_the_listener_cannot_be_built(
    monkeypatch,
) -> None:
    transcriber = FakeTranscriber()

    class RefusingSubmitter:
        def channel(self, *_args, **_kwargs):
            raise RuntimeError("listener unavailable")

    parts = SimpleNamespace(
        submitter=RefusingSubmitter(),
        confidence=0.6,
        turn_silence=object(),
        display=object(),
        countdown=object(),
        model_path="model",
        model_arch="arch",
    )
    monkeypatch.setattr(cli, "metered_mic_transcriber", lambda **_kwargs: transcriber)

    with pytest.raises(RuntimeError, match="listener unavailable"):
        cli.open_microphone_channel(parts, object(), object(), 3)

    assert transcriber.closed is True


def test_closing_a_microphone_unregisters_its_listener() -> None:
    events: list[str] = []

    class Transcriber:
        def stop(self):
            events.append("stop capture")

        def close(self):
            events.append("close capture")

    class Listener:
        def close(self):
            events.append("close listener")

    listener = Listener()
    parts = SimpleNamespace(
        submitter=SimpleNamespace(
            remove_listener=lambda removed: events.append(
                "unregister listener" if removed is listener else "wrong listener"
            )
        )
    )

    cli.close_microphone_channel(parts, Transcriber(), listener)

    assert events == [
        "stop capture",
        "close listener",
        "close capture",
        "unregister listener",
    ]


def test_a_microphone_opened_while_muted_starts_muted() -> None:
    microphone, tui, opened = managed_microphone(INPUTS)
    tui.state.mic.muted = True

    microphone.select("Yeti")
    microphone.reconcile()

    assert opened[0][1].muted is True
    assert opened[0][2].muted is True


def test_a_discovery_failure_is_reported_once_until_it_changes() -> None:
    errors = [
        RuntimeError("PortAudio unavailable"),
        RuntimeError("PortAudio unavailable"),
        RuntimeError("different failure"),
    ]
    microphone, tui, _ = managed_microphone(
        discover=lambda: (_ for _ in ()).throw(errors.pop(0))
    )

    microphone.refresh()
    microphone.refresh()
    microphone.refresh()

    assert tui.notes == [
        "could not discover microphones: PortAudio unavailable",
        "could not discover microphones: different failure",
    ]
    assert tui.microphones == []


def test_the_microphone_worker_refreshes_until_it_is_closed() -> None:
    refreshed = threading.Event()
    microphone, _, _ = managed_microphone(discover=lambda: refreshed.set() or INPUTS)

    microphone.start()
    worker = microphone.worker
    microphone.start()
    try:
        assert refreshed.wait(1)
        assert microphone.worker is worker
    finally:
        microphone.close()

    assert microphone.worker is None


def test_a_user_only_session_registers_one_channel(wiring) -> None:
    cli.main()
    _, channels, _ = wiring["session"]

    assert len(channels) == 1
    transcriber, listener = channels[0]
    assert listener.speaker == "Voice"
    assert transcriber.listeners == [listener]


def test_a_session_without_an_input_device_still_reaches_the_interface(wiring) -> None:
    wiring["input_device"] = None

    cli.main()
    tui, channels, conversation = wiring["session"]

    assert channels == []
    assert tui.session_state.microphone is None
    tui.hooks.on_user_text(UserTextMessage("typed while the microphone is absent"))
    assert conversation.ingested == [
        ("Text", "typed while the microphone is absent", True, ())
    ]


def test_a_chosen_application_opens_the_far_end_channel(wiring) -> None:
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    them = wiring["audio"]
    them.reconcile()

    assert them.listener.speaker == "Audio"
    assert them.transcriber.kwargs["tap"].application == THEM_APPLICATION
    assert them.transcriber.listeners == [them.listener]


def test_a_session_with_no_application_opens_nothing(wiring) -> None:
    """The model this would load is the whole reason it is not built up front."""
    cli.main()
    them = wiring["audio"]
    them.reconcile()

    assert them.transcriber is None


def test_switching_applications_keeps_the_channel_and_moves_the_tap(wiring) -> None:
    """Rebuilding would reload a speech model to answer a change of name."""
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    them = wiring["audio"]
    them.reconcile()
    opened = them.transcriber

    them.select("Brave")
    them.reconcile()

    assert them.transcriber is opened
    assert them.tap.application == "Brave"
    assert them.current == "Brave"


def test_dropping_the_application_closes_the_channel_it_had(wiring) -> None:
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    them = wiring["audio"]
    them.reconcile()
    opened = them.transcriber

    them.select(None)
    them.reconcile()

    assert (opened.stopped, opened.closed) == (True, True)
    assert them.transcriber is None
    assert wiring["session"][0].hooks.on_audio_mute is None


def test_the_speaker_mute_box_stops_the_audio_listener(wiring) -> None:
    """The sidebar's speaker checkbox has to reach the channel it names."""
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["audio"]
    them.reconcile()

    tui.hooks.on_audio_mute(True)

    assert them.listener.muted is True
    assert channels[0][1].muted is False


def test_the_speaker_mute_box_stops_the_audio_capture(wiring) -> None:
    """Muting has to reach the transcriber, or it only discards finished work."""
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["audio"]
    them.reconcile()

    tui.hooks.on_audio_mute(True)

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

    assert getattr(tui.hooks, "on_audio_mute", None) is None


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

    tui.hooks.on_user_text(UserTextMessage("what time is it?"))

    assert conversation.ingested == [("Text", "what time is it?", True, ())]


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
    wiring["audio_stream"] = THEM_APPLICATION
    cli.sys.argv.extend(["--save-config", str(saved)])

    cli.main()

    written = saved.read_text(encoding="utf-8")
    assert 'microphone: "Yeti"' in written
    assert 'audio_stream: "ZOOM VoiceEngine"' in written


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
    wiring["audio_stream"] = THEM_APPLICATION

    windows = {id(listener.turn_silence) for _, listener in channels}
    windows.add(id(tui.countdown.window))

    assert len(windows) == 1


def test_editing_the_window_reaches_the_listeners(wiring) -> None:
    wiring["audio_stream"] = THEM_APPLICATION

    cli.main()
    tui, channels, _ = wiring["session"]
    them = wiring["audio"]
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
    from tagalong.config import load_startup_config

    return load_startup_config(tmp_path / "tagalong.yaml")


def test_a_sidebar_change_is_written_to_the_file_the_session_started_from(
    wiring, tmp_path
) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_policy("voice")

    assert saved_config(tmp_path)["taga_after"] == "voice"


def test_every_sidebar_control_is_remembered(wiring, tmp_path) -> None:
    wiring["tts_enabled"] = True

    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_policy("audio")
    tui.hooks.on_codex_model("gpt-5.6-sol")
    tui.hooks.on_codex_effort("high")
    tui.hooks.on_turn_silence(1.25)
    tui.hooks.on_tts(False)

    saved = saved_config(tmp_path)
    assert saved["taga_after"] == "audio"
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
    tui.hooks.on_policy("voice")

    assert tui.hooks.on_tts(True) is False

    assert saved_config(tmp_path)["tts"] == "off"


def test_a_clamped_window_is_saved_as_the_value_in_force(wiring, tmp_path) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_turn_silence(0.01)

    assert saved_config(tmp_path)["turn_silence"] == 0.25


def test_a_session_reopens_with_what_the_last_one_left(wiring, tmp_path) -> None:
    """The whole point: the file a session writes is the file it next reads."""
    from tagalong.startup import parse_startup_args

    cli.main()
    tui, _, _ = wiring["session"]
    tui.hooks.on_turn_silence(1.25)
    tui.hooks.on_codex_effort("high")

    _, args = parse_startup_args(["--config", str(tmp_path / "tagalong.yaml")])

    assert args.turn_silence == 1.25
    assert args.codex_reasoning == "high"


def test_a_catalog_reasoning_effort_saved_by_the_sidebar_reopens(wiring) -> None:
    """A catalog choice outside the legacy fixed list survives the round trip."""
    cli.main()
    tui, _, _ = wiring["session"]
    tui.hooks.on_codex_model("gpt-5.6-sol")
    tui.hooks.on_codex_effort("ultra")

    cli.main()
    reopened_tui, _, conversation = wiring["session"]

    assert (conversation.settings.model, conversation.settings.reasoning_effort) == (
        "gpt-5.6-sol",
        "ultra",
    )
    assert reopened_tui.session_state.codex_efforts == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    ]


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
    assert cli.LOCK_PATH.name == f"tagalong-{os.getuid()}.lock"


@pytest.mark.usefixtures("wiring")
def test_asking_for_help_does_not_wait_on_the_single_instance_lock(monkeypatch) -> None:
    """``--help`` describes the program; it starts no session to conflict with.

    Taking the lock before parsing made every argument-only invocation fail
    while a session was running, which is exactly when someone reads the help.
    """

    def already_running(*_args, **_kwargs):
        raise RuntimeError("Another tagalong session is already running.")

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
        self.available = frozenset({"Voice"})

    def set_available(self, available):
        self.available = frozenset(available)


def audio_channel(open_channel=None, close_channel=None):
    """A far-end channel over fakes, with its worker thread left unstarted."""
    tui = FakeTUI(SimpleNamespace(), on_audio_mute=None)
    opened: list[object] = []
    closed: list[object] = []

    def open_default(tap):
        transcriber = FakeTranscriber(tap=tap)
        listener = SimpleNamespace(speaker="Audio", set_muted=lambda muted: None)
        opened.append((transcriber, listener))
        return transcriber, listener

    channel = cli.AudioChannel(
        tui,
        FakeGate(),
        open_channel or open_default,
        close_channel or (lambda t, listener: closed.append((t, listener))),
    )
    return channel, tui, opened, closed


def test_choosing_an_application_makes_them_answerable() -> None:
    """A policy naming Audio cannot fire until a far end actually exists."""
    channel, _, _, _ = audio_channel()

    channel.select("Brave")
    channel.reconcile()

    assert channel.gate.available == frozenset({"Voice", "Audio"})


def test_dropping_the_application_makes_them_unanswerable_again() -> None:
    channel, _, _, _ = audio_channel()
    channel.select("Brave")
    channel.reconcile()

    channel.select(None)
    channel.reconcile()

    assert channel.gate.available == frozenset({"Voice"})


def test_the_newest_choice_wins_rather_than_each_one_building() -> None:
    """Moving through the picker produces a choice per keystroke."""
    channel, _, opened, _ = audio_channel()

    channel.select("Brave")
    channel.select("Chromium")
    channel.select("ZOOM VoiceEngine")
    channel.reconcile()

    assert len(opened) == 1
    assert channel.tap.application == "ZOOM VoiceEngine"


def test_asking_again_for_what_is_already_open_changes_nothing() -> None:
    channel, _, opened, closed = audio_channel()
    channel.select("Brave")
    channel.reconcile()

    channel.select("Brave")
    channel.reconcile()

    assert (len(opened), len(closed)) == (1, 0)


def test_a_far_end_that_cannot_be_opened_leaves_the_session_running() -> None:
    """The state a failure lands in is the one the session was already in."""

    def refuse(_tap):
        raise RuntimeError("no pw-record")

    channel, tui, _, _ = audio_channel(open_channel=refuse)

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

    channel, _, _, _ = audio_channel(open_channel=refuse)
    channel.select("Brave")
    channel.reconcile()

    channel.reconcile()

    assert attempts == ["Brave"]


def test_the_worker_serves_choices_until_the_session_closes() -> None:
    channel, _tui, opened, closed = audio_channel()
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
    channel, _, opened, closed = audio_channel()

    channel.close()

    assert (opened, closed) == ([], [])


def test_starting_the_channel_twice_serves_it_once() -> None:
    channel, _, _, _ = audio_channel()
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
    assert cli.audio_stream_setting(application) == recorded
