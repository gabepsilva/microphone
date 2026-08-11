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
from tagalong.control.transcript import TranscriptStore
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
        self.microphones: list[tuple[str, str]] = []
        self.microphone_reports: list[list[tuple[str, str]]] = []
        self.closed_speakers: list[str] = []
        self.app = SimpleNamespace()
        self._call: object | None = None
        self.transcript = TranscriptStore()
        self.partial_publisher = None
        self.session_state_publisher = None

    def transcript_entries(self):
        return list(self.transcript.transcript_entries())

    def bind_partial_publisher(self, publish):
        self.partial_publisher = publish

    def bind_session_state_publisher(self, publish):
        self.session_state_publisher = publish

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
        self.microphone_reports.append(list(microphones))

    def finish_recording(self):
        self.finished_recording = True

    def reset_transcript(self):
        self.resets = getattr(self, "resets", 0) + 1


class FakeTranscriber:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.listeners = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.muted = False
        self.devices: list[int] = []
        self._device = kwargs.get("device")
        self._sd_stream = None

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

    def switch_device(self, device):
        self.devices.append(device)
        self._device = device


class FakeConversation:
    def __init__(self, settings, display, tts):
        self.settings = settings
        self.display = display
        self.tts = tts
        self.thread = SimpleNamespace(id="thread-1")
        self.ingested = []
        self.latency = TurnLatencyEstimator()
        self.prefired = []
        self.generation = 0
        self.interrupts = 0
        self.sessions = 0
        self.new_session_ok = True

    def ingest(self, speaker, text, respond, timestamp=None, images=()):
        del timestamp
        self.ingested.append((speaker, text, respond, images))

    def prefire(self, speaker, text):
        self.prefired.append((speaker, text))
        return True

    def commit_prefire(self, _speaker):
        return True

    def cancel_prefire(self, _speaker):
        return True

    def interrupt(self):
        self.interrupts += 1

    def start_fresh_thread(self):
        if not self.new_session_ok:
            return None
        return SimpleNamespace(id=f"thread-{self.generation + 1}")

    def adopt_fresh_thread(self, started):
        self.generation += 1
        self.sessions += 1
        self.thread = started

    def new_session(self):
        started = self.start_fresh_thread()
        if started is None:
            return False
        self.adopt_fresh_thread(started)
        return True

    def request_model(self, _model):
        return True

    def request_reasoning_effort(self, _effort):
        return True


class FakeTTS:
    def __init__(self, voice, output_sink=None):
        self.voice = voice
        self.output_sink = output_sink
        self.enabled = True
        self.provider = "piper"
        # Set by the tests that need a switch still in flight, which is one of
        # the two things the real engine refuses a provider change for.
        self.switching = False

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_provider(self, provider, voice=None, *, on_applied=None, on_failed=None):
        del on_failed
        if self.switching or provider == self.provider:
            return False
        self.provider = provider
        if voice is not None:
            self.voice = voice
        # Defer the settle callback so Effect.pending is registered first.
        if on_applied is not None:
            applied = self.voice
            threading.Thread(
                target=on_applied, args=(applied,), name="FakeTTSApplied", daemon=True
            ).start()
        return True

    def set_voice(self, voice, *, on_applied=None, on_failed=None):
        del on_failed
        if self.switching or voice == self.voice:
            return False
        self.voice = voice
        if on_applied is not None:
            threading.Thread(
                target=on_applied,
                args=(voice,),
                name="FakeTTSVoiceApplied",
                daemon=True,
            ).start()
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
    monkeypatch.setattr("tagalong.streams._DEFAULT_PLATFORM", "linux")
    config = tmp_path / "tagalong.yaml"
    config.write_text("", encoding="utf-8")
    built: dict[str, object] = {}

    def resolve(_args):
        device = built["input_device"]
        return SimpleNamespace(
            device_index=3 if device is not None else None,
            device=device,
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
                ("low", "medium", "high", "xhigh", "max"),
                "medium",
            ),
            CodexModelOption(
                "gpt-5.6-sol",
                "GPT-5.6 Sol",
                ("low", "medium", "high", "xhigh", "max", "ultra"),
                "low",
            ),
        ],
        raising=False,
    )
    monkeypatch.setattr(cli, "resolve_startup_selection", resolve)
    monkeypatch.setattr(cli, "run_session", run_session)
    monkeypatch.setattr(cli, "print_startup_summary", lambda *a, **k: None)

    original_attach = cli.attach_conversation_hooks

    def tracking_attach(*args, **kwargs):
        controller, actor = original_attach(*args, **kwargs)
        built["controller"] = controller
        built["actor"] = actor
        return controller, actor

    monkeypatch.setattr(cli, "attach_conversation_hooks", tracking_attach)

    monkeypatch.setattr(
        cli,
        "build_speech",
        lambda selection, args: FakeTTS(
            args.tts_voice,
            (
                selection.tts_output["name"]
                if selection.tts_output is not None
                else None
            ),
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
    # A live session on this machine already owns the XDG socket. Main-path
    # tests still exercise the event pump; they must not bind that socket.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    built.update(
        input_device=MIC,
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


def test_switching_microphones_retargets_the_live_capture_without_rebuilding() -> None:
    """A live switch must not tear down Moonshine or leave PortAudio dangling.

    Rebuilding the channel used to stop the recognizer while the PortAudio
    stream kept calling into it; a few seconds later the process crashed.
    Retargeting reopens only the input stream on the existing transcriber.
    """
    microphone, _, opened = managed_microphone(INPUTS)
    microphone.select("Yeti")
    microphone.reconcile()
    _, first_transcriber, first_listener = opened[0]

    microphone.select("Webcam")
    microphone.reconcile()

    assert [index for index, _, _ in opened] == [3]
    assert first_transcriber.devices == [7]
    assert (first_transcriber.stopped, first_transcriber.closed) == (False, False)
    assert first_listener.closed is False
    assert microphone.current == "Webcam"
    assert microphone.transcriber is first_transcriber


def test_a_failed_microphone_switch_keeps_the_current_capture() -> None:
    microphone, tui, opened = managed_microphone(INPUTS)
    microphone.select("Yeti")
    microphone.reconcile()
    _, first_transcriber, _ = opened[0]

    def reject(_device):
        raise RuntimeError("device busy")

    first_transcriber.switch_device = reject

    microphone.select("Webcam")
    microphone.reconcile()
    microphone.reconcile()

    assert microphone.current == "Yeti"
    assert microphone.transcriber is first_transcriber
    assert first_transcriber.stopped is False
    assert tui.notes == ["could not listen to Webcam: device busy"]


def test_retargeting_without_a_live_channel_is_a_no_op() -> None:
    """_retarget guards a concurrent retire; reconcile never calls it this way."""
    microphone, tui, opened = managed_microphone(INPUTS)

    microphone._retarget(3, {"name": "Yeti"})

    assert opened == []
    assert microphone.current is None
    assert tui.notes == []


def test_selecting_no_microphone_closes_capture() -> None:
    microphone, _tui, opened = managed_microphone(INPUTS)
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
    assert microphone.transcriber is None
    microphone.set_muted(True)


def test_a_microphone_open_failure_leaves_the_session_running() -> None:
    microphone, tui, opened = managed_microphone(
        INPUTS, open_error=RuntimeError("device busy")
    )

    microphone.select("Yeti")
    microphone.reconcile()

    assert opened == []
    assert microphone.current is None
    assert microphone.desired == "Yeti"
    assert tui.notes == ["could not listen to Yeti: device busy"]


def test_a_microphone_that_failed_to_open_is_retried_by_the_next_pass() -> None:
    """The failed request is kept on purpose, because holding it is the retry.

    ``MicrophoneChannel`` polls, so a device that is busy now is opened when
    it frees up and the desired selection has to survive to be tried again.
    ``AudioChannel`` abandons instead — nothing there wakes up to revisit a
    request it kept — which is why the two answer ``desired`` differently
    after a failure rather than one of them being wrong.
    """
    attempts: list[int] = []
    applied: list[str | None] = []

    def open_channel(index):
        attempts.append(index)
        if len(attempts) == 1:
            raise RuntimeError("device busy")
        return FakeTranscriber(device=index), SwitchListener()

    microphone = cli.MicrophoneChannel(
        FakeTUI(SessionState()),
        open_channel,
        lambda transcriber, listener: None,
        devices=INPUTS,
        discover=lambda: list(INPUTS),
    )
    microphone.select("Yeti", on_applied=applied.append)

    microphone.reconcile()

    assert (microphone.current, microphone.desired, applied) == (None, "Yeti", [])

    microphone.reconcile()

    assert (microphone.current, applied) == ("Yeti", ["Yeti"])


def test_a_failed_microphone_selection_is_not_remembered(wiring, tmp_path) -> None:
    """Acceptance is not evidence that an asynchronous switch took effect."""
    wiring["input_device"] = None
    cli.main()
    tui, _, _ = wiring["session"]
    microphone = wiring["microphone"]
    microphone.devices = microphone._by_name(INPUTS)
    before = (tmp_path / "tagalong.yaml").read_bytes()

    microphone.open_channel = lambda _index: (_ for _ in ()).throw(
        RuntimeError("device busy")
    )
    tui.hooks.on_microphone("Yeti")
    microphone.reconcile()

    assert microphone.current is None
    assert (tmp_path / "tagalong.yaml").read_bytes() == before


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

    class Stream:
        def stop(self):
            events.append("stop stream")

        def close(self):
            events.append("close stream")

    class Transcriber:
        def __init__(self):
            self._sd_stream = Stream()

        def stop(self):
            events.append("stop capture")

        def close(self):
            events.append("close capture")

    class Listener:
        def close(self):
            events.append("close listener")

    listener = Listener()
    transcriber = Transcriber()
    activity = SimpleNamespace(on_transition=lambda active: None)
    parts = SimpleNamespace(
        submitter=SimpleNamespace(
            remove_listener=lambda removed: events.append(
                "unregister listener" if removed is listener else "wrong listener"
            )
        )
    )

    cli.close_microphone_channel(parts, activity, transcriber, listener)

    assert activity.on_transition is None
    assert events == [
        "stop capture",
        "stop stream",
        "close stream",
        "close listener",
        "close capture",
        "unregister listener",
    ]
    assert transcriber._sd_stream is None


def test_energy_transition_binding_forwards_loud_and_quiet() -> None:
    events: list[str] = []

    class Listener:
        def on_energy_loud(self):
            events.append("loud")

        def on_energy_quiet(self):
            events.append("quiet")

    activity = SimpleNamespace(on_transition=None)
    cli.bind_energy_transitions(activity, Listener())
    activity.on_transition(True)
    activity.on_transition(False)
    cli.clear_energy_transitions(activity)

    assert events == ["loud", "quiet"]
    assert activity.on_transition is None


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


def test_an_unchanged_microphone_list_is_not_reported_again() -> None:
    """Every report repaints the sidebar, and the list is usually the same."""
    microphone, tui, _ = managed_microphone(INPUTS)
    assert microphone.refresh() is True

    assert microphone.refresh() is False
    assert tui.microphone_reports == [
        [("Yeti", "Yeti"), ("Webcam", "Webcam")],
    ]


def test_a_microphone_list_change_is_reported() -> None:
    available: list[tuple[int, dict]] = list(INPUTS[:1])
    microphone, tui, _ = managed_microphone(discover=lambda: list(available))
    microphone.refresh()

    available.extend(INPUTS[1:])
    assert microphone.refresh() is True
    assert tui.microphone_reports == [
        [("Yeti", "Yeti")],
        [("Yeti", "Yeti"), ("Webcam", "Webcam")],
    ]


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


def test_failed_application_selection_is_not_remembered(
    wiring, tmp_path, monkeypatch
) -> None:
    """A failed slow effect must not become the next session's startup state."""
    cli.main()
    tui, _, _ = wiring["session"]
    them = wiring["audio"]
    before = (tmp_path / "tagalong.yaml").read_bytes()
    monkeypatch.setattr(
        cli,
        "ApplicationStreamTranscriber",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no pw-record")),
    )

    tui.hooks.on_audio_stream(THEM_APPLICATION)
    them.reconcile()

    assert them.current is None
    assert (tmp_path / "tagalong.yaml").read_bytes() == before


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
    them.set_muted(True)


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


def test_an_application_opened_while_muted_starts_muted(wiring) -> None:
    """Desired mute state must survive the absence of an audio channel."""
    cli.main()
    tui, _, _ = wiring["session"]
    them = wiring["audio"]
    tui.state.audio.muted = True

    tui.hooks.on_audio_stream(THEM_APPLICATION)
    them.reconcile()

    assert them.transcriber.muted is True
    assert them.listener.muted is True


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


def test_a_session_without_a_speaker_channel_still_records_desired_mute(
    wiring,
) -> None:
    cli.main()
    tui, _, _ = wiring["session"]
    controller = wiring["controller"]

    assert tui.hooks.on_audio_mute is not None
    assert tui.hooks.on_audio_mute(True) is True
    assert controller.state.audio_stream_muted is True
    assert wiring["audio"].transcriber is None


def test_every_interface_hook_is_bound_before_the_session_runs(wiring) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    assert {
        "on_user_text",
        "on_interrupt",
        "on_command",
        "on_entry",
        "on_policy",
        "on_codex_model",
        "on_codex_effort",
        "on_tts",
        "on_tts_provider",
        "on_turn_silence",
        "on_mute",
        "on_audio_mute",
        "on_microphone",
        "on_audio_stream",
        "on_end_turn",
        "on_save",
        "on_attachment_upload",
        "on_quit",
    } <= set(vars(tui.hooks))
    assert tui.hooks.on_mute is not None
    assert tui.session_state_publisher is not None
    assert tui.hooks.on_audio_mute is not None
    assert tui.hooks.on_entry is not None
    assert tui.hooks.on_command is not None
    assert tui.hooks.on_end_turn is not None
    assert tui.hooks.on_save is not None
    assert tui.hooks.on_attachment_upload is not None
    assert tui.hooks.on_quit is not None
    # Non-headless quit_cleanup stops capture channels only (Textual exits itself).
    assert tui.hooks.on_quit() is True


def test_main_finishes_transcript_recording_after_the_session(
    wiring, tmp_path, monkeypatch
) -> None:
    """The recorder is closed after run_session so a clean quit still flushes."""
    closed: list[object] = []

    class TrackingRecorder(cli.TranscriptRecorder):
        def close(self):
            closed.append(self)
            super().close()

    monkeypatch.setattr(
        cli,
        "TranscriptRecorder",
        lambda: TrackingRecorder(directory=tmp_path),
    )
    cli.main()
    tui, _, _ = wiring["session"]

    assert closed
    assert getattr(tui, "finished_recording", False) is True


def test_snapshot_describes_the_live_session_after_startup(wiring) -> None:
    cli.main()
    tui, _, _ = wiring["session"]
    controller = wiring["controller"]
    snapshot = controller.snapshot().state
    microphone = wiring["microphone"]

    assert snapshot.tts_enabled is tui.state.tts_enabled
    assert snapshot.tts_provider.desired == tui.state.tts_provider
    assert snapshot.tts_provider.effective == tui.state.tts_provider
    assert snapshot.response_policy == tui.state.policy
    assert snapshot.codex_model == tui.state.codex_model
    assert snapshot.codex_reasoning == tui.state.codex_effort
    assert snapshot.turn_silence == tui.state.turn_silence
    assert snapshot.microphone.desired == tui.state.microphone
    assert snapshot.microphone.effective == microphone.current
    assert snapshot.audio_stream.desired == tui.state.audio_stream
    assert snapshot.microphone_muted is tui.state.mic.muted
    assert snapshot.audio_stream_muted is tui.state.audio.muted


def test_typed_text_always_requests_a_reply(wiring) -> None:
    cli.main()
    tui, _, conversation = wiring["session"]

    tui.hooks.on_user_text(UserTextMessage("what time is it?"))

    assert conversation.ingested == [("Text", "what time is it?", True, ())]


def test_attach_remote_access_applies_remote_tts_to_the_tui(
    tmp_path, monkeypatch
) -> None:
    from tagalong.application import bind_first_slice
    from tagalong.control import Controller, local_user

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    tui = FakeTUI(SessionState(tts_enabled=True))
    controller = Controller()

    class Speech:
        def set_enabled(self, enabled: bool) -> bool:
            del enabled
            return True

    class Conversation:
        generation = 1

        def ingest(self, *_args, **_kwargs):
            return None

        def interrupt(self) -> None:
            return None

        def start_fresh_thread(self):
            return None

        def adopt_fresh_thread(self, started) -> None:
            del started

    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    pump, server = cli.attach_remote_access(controller, tui)
    try:
        controller.dispatch(
            "tts.set_enabled", {"enabled": False}, actor=local_user("tui")
        )
        deadline = time.time() + 1.0
        while tui.state.tts_enabled is not False and time.time() < deadline:
            time.sleep(0.05)
        assert tui.state.tts_enabled is False
        assert server is not None
        assert server.path.exists()
    finally:
        pump.stop()
        if server is not None:
            server.stop()


def test_attach_remote_access_refreshes_the_sidebar_on_the_ui_thread(
    tmp_path, monkeypatch
) -> None:
    from tagalong.application import bind_first_slice
    from tagalong.control import Controller, local_user

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    tui = FakeTUI(SessionState(tts_enabled=True))
    tui.app = SimpleNamespace(refresh_sidebar=lambda: tui.notes.append("sidebar"))
    tui._call = lambda callback: callback()
    controller = Controller()

    class Speech:
        def set_enabled(self, enabled: bool) -> bool:
            del enabled
            return True

    class Conversation:
        generation = 1

        def ingest(self, *_args, **_kwargs):
            return None

        def interrupt(self) -> None:
            return None

        def start_fresh_thread(self):
            return None

        def adopt_fresh_thread(self, started) -> None:
            del started

    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    pump, server = cli.attach_remote_access(controller, tui)
    try:
        controller.dispatch(
            "tts.set_enabled", {"enabled": False}, actor=local_user("tui")
        )
        deadline = time.time() + 1.0
        while "sidebar" not in tui.notes and time.time() < deadline:
            time.sleep(0.05)
        assert "sidebar" in tui.notes
        assert tui.state.tts_enabled is False
    finally:
        pump.stop()
        if server is not None:
            server.stop()


def test_attach_remote_access_leaves_the_tui_running_without_xdg(
    monkeypatch,
) -> None:
    from tagalong.control import Controller

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    tui = FakeTUI(SessionState(tts_enabled=True))
    pump, server = cli.attach_remote_access(Controller(), tui)
    try:
        assert server is None
    finally:
        pump.stop()


def test_attach_remote_access_survives_a_socket_bind_failure(
    tmp_path, monkeypatch
) -> None:
    from tagalong.control import Controller

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    class BrokenServer:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def start(self) -> None:
            raise OSError("address already in use")

    monkeypatch.setattr(cli, "LocalServer", BrokenServer)
    pump, server = cli.attach_remote_access(Controller(), FakeTUI(SessionState()))
    try:
        assert server is None
    finally:
        pump.stop()


def test_main_headless_builds_headless_host_not_tui(wiring, monkeypatch) -> None:
    from tagalong.headless import HeadlessSession

    built = wiring
    hosts: list[object] = []
    original = cli.build_session_host

    def tracking(*args, **kwargs):
        host = original(*args, **kwargs)
        hosts.append(host)
        return host

    config = str(cli.sys.argv[cli.sys.argv.index("--config") + 1])
    monkeypatch.setattr(cli.sys, "argv", ["tagalong", "--config", config, "--headless"])
    monkeypatch.setattr(cli, "build_session_host", tracking)
    monkeypatch.setattr(HeadlessSession, "run", lambda self: None)

    cli.main()

    assert len(hosts) == 1
    assert isinstance(hosts[0], HeadlessSession)
    assert "controller" in built
    assert built["session"][0] is hosts[0]
    # quit_cleanup must unblock HeadlessSession.run (Textual exits itself).
    host = hosts[0]
    assert host.hooks.on_quit is not None
    assert host.hooks.on_quit() is True
    assert host._stop.is_set()


def test_main_headless_still_acquires_the_instance_lock(wiring, monkeypatch) -> None:
    from tagalong.headless import HeadlessSession

    assert wiring is not None
    acquired: list[bool] = []

    def lock(*_a, **_k):
        acquired.append(True)

    config = str(cli.sys.argv[cli.sys.argv.index("--config") + 1])
    monkeypatch.setattr(cli.sys, "argv", ["tagalong", "--config", config, "--headless"])
    monkeypatch.setattr(cli, "acquire_single_instance_lock", lock)
    monkeypatch.setattr(HeadlessSession, "run", lambda self: None)

    cli.main()
    assert acquired == [True]


def test_run_attached_session_stops_the_socket_when_the_tui_exits(
    tmp_path, monkeypatch
) -> None:
    from tagalong.application import bind_first_slice
    from tagalong.control import Controller
    from tagalong.transport import LocalServer

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    controller = Controller()

    class Speech:
        def set_enabled(self, enabled: bool) -> bool:
            del enabled
            return True

    class Conversation:
        generation = 1

        def ingest(self, *_args, **_kwargs):
            return None

        def interrupt(self) -> None:
            return None

        def start_fresh_thread(self):
            return None

        def adopt_fresh_thread(self, started) -> None:
            del started

    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    seen: dict[str, object] = {}

    def boom(*_args, **_kwargs):
        raise RuntimeError("session ended")

    monkeypatch.setattr(cli, "run_session", boom)
    original = cli.attach_remote_access

    def tracking(controller, tui):
        pump, server = original(controller, tui)
        seen["server"] = server
        return pump, server

    monkeypatch.setattr(cli, "attach_remote_access", tracking)
    with pytest.raises(RuntimeError, match="session ended"):
        cli.run_attached_session(
            controller, FakeTUI(SessionState()), Conversation(), None, None
        )
    server = seen["server"]
    assert isinstance(server, LocalServer)
    assert not server.path.exists()
    # Session teardown stops the coalesce pump (Controller.close).
    assert (
        controller.transcript._pump_thread is None
        or not controller.transcript._pump_thread.is_alive()
    )


def test_the_first_slice_is_bound_to_the_session_controller(wiring) -> None:
    cli.main()
    tui, _, conversation = wiring["session"]
    controller = wiring["controller"]

    assert controller.state.tts_enabled is True
    assert wiring["actor"].id == "tui"

    tui.hooks.on_user_text(UserTextMessage("hello"))
    tui.hooks.on_interrupt()
    # session.new settles on a daemon worker. Wait for the terminal event, not
    # for conversation.sessions: adopt_fresh_thread bumps that counter before
    # display.reset_transcript runs (application.py settle order), so this loop
    # used to exit with the reset still pending. announce is in the finally
    # after roll, so action.applied is the only "settle is done" signal.
    terminal = {"action.applied", "action.failed", "action.superseded"}
    _, subscription = controller.subscribe()
    try:
        tui.hooks.on_command("/new")
        deadline = time.monotonic() + 2.0
        settled = False
        while time.monotonic() < deadline and not settled:
            settled = any(event.name in terminal for event in subscription.drain())
            if not settled:
                subscription.wait(0.05)
        assert settled, "session.new never reached a terminal outcome"
    finally:
        subscription.close()

    assert conversation.ingested == [("Text", "hello", True, ())]
    assert conversation.interrupts == 1
    assert conversation.sessions == 1
    assert getattr(tui, "resets", 0) == 1
    assert controller.state.tts_enabled is True


def test_the_voice_toggle_mutes_the_session_engine(wiring) -> None:
    """Off is the engine told to stop generating, not the engine taken away."""
    cli.main()
    tui, _, conversation = wiring["session"]

    tui.hooks.on_tts(False)
    assert conversation.tts.enabled is False
    assert wiring["controller"].state.tts_enabled is False

    tui.hooks.on_tts(True)
    assert conversation.tts.enabled is True
    assert wiring["controller"].state.tts_enabled is True


def test_speech_is_routed_to_the_chosen_playback_sink(wiring) -> None:
    wiring["tts_output"] = PLAYBACK

    cli.main()
    _, _, conversation = wiring["session"]

    assert conversation.tts.output_sink == "alsa_output.pci"
    assert conversation.tts.voice == "en_US-lessac-medium"


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


def test_speech_is_built_for_the_chosen_provider_and_output(monkeypatch) -> None:
    started: list[tuple] = []
    monkeypatch.setattr(
        cli.SwitchableSpeech,
        "start",
        classmethod(
            lambda _cls, provider, voice, output_sink=None: (
                started.append((provider, voice, output_sink)) or object()
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
            lambda _cls, provider, voice, output_sink=None: (
                started.append(output_sink) or object()
            )
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
    cli.main()
    tui, _, conversation = wiring["session"]

    assert tui.speech is conversation.tts


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


def test_startup_selections_do_not_rewrite_the_config(wiring, tmp_path) -> None:
    """Only client actions persist; bootstrap merely applies resolved inputs."""
    path = tmp_path / "tagalong.yaml"
    before = path.read_bytes()

    cli.main()

    assert wiring["session"]
    assert path.read_bytes() == before


def test_a_controller_settings_change_is_remembered(wiring, tmp_path) -> None:
    """Socket and sidebar share one persistence path: the action handler."""
    cli.main()
    controller = wiring["controller"]
    actor = wiring["actor"]

    controller.dispatch("codex.set_model", {"model": "gpt-5.6-sol"}, actor=actor)
    controller.dispatch("response_policy.set", {"policy": "voice"}, actor=actor)
    controller.dispatch("turn_silence.set", {"seconds": 0.01}, actor=actor)
    controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=actor)
    controller.dispatch("codex.set_reasoning", {"effort": "high"}, actor=actor)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        saved = saved_config(tmp_path)
        if saved.get("tts_provider") == "edge":
            break
        time.sleep(0.01)
    saved = saved_config(tmp_path)
    assert saved["codex_model"] == "gpt-5.6-sol"
    assert saved["taga_after"] == "voice"
    assert saved["turn_silence"] == 0.25
    assert saved["tts_provider"] == "edge"
    assert saved["codex_reasoning"] == "high"


def test_every_sidebar_control_is_remembered(wiring, tmp_path) -> None:
    cli.main()
    tui, _, _ = wiring["session"]

    tui.hooks.on_policy("audio")
    tui.hooks.on_codex_model("gpt-5.6-sol")
    tui.hooks.on_codex_effort("high")
    tui.hooks.on_turn_silence(1.25)
    tui.hooks.on_tts_provider("edge")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        saved = saved_config(tmp_path)
        if saved.get("tts_provider") == "edge":
            break
        time.sleep(0.01)
    saved = saved_config(tmp_path)
    assert saved["taga_after"] == "audio"
    assert saved["codex_model"] == "gpt-5.6-sol"
    assert saved["codex_reasoning"] == "high"
    assert saved["turn_silence"] == 1.25
    assert saved["tts_provider"] == "edge"


def test_applied_microphone_and_application_selections_are_remembered(
    wiring, tmp_path
) -> None:
    cli.main()
    tui, _, _ = wiring["session"]
    microphone = wiring["microphone"]
    microphone.devices = microphone._by_name(INPUTS)
    them = wiring["audio"]

    tui.hooks.on_microphone("Webcam")
    microphone.reconcile()
    tui.hooks.on_audio_stream(THEM_APPLICATION)
    them.reconcile()

    saved = saved_config(tmp_path)
    assert saved["microphone"] == "Webcam"
    assert saved["audio_stream"] == THEM_APPLICATION


def test_muting_the_voice_is_not_remembered(wiring, tmp_path) -> None:
    """Silence is for the replies at hand, not for every session that follows."""
    cli.main()
    tui, _, conversation = wiring["session"]
    # Something else has to be saved first, or the file is never written at
    # all and "the mute was not recorded" would pass for the wrong reason.
    tui.hooks.on_policy("voice")

    tui.hooks.on_tts(False)

    assert conversation.tts.enabled is False
    assert "tts" not in saved_config(tmp_path)


def test_a_refused_change_is_not_remembered(wiring, tmp_path) -> None:
    """A refused change moved nothing, so the file must not move either."""
    cli.main()
    tui, _, conversation = wiring["session"]

    assert tui.hooks.on_tts_provider("edge") is True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if saved_config(tmp_path).get("tts_provider") == "edge":
            break
        time.sleep(0.01)
    assert saved_config(tmp_path)["tts_provider"] == "edge"
    # The engine is still building the engine it just accepted, so the next
    # request is refused rather than queued behind it.
    conversation.tts.switching = True

    assert tui.hooks.on_tts_provider("piper") is False
    assert saved_config(tmp_path)["tts_provider"] == "edge"


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
    tui = FakeTUI(SessionState(), on_audio_mute=None)
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


def test_selection_completion_reports_the_effective_value() -> None:
    applied: list[str | None] = []
    selections = cli._SelectionRequests()
    request = selections.replace("requested", applied.append)

    assert selections.complete(request, "effective") is True
    assert applied == ["effective"]


def test_selection_completion_runs_after_the_channel_lock_is_released() -> None:
    channel, _, _, _ = audio_channel()
    observations: list[tuple[str | None, bool]] = []

    def observe(effective):
        acquired = channel.lock.acquire(blocking=False)
        observations.append((effective, acquired))
        if acquired:
            channel.lock.release()

    channel.select("Brave", on_applied=observe)

    channel.reconcile()

    assert observations == [("Brave", True)]


def test_an_aba_selection_only_completes_the_newest_request() -> None:
    applied: list[str] = []
    channel: cli.AudioChannel

    def open_first(tap):
        channel.select("Chromium", on_applied=lambda _value: applied.append("middle"))
        channel.select("Brave", on_applied=lambda _value: applied.append("newest"))
        return FakeTranscriber(tap=tap), SwitchListener()

    channel, _, _, _ = audio_channel(open_channel=open_first)
    channel.select("Brave", on_applied=lambda _value: applied.append("oldest"))

    channel.reconcile()
    assert applied == []
    channel.reconcile()

    assert applied == ["newest"]


def test_a_failed_open_does_not_discard_a_newer_request() -> None:
    channel: cli.AudioChannel

    def refuse(_tap):
        channel.select("Chromium")
        raise RuntimeError("no pw-record")

    channel, _, _, _ = audio_channel(open_channel=refuse)
    channel.select("Brave")

    channel.reconcile()

    assert channel.desired == "Chromium"


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
