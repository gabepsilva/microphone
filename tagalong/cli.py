#!/usr/bin/env python3
"""Wire the always-listening Voice/Text/Audio/Taga conversation together.

This module is the composition root and nothing else. The runtime parts it
assembles are independent of one another — none of them imports any of the
others:

  * ``capture``  — microphone and application-stream audio into Moonshine,
                   the second by way of the PipeWire tap in ``streams``
  * ``listener`` — transcription events into completed turns
  * ``codex``    — the Codex thread and its streamed turns
  * ``speech``   — which synthesizer speaks Taga's responses, and the two
                   engines behind it in ``tts`` and ``piper_tts``

``startup`` resolves what the session will be before any of that is built,
drawing on ``choosers`` for the interactive questions and ``catalog`` for the
model list. It is the only module besides this one that imports more than its
own concern, and it holds nothing the running session needs.

Keep it that way: an import between two of the four above is the first sign
that a boundary is in the wrong place.
"""

from __future__ import annotations

import atexit
import fcntl
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from moonshine_voice import get_model_for_language
from moonshine_voice.moonshine_api import ModelArch

from .capture import (
    ApplicationStreamTranscriber,
    CaptureSettings,
    SoundActivityReporter,
    metered_mic_transcriber,
    release_microphone_input,
)
from .catalog import probe_codex_models
from .choosers import NO_AUDIO_STREAM, input_devices
from .codex import CodexConversation, CodexSettings
from .commands import CommandRouter
from .config import StartupConfigFile, save_startup_config
from .domain import (
    PrefirePlan,
    SpeakerGate,
    SpeakerPresence,
    TurnSilence,
    TurnSilenceClock,
)
from .listener import TranscriptSubmitter
from .recording import TranscriptRecorder
from .session import sweep_orphans
from .speech import SwitchableSpeech
from .startup import (
    build_session_state,
    parse_startup_args,
    print_startup_summary,
    resolve_startup_selection,
    run_session,
    startup_settings,
    validate_codex_reasoning,
)
from .streams import ApplicationRefresher, StreamTap

# The name carries the user ID because the fallback is a shared directory: on a
# multi-user box a fixed name in /tmp is one user's lock file blocking every
# other user's session, or refusing to open at all because they own it.
LOCK_PATH = (
    Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
    / f"tagalong-{os.getuid()}.lock"
)
_INSTANCE_LOCK = None
_LOCK_RELEASE_REGISTERED = False


def _release_single_instance_lock():
    """Release the process lock, if held."""
    global _INSTANCE_LOCK
    if _INSTANCE_LOCK is None:
        return
    _INSTANCE_LOCK.close()
    _INSTANCE_LOCK = None


def acquire_single_instance_lock(lock_path: Path = LOCK_PATH):
    """Hold an exclusive process lock for the life of this process.

    The lock is advisory and tied to the open file descriptor, so a forced
    process exit still releases it when the kernel closes descriptors.
    """
    global _INSTANCE_LOCK, _LOCK_RELEASE_REGISTERED
    if _INSTANCE_LOCK is not None:
        return
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError("Another tagalong session is already running.") from error
    _INSTANCE_LOCK = lock_file
    if not _LOCK_RELEASE_REGISTERED:
        atexit.register(_release_single_instance_lock)
        _LOCK_RELEASE_REGISTERED = True


def build_speech(selection, args):
    """Build the session's speech.

    Every session has a synthesizer. "No voice reply" only stops audio being
    generated, so there is nothing for a session to start without: the engine
    is what the sidebar mutes, switches, and unmutes.
    """
    return SwitchableSpeech.start(
        selection.tts_provider,
        args.tts_voice,
        output_sink=(
            selection.tts_output["name"] if selection.tts_output is not None else None
        ),
    )


def sound_taps(tui):
    """The two level taps, built before the listeners that read them.

    Each channel's listener reads the other's tap: the microphone has to know
    when the far end is playing to tell its voice coming out of the speakers
    from the person sitting in the room. Both are built whether or not a far
    end exists yet, because one can be chosen at any point in the session and
    a reporter costs a few numbers until it is.
    """
    return SoundActivityReporter(tui, "mic"), SoundActivityReporter(tui, "audio")


BASE_SPEAKERS = frozenset({"Voice"})


def audio_stream_setting(application):
    """Record a chosen application the way ``--audio-stream`` reads it back."""
    return NO_AUDIO_STREAM if application is None else application


def speaker_gate(selection):
    """Build the response policy, limited to the speakers this session has.

    It starts with the microphone alone. Selecting "both" before a far end
    exists must not make Audio replies possible, and the gate is told when one
    arrives rather than inferring it.
    """
    return SpeakerGate(selection.policy.speakers, BASE_SPEAKERS)


@dataclass(frozen=True)
class ChannelParts:
    """What a capture channel is built from, gathered once.

    The far end's channel is built long after the composition root has
    returned — when someone picks an application — so what it needs has to
    outlive the wiring rather than be closed over inside it.
    """

    submitter: object
    display: object
    confidence: float
    turn_silence: object
    countdown: object
    model_path: str
    model_arch: object


def muting(transcriber, listener):
    """Build the mute hook for one channel, from the capture end backwards.

    Both ends are needed. The transcriber stops the work — a muted channel that
    only discards transcripts has already paid for them. The listener stops
    everything that was arranged around the words: its buffer, its silence
    timers, and any turn it has speculatively started.

    Capture is gated first so that no audio recorded after the click can reach
    a listener that has already cleared itself.
    """

    def set_muted(muted):
        transcriber.set_muted(muted)
        listener.set_muted(muted)

    return set_muted


def open_audio_channel(parts, activity, tap):
    """Build the far end's listener and transcriber around one tap."""
    listener = parts.submitter.channel(
        parts.confidence,
        parts.turn_silence,
        "Audio",
        parts.display,
        countdown=parts.countdown,
        # No suppressors: the tap carries the links this session made and
        # nothing else, and Taga's own playback is never one of them. It is
        # not that its voice is filtered off this channel — it never reaches it.
        presence=SpeakerPresence(activity),
    )
    bind_energy_transitions(activity, listener)
    transcriber = ApplicationStreamTranscriber(
        model_path=parts.model_path,
        model_arch=parts.model_arch,
        tap=tap,
        # A smaller block only to sharpen the tap: this channel decides whether
        # the far end is talking a whole block at a time, and the 4096 default
        # put that a quarter of a second behind the microphone's. Transcription
        # is unaffected — it runs off ``update_interval``, not the read size.
        capture=CaptureSettings(update_interval=0.25, blocksize=1024),
        level_reporter=activity,
    )
    transcriber.add_listener(listener)
    return transcriber, listener


def open_microphone_channel(parts, activity, presence, device_index):
    """Build the user's listener and transcriber for one available device."""
    transcriber = metered_mic_transcriber(
        model_path=parts.model_path,
        model_arch=parts.model_arch,
        update_interval=0.25,
        device=device_index,
        samplerate=16000,
        channels=1,
        level_reporter=activity,
    )
    try:
        listener = parts.submitter.channel(
            parts.confidence,
            parts.turn_silence,
            "Voice",
            parts.display,
            countdown=parts.countdown,
            presence=presence,
        )
    except Exception:
        transcriber.close()
        raise
    bind_energy_transitions(activity, listener)
    transcriber.add_listener(listener)
    return transcriber, listener


def bind_energy_transitions(activity, listener):
    """Arm and cancel silence from level-tap transitions on ``listener``."""

    def on_transition(active: bool) -> None:
        if active:
            listener.on_energy_loud()
        else:
            listener.on_energy_quiet()

    activity.on_transition = on_transition


def clear_energy_transitions(activity):
    """Drop a channel's energy hook so a closed listener is never called."""
    activity.on_transition = None


def close_microphone_channel(parts, activity, transcriber, listener):
    """Retire one microphone without leaving its listener registered.

    Capture is stopped first, then the PortAudio stream is released before the
    model is closed. Leaving the stream alive is what turns a microphone
    switch into a delayed crash: the callback keeps firing into a freed
    Moonshine model until the process dies.
    """
    clear_energy_transitions(activity)
    transcriber.stop()
    release_microphone_input(transcriber)
    listener.close()
    transcriber.close()
    parts.submitter.remove_listener(listener)


def close_audio_channel(parts, activity, transcriber, listener):
    """Retire the far end's channel in the order the session's shutdown uses.

    Nothing may reach a listener after it has flushed, and the listener is
    unregistered last so no reply can sweep a closed channel's buffer in as
    context for a far end nobody is listening to any more.
    """
    clear_energy_transitions(activity)
    transcriber.stop()
    listener.close()
    transcriber.close()
    parts.submitter.remove_listener(listener)


@dataclass(frozen=True)
class _SelectionRequest:
    """One desired selection and the work to run if it becomes effective."""

    value: str | None
    on_applied: Callable[[str | None], object] | None = None


class _SelectionRequests:
    """Keep selection completion attached to the request that produced it.

    Device selection is intentionally asynchronous. A tiny lock protects only
    replacement and completion of the desired intent; slow device work never
    holds it, so a newer UI choice can supersede an in-flight request without
    waiting for a model or capture device to finish opening.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._request = _SelectionRequest(None)

    @property
    def desired(self):
        with self._lock:
            return self._request.value

    def replace(self, value, on_applied=None):
        request = _SelectionRequest(value, on_applied)
        with self._lock:
            self._request = request
        return request

    def snapshot(self):
        with self._lock:
            return self._request

    def complete(self, request, effective):
        """Notify only if *request* is still the newest desired selection."""
        with self._lock:
            if self._request is not request:
                return False
            self._request = _SelectionRequest(request.value)
        if request.on_applied is not None:
            request.on_applied(effective)
        return True

    def abandon(self, request):
        """Forget a failed request unless a newer selection superseded it."""
        with self._lock:
            if self._request is request:
                self._request = _SelectionRequest(None)


class MicrophoneChannel:
    """Keep a session alive while input devices appear, disappear, or change."""

    POLL_SECONDS = 2.0
    JOIN_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        tui,
        open_channel,
        close_channel,
        *,
        devices=(),
        discover=None,
    ):
        self.tui = tui
        self.open_channel = open_channel
        self.close_channel = close_channel
        self.discover = input_devices if discover is None else discover
        self.poll = self.POLL_SECONDS
        self.devices = self._by_name(devices)
        self._selection = _SelectionRequests()
        self.current = None
        self.transcriber = None
        self.listener = None
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.worker = None
        self._discovery_error = None
        self._open_error = None
        # Empty rather than unknown: the sidebar starts with no microphones,
        # so a first pass that finds none has nothing to report either.
        self.offered = []

    @staticmethod
    def _by_name(devices):
        return {device["name"]: (index, device) for index, device in devices}

    @staticmethod
    def _options(devices):
        return [(device["name"], device["name"]) for _, device in devices]

    def start(self):
        """Begin refreshing devices and serving choices."""
        if self.worker is not None:
            return
        self.worker = threading.Thread(
            target=self._serve, name="MicrophoneSwitch", daemon=True
        )
        self.worker.start()

    @property
    def desired(self):
        """Return the newest requested microphone name."""
        return self._selection.desired

    def select(self, microphone, *, on_applied=None):
        """Ask for a named microphone, or for none. Always accepted."""
        self._selection.replace(microphone, on_applied)
        self.wake.set()
        return True

    def refresh(self):
        """Publish currently available inputs, only when the list changed.

        Every report repaints the sidebar, and the list is the same on almost
        every pass, so an unconditional one would redraw the interface every
        few seconds for the whole session.
        """
        try:
            devices = self.discover()
        except RuntimeError as error:
            message = str(error)
            if message != self._discovery_error:
                self.tui.note(f"could not discover microphones: {message}")
                self._discovery_error = message
            devices = []
        else:
            self._discovery_error = None
        self.devices = self._by_name(devices)
        offered = self._options(devices)
        if offered == self.offered:
            return False
        self.offered = offered
        self.tui.set_microphones(offered)
        return True

    def _serve(self):
        while not self.stopping.is_set():
            self.refresh()
            self.reconcile()
            self.wake.wait(self.poll)
            self.wake.clear()

    def reconcile(self):
        """Make the open capture channel agree with the newest available choice.

        Moving between two live devices retargets the existing capture — only
        the PortAudio stream is reopened — so Moonshine stays loaded and the
        old callback is torn down cleanly. Building from scratch (or retiring
        entirely) is reserved for going to or from no microphone at all.
        """
        applied = False
        effective = None
        with self.lock:
            request = self._selection.snapshot()
            wanted = request.value
            if wanted == self.current:
                applied = True
            elif wanted is None:
                self._retire()
                applied = True
            else:
                selected = self.devices.get(wanted)
                if selected is not None:
                    index, device = selected
                    if self.transcriber is not None:
                        applied = self._retarget(index, device)
                    else:
                        applied = self._open(index, device)
            effective = self.current
        if applied:
            self._selection.complete(request, effective)

    def _retarget(self, device_index, device):
        """Point the live capture at another input without reloading the model."""
        transcriber = self.transcriber
        if transcriber is None:
            return False
        try:
            transcriber.switch_device(device_index)
        except Exception as error:
            message = f"could not listen to {device['name']}: {error}"
            if message != self._open_error:
                self.tui.note(message)
                self._open_error = message
            return False
        self._open_error = None
        self.current = device["name"]
        return True

    def _open(self, device_index, device):
        transcriber = None
        listener = None
        try:
            transcriber, listener = self.open_channel(device_index)
            transcriber.start()
        except Exception as error:
            if transcriber is not None and listener is not None:
                self.close_channel(transcriber, listener)
            message = f"could not listen to {device['name']}: {error}"
            if message != self._open_error:
                self.tui.note(message)
                self._open_error = message
            return False
        self._open_error = None
        self.transcriber, self.listener = transcriber, listener
        self.current = device["name"]
        self.tui.hooks.on_mute = muting(transcriber, listener)
        if self.tui.state.mic.muted:
            self.tui.hooks.on_mute(True)
        return True

    def _retire(self):
        transcriber, listener = self.transcriber, self.listener
        self.transcriber = self.listener = None
        self.current = None
        self.tui.hooks.on_mute = None
        if transcriber is not None and listener is not None:
            self.close_channel(transcriber, listener)

    def close(self):
        """Stop refreshing devices and close the active microphone, if any."""
        self.stopping.set()
        self.wake.set()
        if self.worker is not None:
            self.worker.join(timeout=self.JOIN_TIMEOUT_SECONDS)
            self.worker = None
        self.select(None)
        self.reconcile()


class AudioChannel:
    """Transcribe a far end's application, from the moment one is chosen.

    This costs a second Moonshine model — seconds to load, and it stays
    resident — so a session that never listens to a far end never builds one.
    That is also why the work runs on its own thread: the choice arrives as a
    keystroke in the sidebar, and the interface has to go on drawing while a
    model loads.

    Requests are recorded and reconciled rather than executed where they
    arrive. Someone moving through the picker produces a choice per keystroke,
    and each one would otherwise start building a channel that the next one
    made pointless. Here the newest choice simply wins, and a run of them
    costs one build.

    Switching between two applications is not a build at all. The tap follows
    a name, so the existing channel is pointed at the new one and its own
    relinking does the rest.
    """

    JOIN_TIMEOUT_SECONDS = 30

    def __init__(self, tui, gate, open_channel, close_channel):
        self.tui = tui
        self.gate = gate
        self.open_channel = open_channel
        self.close_channel = close_channel
        self._selection = _SelectionRequests()
        self.current = None
        self.tap = None
        self.transcriber = None
        self.listener = None
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.worker = None

    def start(self):
        """Begin serving choices."""
        if self.worker is not None:
            return
        self.worker = threading.Thread(
            target=self._serve, name="AudioChannelSwitch", daemon=True
        )
        self.worker.start()

    @property
    def desired(self):
        """Return the newest requested application name."""
        return self._selection.desired

    def select(self, application, *, on_applied=None):
        """Ask for an application, or for none. Always accepted."""
        self._selection.replace(application, on_applied)
        self.wake.set()
        return True

    def _serve(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            if self.stopping.is_set():
                return
            self.reconcile()

    def reconcile(self):
        """Make the session agree with the application last asked for.

        Held under a lock because shutdown reconciles too, on the thread that
        is closing the session, and two of these running at once would build a
        channel the other is retiring.
        """
        applied = False
        effective = None
        with self.lock:
            request = self._selection.snapshot()
            wanted = request.value
            if wanted == self.current:
                applied = True
            elif wanted is None:
                self._retire()
                applied = True
            elif self.tap is None:
                if self._open(wanted):
                    applied = True
                else:
                    self._selection.abandon(request)
            else:
                self.tap.follow(wanted)
                self.current = wanted
                applied = True
            effective = self.current
        if applied:
            self._selection.complete(request, effective)

    def _open(self, application):
        tap = StreamTap(application)
        try:
            transcriber, listener = self.open_channel(tap)
            transcriber.start()
        except Exception as error:
            # A far end that cannot be opened leaves the session running
            # without one, which is the state it was already in.
            self.tui.note(f"could not listen to {application}: {error}")
            return False
        self.tap, self.transcriber, self.listener = tap, transcriber, listener
        self.tui.hooks.on_audio_mute = muting(transcriber, listener)
        if self.tui.state.audio.muted:
            self.tui.hooks.on_audio_mute(True)
        self.gate.set_available(BASE_SPEAKERS | {"Audio"})
        self.current = application
        return True

    def _retire(self):
        """Close the channel, if one is open, and forget the far end."""
        transcriber, listener = self.transcriber, self.listener
        self.tap = self.transcriber = self.listener = None
        self.current = None
        self.tui.hooks.on_audio_mute = None
        self.gate.set_available(BASE_SPEAKERS)
        self.close_channel(transcriber, listener)

    def close(self):
        """Stop serving choices and close whatever is open."""
        self.stopping.set()
        self.wake.set()
        if self.worker is not None:
            self.worker.join(timeout=self.JOIN_TIMEOUT_SECONDS)
            self.worker = None
        self.select(None)
        self.reconcile()


def microphone_presence(mic_activity, audio_activity, tts):
    """Decide when the microphone is hearing the person sitting in front of it.

    Everything the speakers play reaches an open microphone, so on a laptop the
    tap goes busy for the assistant's own reply and for the far end as readily
    as for the user — and a turn held open by those would be held open for as
    long as they went on. Both are already known here: the speech engine says
    when it still owes a sentence, and the far end's own tap says when it is
    playing. That tap reports nothing until a far end is chosen, which is the
    same thing it reports when one is chosen and quiet.

    On a headset neither ever fires, and the suppressors cost nothing.
    """
    suppressors = [lambda: audio_activity.hearing_sound]
    if tts is not None:
        suppressors.append(tts.is_speaking)
    return SpeakerPresence(mic_activity, suppressors)


def remembering(hook, config, key, encode=lambda value: value):
    """Wrap a sidebar hook so an accepted change is written to the config file.

    Only an accepted change is recorded. A hook that refuses — a model the
    session cannot switch to, a speech engine still busy becoming the last one
    it was asked for — has changed nothing, and saving it would make the next
    session start somewhere this one never went.
    """

    def apply(value):
        accepted = hook(value)
        if accepted is not False:
            config.record(key, encode(value))
        return accepted

    return apply


def remembering_applied(hook, config, key, encode=lambda value: value):
    """Record an asynchronous selection only after it becomes effective."""

    def apply(value):
        return hook(
            value,
            on_applied=lambda applied: config.record(key, encode(applied)),
        )

    return apply


def attach_conversation_hooks(tui, conversation, tts, config, turn_silence):
    """Point the interface's controls at the conversation, its speech, and disk."""
    tui.hooks.on_user_text = lambda message: conversation.ingest(
        "Text",
        message.text,
        respond=True,
        images=message.images,
    )
    tui.hooks.on_interrupt = conversation.interrupt
    tui.hooks.on_codex_model = remembering(
        conversation.request_model, config, "codex_model"
    )
    tui.hooks.on_codex_effort = remembering(
        conversation.request_reasoning_effort, config, "codex_reasoning"
    )
    # Muting is not remembered: it is a thing done to one reply, or a few, and
    # a session that starts silent every time because of a moment's quiet is
    # not what "No voice reply" was asked to mean.
    tui.hooks.on_tts = tts.set_enabled
    tui.hooks.on_tts_provider = remembering(tts.set_provider, config, "tts_provider")
    tui.hooks.on_turn_silence = remembering_turn_silence(turn_silence, config)


def wire_transcript_recording(tui, conversation, recorder):
    """Attach continuous transcript recording and the slash commands that roll it."""
    tui.hooks.on_entry = recorder.record
    commands = build_command_router(tui, conversation, recorder)
    tui.hooks.on_command = commands.handle
    tui.hooks.list_commands = commands.specs


def build_command_router(tui, conversation, recorder) -> CommandRouter:
    """Register the session's typed slash commands and their palette copy."""
    commands = CommandRouter(tui)
    commands.register(
        "new",
        lambda command: reset_codex_session(command, conversation, tui, recorder),
        description="Start a fresh session and clear the transcript",
        aliases=("clear",),
    )
    commands.register(
        "help",
        lambda command: show_command_help(command, commands, tui),
        description="List available slash commands",
        aliases=("?",),
    )
    return commands


def show_command_help(command, commands: CommandRouter, tui) -> None:
    """Print the command catalog, or detail for one name when given."""
    if command.arguments:
        token = command.arguments[0].removeprefix("/")
        spec = commands.lookup(token)
        if spec is None:
            tui.note(f"unknown command: /{token}")
            return
        tui.note(spec.detail_line())
        return
    lines = ["commands:", *(spec.listing_line() for spec in commands.specs())]
    tui.note("\n".join(lines))


def reset_codex_session(command, conversation, tui, recorder):
    """Start a fresh Taga conversation and clear its visible transcript."""
    if command.arguments:
        tui.note("usage: /new")
        return
    if conversation.new_session():
        # Sweep open entries into the current file before rolling to a new one.
        tui.reset_transcript()
        recorder.roll()


def finish_recorded_session(tui, recorder, applications) -> None:
    """Sweep unfinished entries, close the file, and stop background refreshers."""
    tui.finish_recording()
    recorder.close()
    applications.stop()


def remembering_turn_silence(turn_silence, config):
    """Adopt a typed window and record the value actually applied.

    The applied value is what is saved, not the typed one: the two differ
    whenever the setting clamps, and the file has to describe the session that
    ran rather than the request that produced it.
    """

    def apply(seconds):
        applied = turn_silence.set(seconds)
        config.record("turn_silence", applied)
        return applied

    return apply


def main():
    parser, args = parse_startup_args()
    # Parsing comes first so `--help` and a rejected argument still answer while
    # a session is running; the lock comes before anything that reads or changes
    # the audio graph.
    acquire_single_instance_lock()
    # After the lock, so two sessions can never sweep each other's helpers, and
    # before anything is started, so this session's own are never candidates.
    sweep_orphans()
    codex_models = probe_codex_models()
    validate_codex_reasoning(parser, args, codex_models)
    selection = resolve_startup_selection(args)
    audio_stream = selection.audio_stream

    settings = startup_settings(selection, args)
    if args.save_config is not None:
        try:
            save_startup_config(args.save_config, settings)
        except RuntimeError as error:
            parser.error(str(error))
        print(f"Saved startup config: {args.save_config}", file=sys.stderr)
    # Sidebar changes are written back to the file the session started from,
    # so the next run opens where this one left off.
    config = StartupConfigFile(args.config, settings)
    model_arch = getattr(ModelArch, args.model.replace("-", "_").upper())

    print_startup_summary(args, selection)
    model_path, downloaded_arch = get_model_for_language(
        wanted_language=args.language,
        wanted_model_arch=model_arch,
    )

    gate = speaker_gate(selection)

    from .tui import VoiceCodexTUI

    turn_silence = TurnSilence(args.turn_silence)
    countdown = TurnSilenceClock(turn_silence)
    tts = build_speech(selection, args)
    tui = VoiceCodexTUI(
        build_session_state(args, selection, codex_models),
        countdown=countdown,
        speech=tts,
        on_policy=remembering(gate.set_policy, config, "taga_after"),
    )
    recorder = TranscriptRecorder()
    transcript_display = tui
    conversation = CodexConversation(
        CodexSettings(
            sandbox=args.sandbox,
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning,
            service_tier="fast" if args.codex_fast else None,
            prefire=args.codex_prefire,
        ),
        transcript_display,
        tts,
    )

    attach_conversation_hooks(tui, conversation, tts, config, turn_silence)
    wire_transcript_recording(tui, conversation, recorder)
    # The plan reads the conversation's own measured time-to-first-word, so
    # the moment a turn is guessed at tracks what Taga is actually doing.
    submitter = TranscriptSubmitter(
        conversation,
        gate,
        tts,
        prefire_plan=(
            PrefirePlan(conversation.latency) if args.codex_prefire else None
        ),
    )
    tui.hooks.on_end_turn = submitter.end_turn

    # Built before the listeners because each one reads the other's tap: the
    # microphone has to know when the far end is playing to tell its voice
    # from the person in the room.
    mic_activity, audio_activity = sound_taps(tui)

    tui.set_codex(thread=conversation.thread.id)

    parts = ChannelParts(
        submitter=submitter,
        display=transcript_display,
        confidence=args.confidence,
        turn_silence=turn_silence,
        countdown=countdown,
        model_path=model_path,
        model_arch=downloaded_arch,
    )
    initial_devices = (
        [(selection.device_index, selection.device)]
        if selection.device_index is not None and selection.device is not None
        else []
    )
    microphone = MicrophoneChannel(
        tui,
        partial(
            open_microphone_channel,
            parts,
            mic_activity,
            microphone_presence(mic_activity, audio_activity, tts),
        ),
        partial(close_microphone_channel, parts, mic_activity),
        devices=initial_devices,
    )
    desired_microphone = (
        selection.device["name"] if selection.device is not None else args.microphone
    )
    tui.hooks.on_microphone = remembering_applied(
        microphone.select, config, "microphone"
    )
    microphone.select(desired_microphone)
    microphone.reconcile()
    microphone.start()

    them = AudioChannel(
        tui,
        gate,
        partial(open_audio_channel, parts, audio_activity),
        partial(close_audio_channel, parts, audio_activity),
    )
    them.start()
    applications = ApplicationRefresher(tui)
    applications.start()
    tui.hooks.on_quit = applications.stop
    tui.hooks.on_audio_stream = remembering_applied(
        them.select, config, "audio_stream", encode=audio_stream_setting
    )
    them.select(audio_stream)
    run_session(
        tui,
        [],
        conversation,
        microphone=microphone,
        audio=them,
    )
    finish_recorded_session(tui, recorder, applications)


def run_entrypoint():
    """Run the application, reporting failures without a traceback.

    Both compatibility scripts call this so the shutdown behavior they
    advertise cannot drift apart.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
