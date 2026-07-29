#!/usr/bin/env python3
"""Wire the always-listening User/Them/Codex conversation together.

This module is the composition root and nothing else. The runtime parts it
assembles are independent of one another — none of them imports any of the
others:

  * ``capture``  — microphone and application-stream audio into Moonshine,
                   the second by way of the PipeWire tap in ``streams``
  * ``listener`` — transcription events into completed turns
  * ``codex``    — the Codex thread and its streamed turns
  * ``speech``   — which synthesizer speaks Codex responses, and the two
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
)
from .catalog import probe_codex_models
from .choosers import NO_THEM_STREAM
from .codex import CodexConversation, CodexSettings
from .config import StartupConfigFile, save_startup_config
from .domain import (
    PrefirePlan,
    SpeakerGate,
    SpeakerPresence,
    TurnSilence,
    TurnSilenceClock,
)
from .listener import TranscriptSubmitter, tts_switch
from .presentation import NO_DEVICE
from .session import sweep_orphans
from .speech import SwitchableSpeech, provider_switch
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
    / f"voice-codex-{os.getuid()}.lock"
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
        raise RuntimeError("Another voice_codex session is already running.") from error
    _INSTANCE_LOCK = lock_file
    if not _LOCK_RELEASE_REGISTERED:
        atexit.register(_release_single_instance_lock)
        _LOCK_RELEASE_REGISTERED = True


def build_speech(selection, args):
    """Build the session's speech, or nothing when the session is silent."""
    if not selection.tts_enabled:
        return None
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
    return SoundActivityReporter(tui, "mic"), SoundActivityReporter(tui, "them")


BASE_SPEAKERS = frozenset({"User Voice"})


def them_stream_setting(application):
    """Record a chosen application the way ``--them-stream`` reads it back."""
    return NO_THEM_STREAM if application is None else application


def speaker_gate(selection):
    """Build the response policy, limited to the speakers this session has.

    It starts with the microphone alone. Selecting "both" before a far end
    exists must not make Them replies possible, and the gate is told when one
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


def open_them_channel(parts, activity, tap):
    """Build the far end's listener and transcriber around one tap."""
    listener = parts.submitter.channel(
        parts.confidence,
        parts.turn_silence,
        "Them",
        parts.display,
        countdown=parts.countdown,
        # No suppressors: the tap carries the links this session made and
        # nothing else, and Codex's own playback is never one of them. It is
        # not that its voice is filtered off this channel — it never reaches it.
        presence=SpeakerPresence(activity),
    )
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


def close_them_channel(parts, transcriber, listener):
    """Retire the far end's channel in the order the session's shutdown uses.

    Nothing may reach a listener after it has flushed, and the listener is
    unregistered last so no reply can sweep a closed channel's buffer in as
    context for a far end nobody is listening to any more.
    """
    transcriber.stop()
    listener.close()
    transcriber.close()
    parts.submitter.remove_listener(listener)


class ThemChannel:
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
        self.desired = None
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
            target=self._serve, name="ThemChannelSwitch", daemon=True
        )
        self.worker.start()

    def select(self, application):
        """Ask for an application, or for none. Always accepted."""
        self.desired = application
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
        with self.lock:
            wanted = self.desired
            if wanted == self.current:
                return
            if wanted is None:
                self._retire()
            elif self.tap is None:
                self._open(wanted)
            else:
                self.tap.follow(wanted)
                self._announce(wanted)

    def _open(self, application):
        tap = StreamTap(application)
        try:
            transcriber, listener = self.open_channel(tap)
            transcriber.start()
        except Exception as error:
            # A far end that cannot be opened leaves the session running
            # without one, which is the state it was already in.
            self.tui.note(f"could not listen to {application}: {error}")
            self.desired = None
            return
        self.tap, self.transcriber, self.listener = tap, transcriber, listener
        self.tui.hooks.on_them_mute = muting(transcriber, listener)
        self.gate.set_available(BASE_SPEAKERS | {"Them"})
        self._announce(application)

    def _announce(self, application):
        self.current = application
        self.tui.set_audio("them", device=application)

    def _retire(self):
        """Close the channel, if one is open, and forget the far end."""
        transcriber, listener = self.transcriber, self.listener
        self.tap = self.transcriber = self.listener = None
        self.current = None
        self.tui.hooks.on_them_mute = None
        self.gate.set_available(BASE_SPEAKERS)
        self.tui.set_audio("them", device=NO_DEVICE)
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


def microphone_presence(mic_activity, them_activity, tts):
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
    suppressors = [lambda: them_activity.hearing_sound]
    if tts is not None:
        suppressors.append(tts.is_speaking)
    return SpeakerPresence(mic_activity, suppressors)


def remembering(hook, config, key, encode=lambda value: value):
    """Wrap a sidebar hook so an accepted change is written to the config file.

    Only an accepted change is recorded. A hook that refuses — a model the
    session cannot switch to, a speech engine it does not have — has changed
    nothing, and saving it would make the next session start somewhere this
    one never went.
    """

    def apply(value):
        accepted = hook(value)
        if accepted is not False:
            config.record(key, encode(value))
        return accepted

    return apply


def attach_conversation_hooks(tui, conversation, tts, config, turn_silence):
    """Point the interface's controls at the conversation, its speech, and disk."""
    tui.hooks.on_user_text = lambda text: conversation.ingest(
        "User Text", text, respond=True
    )
    tui.hooks.on_interrupt = conversation.interrupt
    tui.hooks.on_codex_model = remembering(
        conversation.request_model, config, "codex_model"
    )
    tui.hooks.on_codex_effort = remembering(
        conversation.request_reasoning_effort, config, "codex_reasoning"
    )
    tui.hooks.on_tts = remembering(
        tts_switch(tts), config, "tts", encode=lambda on: "on" if on else "off"
    )
    tui.hooks.on_tts_provider = remembering(
        provider_switch(tts), config, "tts_provider"
    )
    tui.hooks.on_turn_silence = remembering_turn_silence(turn_silence, config)


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
    them_stream = selection.them_stream

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
        on_policy=remembering(gate.set_policy, config, "codex_after"),
    )
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
    # The plan reads the conversation's own measured time-to-first-word, so
    # the moment a turn is guessed at tracks what Codex is actually doing.
    submitter = TranscriptSubmitter(
        conversation,
        gate,
        tts,
        prefire_plan=(
            PrefirePlan(conversation.latency) if args.codex_prefire else None
        ),
    )

    # Built before the listeners because each one reads the other's tap: the
    # microphone has to know when the far end is playing to tell its voice
    # from the person in the room.
    mic_activity, them_activity = sound_taps(tui)

    user_listener = submitter.channel(
        args.confidence,
        turn_silence,
        "User Voice",
        transcript_display,
        countdown=countdown,
        presence=microphone_presence(mic_activity, them_activity, tts),
    )
    user_transcriber = metered_mic_transcriber(
        model_path=model_path,
        model_arch=downloaded_arch,
        update_interval=0.25,
        device=selection.device_index,
        samplerate=16000,
        channels=1,
        level_reporter=mic_activity,
    )
    user_transcriber.add_listener(user_listener)

    tui.hooks.on_mute = muting(user_transcriber, user_listener)
    tui.set_audio("mic", device=selection.device["name"])
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
    them = ThemChannel(
        tui,
        gate,
        partial(open_them_channel, parts, them_activity),
        partial(close_them_channel, parts),
    )
    them.start()
    applications = ApplicationRefresher(tui)
    applications.start()
    tui.hooks.on_quit = applications.stop
    tui.hooks.on_them_stream = remembering(
        them.select, config, "them_stream", encode=them_stream_setting
    )
    them.select(them_stream)
    run_session(tui, [(user_transcriber, user_listener)], conversation, them)
    applications.stop()


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
