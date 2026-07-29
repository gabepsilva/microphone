#!/usr/bin/env python3
"""Turn streamed transcription events into completed turns for Codex.

The silence timer here is a daemon thread, cancelled on mute and on close, so
a pending flush cannot fire against a torn-down display.
"""

from __future__ import annotations

import sys
import threading

from moonshine_voice.transcriber import TranscriptEventListener

from .presentation import TranscriptSink


class ConversationListener(TranscriptEventListener):
    # A deadline reached while its speaker is still audible is pushed back by
    # this much rather than by another whole window: the transcription that
    # will cancel it properly is about half a second behind the speech, so the
    # grace only has to outlast that, and a speaker who really has stopped is
    # answered a grace late instead of a window late.
    EXTENSION_GRACE = 0.5

    def __init__(  # noqa: PLR0913 - pre-existing: audio adapter wiring
        self,
        confidence_threshold,
        turn_silence,
        speaker,
        submit,
        presentation: TranscriptSink,
        on_speech=None,
        countdown=None,
        prefire=None,
        presence=None,
    ):
        self.confidence_threshold = confidence_threshold
        self.turn_silence = turn_silence
        self.speaker = speaker
        self.submit = submit
        self.presentation = presentation
        self.on_speech = on_speech
        # Optional so a listener can run without a display attached to it.
        self.countdown = countdown
        # Optional so a session can wait out the whole window instead, which
        # is what ``--no-codex-prefire`` asks for and what the tests compare
        # every pre-fired path against.
        self.prefire = prefire
        # Optional so a channel with no level tap behind it — and every test
        # that predates one — waits the window out exactly as before.
        self.presence = presence
        self.lock = threading.Lock()
        self.pending = []
        self.timer = None
        self.prefire_timer = None
        self.prefired = False
        self.timer_generation = 0
        self.extensions = 0
        self.speech_callback_triggered = False
        self.muted = False

    def _stop_counting(self):
        """Take this speaker off the countdown the interface is showing."""
        if self.countdown is not None:
            self.countdown.cleared(self.speaker)

    def _stop_timers(self):
        """Cancel both timers and retire their generation. Caller holds the lock."""
        self.timer_generation += 1
        for timer in (self.timer, self.prefire_timer):
            if timer is not None:
                timer.cancel()
        self.timer = None
        self.prefire_timer = None

    def _drop_prefire(self):
        """Abandon a speculative turn this listener started. Lock not held."""
        if self.prefire is None:
            return
        with self.lock:
            outstanding = self.prefired
            self.prefired = False
        if outstanding:
            self.prefire.cancel(self.speaker)

    def set_muted(self, muted):
        """Stop submitting microphone speech while preserving the listener."""
        with self.lock:
            self.muted = muted
            if muted:
                self.pending.clear()
                self._stop_timers()
                self.extensions = 0
        if muted:
            self._drop_prefire()
            self._stop_counting()

    def _is_muted(self):
        with self.lock:
            return self.muted

    def _text(self, line):
        if line.words:
            return " ".join(
                word.word.strip()
                for word in line.words
                if word.confidence >= self.confidence_threshold
            ).strip()
        return line.text.strip()

    def _flush(self, generation, extendable=True):
        text = ""
        with self.lock:
            if generation != self.timer_generation:
                return
            extended = extendable and self._extending()
            if extended:
                self.extensions += 1
                # Re-armed while the lock is still held, so the turn is never
                # momentarily left with no deadline on it at all.
                self._start_timer(self.EXTENSION_GRACE, speculate=False)
            else:
                text = " ".join(self.pending).strip()
                self.pending.clear()
                self.timer = None
            prefired = self.prefired
            self.prefired = False
        if extended:
            # The speaker is still going, so a speculative answer to what they
            # had said by now would be answering half a sentence. The countdown
            # was re-armed above, which is what puts the wait back on screen.
            if prefired and self.prefire is not None:
                self.prefire.cancel(self.speaker)
            return
        self._stop_counting()
        if not text:
            return
        self.presentation.finish_turn(self.speaker)
        # A speculative turn that survived to here was right: the window
        # closed without the speaker resuming, so it is the reply. Submitting
        # again would answer the same words twice.
        if prefired and self.prefire is not None and self.prefire.commit(self.speaker):
            return
        self.submit(self.speaker, text)

    def _speculate(self, generation):
        """Start answering before the window closes, if it is still this turn."""
        with self.lock:
            if generation != self.timer_generation:
                return
            self.prefire_timer = None
            text = " ".join(self.pending).strip()
            if not text or self.prefired:
                return
            self.prefired = True
        if self.prefire is None or not self.prefire.start(self.speaker, text):
            with self.lock:
                self.prefired = False

    def _prefire_delay(self, window):
        """Seconds to wait before guessing this turn is over, or None to wait.

        A guess that would land at or after the deadline is not a guess; it is
        a slower way of doing what the deadline already does.
        """
        if self.prefire is None:
            return None
        delay = self.prefire.delay(window)
        return delay if 0 < delay < window else None

    def _extension_budget(self) -> int:
        """How many graces one turn may be held open for.

        Bounded by the window itself, so however noisy a room is, a turn waits
        at most twice the silence it was configured with before it is sent. A
        level tap cannot tell speech from a fan, and without a ceiling a room
        loud enough to hold the tap open would never submit anything at all.
        """
        return max(1, int(self.turn_silence.seconds / self.EXTENSION_GRACE))

    def _extending(self) -> bool:
        """Whether this deadline should be pushed back rather than fired.

        Caller holds the lock. An empty buffer is never extended: there is no
        turn to protect, and holding one open would keep a speaker on the
        countdown for sound that is never going to become words.
        """
        return (
            self.presence is not None
            and bool(self.pending)
            and self.extensions < self._extension_budget()
            and self.presence.speaking()
        )

    def _start_timer(self, window=None, speculate=True):
        """Arm the deadline, and the speculative turn that runs ahead of it.

        ``window`` is given only for an extension, which runs on a grace
        rather than the configured wait, and which does not speculate: the
        reason it was extended is that the speaker is probably mid-sentence.
        """
        window = self.turn_silence.seconds if window is None else window
        self._stop_timers()
        generation = self.timer_generation
        self.timer = threading.Timer(window, self._flush, args=(generation,))
        self.timer.daemon = True
        self.timer.start()
        delay = self._prefire_delay(window) if speculate else None
        if delay is not None:
            self.prefire_timer = threading.Timer(
                delay, self._speculate, args=(generation,)
            )
            self.prefire_timer.daemon = True
            self.prefire_timer.start()
        if self.countdown is not None:
            self.countdown.started(self.speaker, window)

    def _cancel_timer(self):
        with self.lock:
            self._stop_timers()
            # Transcription has caught up with the speech the tap heard, so
            # whatever the extensions were spent on has been paid for.
            self.extensions = 0
        self._drop_prefire()
        self._stop_counting()

    def flush_now(self):
        """Submit what is already transcribed without waiting out the silence.

        The silence window exists to decide that a speaker has stopped talking.
        When another speaker's turn is already being answered that question has
        been overtaken: whatever this listener has buffered is context for the
        reply being built now, and holding it until this speaker's own timer
        fires would deliver it one request too late.

        For the same reason this never extends. Holding the buffer back
        because its speaker is still audible is exactly what the caller has
        already decided not to wait for.
        """
        with self.lock:
            self._stop_timers()
            generation = self.timer_generation
        self._flush(generation, extendable=False)

    def on_line_started(self, event):  # noqa: ARG002 - Textual/Codex callback signature is fixed
        # Speech has resumed. Keep all completed lines buffered and wait for
        # this new line to finish before considering the turn complete.
        if self._is_muted():
            return
        self._cancel_timer()
        self.speech_callback_triggered = False

    def on_line_text_changed(self, event):
        # Partial text means this speaker is actively continuing the same turn.
        if self._is_muted():
            return
        self._cancel_timer()
        partial = self._text(event.line)
        self.presentation.update(self.speaker, partial)
        if (
            partial
            and self.on_speech is not None
            and not self.speech_callback_triggered
        ):
            self.speech_callback_triggered = self.on_speech(partial)

    def on_line_completed(self, event):
        if self._is_muted():
            return
        text = self._text(event.line)
        with self.lock:
            if text:
                self.pending.append(text)
                self.presentation.commit(self.speaker, text)
            if self.pending:
                # A completed line is a fresh window, so the budget it may be
                # held open on starts over with it.
                self.extensions = 0
                self._start_timer()

    def close(self):
        with self.lock:
            self._stop_timers()
        self._drop_prefire()
        self._stop_counting()
        self.presentation.close_speaker(self.speaker)


class PrefireChannel:
    """The four moments a speculative turn has, as one listener sees them.

    A listener knows when a turn is probably over, when it turned out to be
    over, and when it turned out not to be. It does not know about echo
    gating, response policy, or Codex threads. This is the seam between the
    two: schedule on one side, consequences on the other.
    """

    def __init__(self, submitter):
        self.submitter = submitter

    def delay(self, window):
        return self.submitter.prefire_plan.delay(window)

    def start(self, speaker, text) -> bool:
        return self.submitter.prefire(speaker, text)

    def commit(self, speaker) -> bool:
        return self.submitter.commit_prefire(speaker)

    def cancel(self, speaker) -> bool:
        return self.submitter.cancel_prefire(speaker)


class TranscriptSubmitter:
    """Send completed turns to Codex, discarding the assistant's own TTS echo.

    Both microphones can hear Codex speaking. A transcript that matches recent
    speech is dropped rather than answered, and a partial that matches it must
    not interrupt playback either.
    """

    ECHO_PRONE_SPEAKERS = ("User Voice", "Them")

    def __init__(self, conversation, gate, tts, stream=sys.stderr, prefire_plan=None):
        self.conversation = conversation
        self.gate = gate
        self.tts = tts
        self.stream = stream
        # Absent when the session waits out every window in full, which is
        # what ``--no-codex-prefire`` selects.
        self.prefire_plan = prefire_plan
        self.listeners = []

    def add_listener(self, listener):
        """Register a channel whose buffer may be swept into a reply's context."""
        self.listeners.append(listener)

    def remove_listener(self, listener):
        """Retire a channel that no longer exists.

        A listener left registered after its transcriber is closed is not
        inert: it still holds whatever it had buffered, and the next reply
        would sweep that stale text in as context for a far end nobody is
        listening to any more.
        """
        if listener in self.listeners:
            self.listeners.remove(listener)

    def channel(  # noqa: PLR0913 - audio adapter wiring, as the listener itself
        self,
        confidence_threshold,
        turn_silence,
        speaker,
        presentation,
        countdown=None,
        presence=None,
    ):
        """Build a listener for a speaker and register it in one step.

        Registration is not left to the caller because a listener that submits
        turns but was never registered still works — it just silently stops
        contributing context, which is the kind of omission a session only
        reveals as a reply that is missing something.
        """
        listener = ConversationListener(
            confidence_threshold,
            turn_silence,
            speaker,
            self.submit,
            presentation,
            on_speech=self.handle_speech,
            countdown=countdown,
            prefire=self._prefire_channel(),
            presence=presence,
        )
        self.add_listener(listener)
        return listener

    def _prefire_channel(self):
        return None if self.prefire_plan is None else PrefireChannel(self)

    def _is_echo(self, speaker, text) -> bool:
        """Report whether a transcript is Codex hearing itself speak."""
        return (
            self.tts is not None
            and speaker in self.ECHO_PRONE_SPEAKERS
            and self.tts.is_likely_echo(text)
        )

    def prefire(self, speaker, text) -> bool:
        """Start answering a turn the silence window has not yet confirmed.

        The same two gates a real submission passes apply here. A speculative
        turn answering the assistant's own echo, or answering a speaker the
        policy stays silent for, would be a turn nobody could have wanted —
        and unlike a late reply, nothing downstream would catch it.
        """
        if not self.gate.should_respond(speaker) or self._is_echo(speaker, text):
            return False
        self._sweep_context(speaker)
        return self.conversation.prefire(speaker, text)

    def commit_prefire(self, speaker) -> bool:
        return self.conversation.commit_prefire(speaker)

    def cancel_prefire(self, speaker) -> bool:
        return self.conversation.cancel_prefire(speaker)

    def _sweep_context(self, replying_to):
        """Flush the channels that only supply context, so this reply carries it.

        Only the channels the policy does not answer are swept. Flushing one
        the policy does answer would queue a second reply from speech its own
        speaker has not finished, which is a turn nobody asked for rather than
        context for this one.
        """
        for listener in self.listeners:
            if listener.speaker != replying_to and not self.gate.should_respond(
                listener.speaker
            ):
                listener.flush_now()

    def submit(self, speaker, text):
        if self._is_echo(speaker, text):
            print(
                f"[ignored likely Codex TTS echo from {speaker}: {text}]",
                file=self.stream,
                flush=True,
            )
            return
        respond = self.gate.should_respond(speaker)
        # Swept before this turn is ingested, so the context a speaker supplied
        # earlier is ordered earlier in the request that carries both.
        if respond:
            self._sweep_context(speaker)
        self.conversation.ingest(speaker, text, respond=respond)

    def handle_speech(self, partial):
        """Interrupt playback for real speech; report whether it was real."""
        if self.tts is None or self.tts.is_likely_echo(partial):
            return False
        self.tts.interrupt()
        return True


def tts_switch(tts):
    """Build the interface's speech toggle.

    It reports whether the session has speech at all, so a session started
    without Edge TTS says so instead of silently appearing to enable it.
    """

    def toggle(enabled):
        if tts is None:
            return False
        tts.set_enabled(enabled)
        return True

    return toggle
