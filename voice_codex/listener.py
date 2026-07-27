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
    def __init__(  # noqa: PLR0913 - pre-existing: audio adapter wiring
        self,
        confidence_threshold,
        turn_silence,
        speaker,
        submit,
        presentation: TranscriptSink,
        on_speech=None,
        countdown=None,
    ):
        self.confidence_threshold = confidence_threshold
        self.turn_silence = turn_silence
        self.speaker = speaker
        self.submit = submit
        self.presentation = presentation
        self.on_speech = on_speech
        # Optional so a listener can run without a display attached to it.
        self.countdown = countdown
        self.lock = threading.Lock()
        self.pending = []
        self.timer = None
        self.timer_generation = 0
        self.speech_callback_triggered = False
        self.muted = False

    def _stop_counting(self):
        """Take this speaker off the countdown the interface is showing."""
        if self.countdown is not None:
            self.countdown.cleared(self.speaker)

    def set_muted(self, muted):
        """Stop submitting microphone speech while preserving the listener."""
        with self.lock:
            self.muted = muted
            if muted:
                self.timer_generation += 1
                self.pending.clear()
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None
        if muted:
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

    def _flush(self, generation):
        with self.lock:
            if generation != self.timer_generation:
                return
            text = " ".join(self.pending).strip()
            self.pending.clear()
            self.timer = None
        self._stop_counting()
        if text:
            self.presentation.finish_turn(self.speaker)
            self.submit(self.speaker, text)

    def _start_timer(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer_generation += 1
        self.timer = threading.Timer(
            self.turn_silence.seconds,
            self._flush,
            args=(self.timer_generation,),
        )
        self.timer.daemon = True
        self.timer.start()
        if self.countdown is not None:
            self.countdown.started(self.speaker)

    def _cancel_timer(self):
        with self.lock:
            self.timer_generation += 1
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
        self._stop_counting()

    def flush_now(self):
        """Submit what is already transcribed without waiting out the silence.

        The silence window exists to decide that a speaker has stopped talking.
        When another speaker's turn is already being answered that question has
        been overtaken: whatever this listener has buffered is context for the
        reply being built now, and holding it until this speaker's own timer
        fires would deliver it one request too late.
        """
        with self.lock:
            self.timer_generation += 1
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            generation = self.timer_generation
        self._flush(generation)

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
                self._start_timer()

    def close(self):
        with self.lock:
            self.timer_generation += 1
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
        self._stop_counting()
        self.presentation.close_speaker(self.speaker)


class TranscriptSubmitter:
    """Send completed turns to Codex, discarding the assistant's own TTS echo.

    Both microphones can hear Codex speaking. A transcript that matches recent
    speech is dropped rather than answered, and a partial that matches it must
    not interrupt playback either.
    """

    ECHO_PRONE_SPEAKERS = ("User Voice", "Them")

    def __init__(self, conversation, gate, tts, stream=sys.stderr):
        self.conversation = conversation
        self.gate = gate
        self.tts = tts
        self.stream = stream
        self.listeners = []

    def add_listener(self, listener):
        """Register a channel whose buffer may be swept into a reply's context."""
        self.listeners.append(listener)

    def channel(
        self, confidence_threshold, turn_silence, speaker, presentation, countdown=None
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
        )
        self.add_listener(listener)
        return listener

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
        if (
            self.tts is not None
            and speaker in self.ECHO_PRONE_SPEAKERS
            and self.tts.is_likely_echo(text)
        ):
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
