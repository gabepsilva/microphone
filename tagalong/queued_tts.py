"""Shared queue lifecycle for sentence-at-a-time speech engines."""

from __future__ import annotations

import queue
import threading

from .domain import EchoMemory, SpeechActivity, TurnGate
from .playback import AudioPlayer


class QueuedSentenceTTS:
    """Own the provider-independent lifecycle of queued speech."""

    QUEUED_RETENTION_SECONDS = 120
    SPOKEN_RETENTION_SECONDS = 12

    # Set by the subclass once it has built the thread that drains the queue.
    # ``close`` joins it, so an engine that never assigns one cannot be shut
    # down; every provider starts its worker as the last step of ``__init__``.
    worker: threading.Thread

    def __init__(self, playback: AudioPlayer) -> None:
        """Take the player this engine speaks through and open its queue.

        A provider builds the player itself, because only it knows what the
        stream it synthesizes needs. Everything after that is the same
        whichever provider produced the audio.
        """
        self.playback = playback
        self.sentences = queue.Queue()
        self.stop_item = object()
        self.shutdown_requested = threading.Event()
        self.turns = TurnGate()
        self.echo = EchoMemory()
        self.activity = SpeechActivity()

    def _abandoned(self, turn) -> bool:
        """Report whether a turn's speech is no longer wanted.

        Synthesis, trimming, and playback each hand off to a thread or an
        event loop, so a turn can be interrupted or the engine closed between
        any two of them. Every stage re-asks rather than trusting the answer
        the stage before it got.
        """
        return self.shutdown_requested.is_set() or not self.turns.is_active(turn)

    def begin_turn(self) -> None:
        self.turns.begin_turn()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or silence future speech without rebuilding the pipeline."""
        if self.turns.set_enabled(enabled):
            self.interrupt()

    def speak(self, text: str) -> None:
        turn, accepting = self.turns.accepting_turn()
        if text and accepting and not self.shutdown_requested.is_set():
            self.echo.remember(text, retention=self.QUEUED_RETENTION_SECONDS)
            self.activity.queued()
            self.sentences.put_nowait((turn, text))

    def interrupt(self) -> None:
        """Stop the current response and discard all of its queued speech."""
        if self.shutdown_requested.is_set():
            return
        self.turns.cancel()
        self._discard_queued()
        self.activity.silenced()
        self.playback.stop()

    def _discard_queued(self) -> None:
        """Drop waiting sentences and stop treating them as unheard echoes."""
        while True:
            try:
                queued = self.sentences.get_nowait()
            except queue.Empty:
                return
            if queued is self.stop_item:
                self.sentences.put_nowait(queued)
                return
            self._forget_unspoken(queued[1])

    def _forget_unspoken(self, text: str) -> None:
        """Cut a never-played sentence's retention back to a spoken one's."""
        self.echo.remember(text, retention=self.SPOKEN_RETENTION_SECONDS, replace=True)

    def is_likely_echo(self, text: str) -> bool:
        """Return True when a transcript resembles recently queued TTS."""
        return self.echo.matches(text)

    def is_speaking(self) -> bool:
        """Report whether this engine still has speech to deliver."""
        return self.activity.speaking

    def close(self) -> None:
        self.shutdown_requested.set()
        self.activity.silenced()
        self.playback.stop()
        self.sentences.put_nowait(self.stop_item)
        self.worker.join(timeout=3)
