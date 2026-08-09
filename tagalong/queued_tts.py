"""Shared queue lifecycle for sentence-at-a-time speech engines."""

from __future__ import annotations

import queue
import threading

from .domain import EchoMemory, SpeechActivity, TurnGate
from .media_controls import (
    MEDIA_STATUS_PLAYING,
    MEDIA_STATUS_STOPPED,
    MediaControlsPort,
    NullMediaControls,
)
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
        # The desktop's media surface, neutral until a session says otherwise.
        # Announcements mirror the speech state; the port decides what, if
        # anything, the desktop gets to hold on to.
        self.media_controls: MediaControlsPort = NullMediaControls()
        self.activity.observe(self._on_activity)

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

    def _on_activity(self, activity) -> None:
        """Mirror the speech state out to the media port, if it is interested.

        The status climbs with the first queued sentence and falls with the
        last finished or silenced one — the span the room hears speech, which
        is exactly the span a media surface should show.
        """
        status = MEDIA_STATUS_PLAYING if activity.speaking else MEDIA_STATUS_STOPPED
        self.media_controls.publish(status)

    def set_media_controls(self, port: MediaControlsPort) -> None:
        """Announce through *port* from now on, starting with the current state.

        The port only earns its place if it is right the moment it arrives: a
        session that is mid-reply when the desktop notices it must not appear
        idle.
        """
        self.media_controls = port
        self._on_activity(self.activity)

    def on_sentence_audible(self, text: str) -> None:
        """Tell the desktop which sentence is actually coming out of the speaker.

        Called by a provider right before playback of one sentence begins. The
        title is what MPRIS marquees show, so it is the sentence being said,
        not the sentence being synthesized or the one queued up next.
        """
        self.media_controls.publish(MEDIA_STATUS_PLAYING, text)

    def close(self) -> None:
        self.shutdown_requested.set()
        self.activity.silenced()
        self.playback.stop()
        self.sentences.put_nowait(self.stop_item)
        self.worker.join(timeout=3)
