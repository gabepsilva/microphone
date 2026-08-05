"""The queue lifecycle every sentence engine inherits.

These are the semantics ``QueuedSentenceTTS`` owns outright: which sentences
are accepted, what interruption does to the queue and to echo memory, and what
``is_speaking`` reports along the way. They are exercised once, against a
minimal engine, because Edge and Piper run this code rather than each
implementing it. What the providers do with a sentence once the queue hands it
over is covered per engine, and the lifecycle each provider's worker must
honour is covered in ``test_speech_contract.py``.
"""

from __future__ import annotations

import threading
import time

import pytest

from tagalong.playback import AudioPlayer
from tagalong.queued_tts import QueuedSentenceTTS

WAIT_SECONDS = 10


class FakePlayback(AudioPlayer):
    """Count the stops the lifecycle asks for, without a player process.

    A real ``AudioPlayer`` rather than a stand-in, so the engine is built
    through the same constructor the providers use; nothing here ever starts
    the process it names.
    """

    def __init__(self):
        super().__init__("/usr/bin/ffplay")
        self.stops = 0

    def stop(self):
        self.stops += 1


class LoopbackEngine(QueuedSentenceTTS):
    """The smallest engine the lifecycle can drive: a worker that only reads.

    The provider-shaped work is replaced by recording the sentence, so a test
    can tell a sentence the queue accepted from one it refused without waiting
    on synthesis or playback.
    """

    def __init__(self, held=None):
        super().__init__(FakePlayback())
        self.spoken: list[str] = []
        self.arrived = threading.Event()
        # Set by tests that need the worker parked on one sentence while they
        # queue others behind it.
        self.held = held
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while True:
            item = self.sentences.get()
            if item is self.stop_item:
                return
            turn, text = item
            if self.held is not None:
                self.held.wait(WAIT_SECONDS)
            if not self._abandoned(turn):
                self.spoken.append(text)
            self.arrived.set()
            self.echo.remember(
                text, retention=self.SPOKEN_RETENTION_SECONDS, replace=True
            )
            self.activity.finished()


def wait_until(predicate):
    """Wait for a lifecycle state rather than for a fixed delay."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def engine():
    started = LoopbackEngine()
    yield started
    started.close()


@pytest.fixture
def parked():
    """An engine whose worker is holding the sentence it took."""
    release = threading.Event()
    started = LoopbackEngine(held=release)
    yield started
    release.set()
    started.close()


# --------------------------------------------------------------------------
# What the queue accepts
# --------------------------------------------------------------------------


def test_empty_text_is_never_queued(engine) -> None:
    engine.begin_turn()
    engine.speak("")
    engine.speak("Real speech.")

    assert engine.arrived.wait(WAIT_SECONDS)
    assert wait_until(lambda: engine.spoken == ["Real speech."])


def test_speech_queued_after_an_interrupt_is_refused_until_a_new_turn(
    engine,
) -> None:
    engine.begin_turn()
    engine.interrupt()

    engine.speak("Belongs to the turn that was cut off.")
    assert engine.sentences.empty()

    engine.begin_turn()
    engine.speak("Belongs to the turn that replaced it.")

    assert wait_until(
        lambda: engine.spoken == ["Belongs to the turn that replaced it."]
    )


def test_a_silenced_engine_refuses_speech_until_it_is_enabled_again(
    engine,
) -> None:
    engine.set_enabled(False)
    engine.begin_turn()
    engine.speak("Said while muted.")

    assert engine.sentences.empty()

    engine.set_enabled(True)
    engine.begin_turn()
    engine.speak("Said once unmuted.")

    assert wait_until(lambda: engine.spoken == ["Said once unmuted."])


def test_silencing_an_engine_stops_what_it_is_already_saying(engine) -> None:
    """Muting is immediate: the queue empties rather than draining."""
    engine.begin_turn()

    engine.set_enabled(False)

    assert engine.playback.stops == 1


def test_enabling_an_engine_that_is_already_speaking_lets_it_finish(
    engine,
) -> None:
    engine.begin_turn()

    engine.set_enabled(True)

    assert engine.playback.stops == 0


def test_speech_queued_after_close_is_refused(engine) -> None:
    engine.begin_turn()
    engine.close()

    engine.speak("Arrived during shutdown.")

    assert engine.spoken == []


# --------------------------------------------------------------------------
# What the engine reports it is doing
# --------------------------------------------------------------------------


def test_an_engine_with_nothing_to_say_is_not_speaking(engine) -> None:
    assert engine.is_speaking() is False


def test_an_engine_is_speaking_from_the_moment_a_sentence_is_accepted(
    parked,
) -> None:
    """Not from the moment audio starts: synthesis is part of the answer."""
    parked.begin_turn()
    parked.speak("Hello there.")

    assert parked.is_speaking() is True


def test_an_engine_falls_quiet_once_the_last_sentence_has_played(engine) -> None:
    engine.begin_turn()
    engine.speak("Hello there.")

    assert wait_until(lambda: engine.is_speaking() is False)


def test_an_interrupted_engine_stops_reporting_speech(parked) -> None:
    parked.begin_turn()
    parked.speak("First.")
    parked.speak("Second.")

    parked.interrupt()

    assert parked.is_speaking() is False


# --------------------------------------------------------------------------
# Interruption and echo memory
# --------------------------------------------------------------------------


def test_an_interrupt_stops_treating_dropped_speech_as_an_echo(parked) -> None:
    """A sentence the interrupt threw away must not filter the microphones.

    Queued speech is remembered for two minutes because it may not be said
    for a while. One that is dropped is never said at all, so its retention
    has to fall back to a spoken sentence's — otherwise the speaker's own
    words are filtered as echo of a reply they cut off and never heard.
    """
    parked.begin_turn()
    parked.speak("Holds the worker.")
    dropped = "I can roll it back if you would rather not ship today."
    parked.speak(dropped)
    assert wait_until(lambda: not parked.sentences.empty())

    parked.interrupt()

    assert parked.spoken == []
    elapsed = parked.SPOKEN_RETENTION_SECONDS + 1
    parked.echo._clock = lambda: time.monotonic() + elapsed
    assert parked.is_likely_echo(dropped) is False


def test_queued_speech_is_echo_matchable_before_it_is_ever_played(parked) -> None:
    parked.begin_turn()
    parked.speak("Holds the worker.")
    parked.speak("Still waiting its turn.")

    assert parked.is_likely_echo("still waiting its turn") is True
    assert parked.is_likely_echo("what time is the meeting") is False


def test_interrupting_a_closed_engine_leaves_the_shutdown_alone(engine) -> None:
    """A closed engine is torn down already.

    ``interrupt`` must return without touching turn state or re-entering the
    queue drain, so a stop request in flight cannot be undone by one.
    """
    engine.begin_turn()
    engine.close()
    cancelled_before = engine.turns.cancelled

    engine.interrupt()

    assert engine.turns.cancelled == cancelled_before
    assert engine.worker.is_alive() is False


def test_draining_the_queue_leaves_the_stop_item_for_the_worker(parked) -> None:
    """Discarding queued speech must not throw away what ends the worker.

    ``_discard_queued`` walks the same queue the stop item sits in. Called
    directly here because ``interrupt`` refuses once a close has started, so
    this guard has no reachable caller — it exists so that ordering cannot
    become load-bearing later.
    """
    parked.begin_turn()
    parked.speak("Holds the worker.")
    parked.speak("Waiting behind the stop item.")
    parked.sentences.put_nowait(parked.stop_item)

    parked._discard_queued()

    assert parked.sentences.get_nowait() is parked.stop_item


def test_closing_an_engine_ends_its_worker(engine) -> None:
    engine.close()

    assert engine.worker.is_alive() is False
