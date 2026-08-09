"""The lifecycle every SpeechEngine owes the session, whatever it is.

The queue semantics themselves are one implementation and are covered once in
``test_queued_tts.py``. What differs is what each engine runs underneath them:
Edge drives an asyncio pipeline with a prefetch window, Piper a worker that
waits on a model it is still loading, and ``SwitchableSpeech`` forwards to a
delegate it can replace mid-session. Those are three genuinely different ways
to break the same promises, so the promises are asserted against all three.

Each provider brings its own boundary to fake, which is why the harnesses are
built here rather than shared with the per-engine suites: those cover synthesis,
trimming, prefetch, and failure reporting, and the fakes they need for it are
richer than a lifecycle assertion should have to understand.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from tagalong.piper_tts import PiperSentenceTTS
from tagalong.speech import SwitchableSpeech
from tagalong.tts import EdgeSentenceTTS

WAIT_SECONDS = 10
SAMPLE_RATE = 22050


class Player:
    """Stand in for ffplay, and hold the pipeline open when a test asks."""

    def __init__(self, recorder, gate):
        self.recorder = recorder
        self.gate = gate
        self.terminated = False
        # A clean exit, read after the write returns; anything else would send
        # these tests down the player's failure-reporting path.
        self.returncode = 0

    def communicate(self, input=None):
        self.recorder(input)
        if self.gate is not None:
            self.gate.wait(WAIT_SECONDS)
        return b"", b""

    def poll(self):
        return 0

    def send_signal(self, sig):
        del sig

    def terminate(self):
        # A terminated ffplay exits, so the blocked write returns. Modelling
        # that is what lets a test assert shutdown actually completes.
        self.terminated = True
        if self.gate is not None:
            self.gate.set()

    def wait(self, timeout=None):  # noqa: ARG002 - matches Popen.wait
        return 0

    def kill(self):
        self.terminate()


class Harness:
    """One started engine plus what reached its player."""

    def __init__(self, engine, played, arrived, gate, players):
        self.engine = engine
        self.played = played
        self.arrived = arrived
        self.gate = gate
        self.players = players

    def wait_for_play(self, timeout=WAIT_SECONDS):
        return self.arrived.wait(timeout)

    def stayed_silent(self):
        """Report that nothing played, without waiting out the full timeout.

        Every stage between the queue and the player is a fake, so anything
        that was going to play has played long before this returns.
        """
        return not self.arrived.wait(1.5)

    def release(self):
        if self.gate is not None:
            self.gate.set()

    def close(self):
        self.release()
        self.engine.close()


def recorder_for(played, arrived):
    def record(audio):
        played.append(audio)
        arrived.set()

    return record


def fake_popen(played, arrived, gate, players):
    def popen(command, **kwargs):  # noqa: ARG001 - matches subprocess.Popen
        player = Player(recorder_for(played, arrived), gate)
        players.append(player)
        return player

    return popen


class FakeChunk:
    def __init__(self, audio):
        self.audio_int16_bytes = audio


class FakeVoice:
    """A loaded PiperVoice that yields audible PCM without onnxruntime."""

    def __init__(self):
        self.config = SimpleNamespace(sample_rate=SAMPLE_RATE)

    def synthesize(self, text):  # noqa: ARG002 - matches PiperVoice.synthesize
        yield FakeChunk(numpy.full(2205, 8000, dtype=numpy.int16).tobytes())


class FakeCommunicate:
    """Stand in for edge_tts.Communicate without a network call."""

    def __init__(self, text, voice_name):
        self.text = text
        self.voice_name = voice_name

    async def stream(self):
        yield {"type": "audio", "data": f"audio:{self.text}".encode()}


def _fake_processes(monkeypatch, played, arrived, gate, players):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", fake_popen(played, arrived, gate, players))


def build_edge(monkeypatch, gate):
    played: list[bytes] = []
    arrived = threading.Event()
    players: list[Player] = []
    _fake_processes(monkeypatch, played, arrived, gate, players)
    # ffmpeg trims through a completed subprocess rather than the player.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=kwargs["input"]),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural")
    return Harness(engine, played, arrived, gate, players)


def build_piper(monkeypatch, gate):
    played: list[bytes] = []
    arrived = threading.Event()
    players: list[Player] = []
    _fake_processes(monkeypatch, played, arrived, gate, players)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: FakeVoice())),
    )
    monkeypatch.setattr(
        "tagalong.piper_tts.ensure_model", lambda _voice, _home: Path("model.onnx")
    )
    engine = PiperSentenceTTS("en_US-lessac-medium")
    return Harness(engine, played, arrived, gate, players)


def build_switchable(monkeypatch, gate):
    """A session-level engine wrapping a real one, as the runtime holds it."""
    inner = build_piper(monkeypatch, gate)
    engine = SwitchableSpeech.start(
        "piper", build=lambda *_args, **_kwargs: inner.engine
    )
    return Harness(engine, inner.played, inner.arrived, gate, inner.players)


BUILDERS = {
    "edge": build_edge,
    "piper": build_piper,
    "switchable": build_switchable,
}


@pytest.fixture(params=sorted(BUILDERS))
def speech(request, monkeypatch):
    """A started engine of every kind, with nothing holding its player."""
    harness = BUILDERS[request.param](monkeypatch, None)
    yield harness
    harness.close()


@pytest.fixture(params=sorted(BUILDERS))
def held(request, monkeypatch):
    """A started engine whose player stays open until a test releases it."""
    harness = BUILDERS[request.param](monkeypatch, threading.Event())
    yield harness
    harness.close()


def wait_until(predicate):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------
# Speaking at all
# --------------------------------------------------------------------------


def test_a_sentence_spoken_in_a_turn_reaches_the_player(speech) -> None:
    speech.engine.begin_turn()
    speech.engine.speak("The deploy finished.")

    assert speech.wait_for_play()
    assert speech.played


def test_an_engine_falls_quiet_once_its_last_sentence_has_played(speech) -> None:
    speech.engine.begin_turn()
    speech.engine.speak("The deploy finished.")
    assert speech.wait_for_play()

    assert wait_until(lambda: speech.engine.is_speaking() is False)


# --------------------------------------------------------------------------
# Interruption reaches the audio, not just the queue
# --------------------------------------------------------------------------


def test_an_interrupt_stops_audio_that_is_already_playing(held) -> None:
    held.engine.begin_turn()
    held.engine.speak("The deploy finished.")
    assert held.wait_for_play()

    held.engine.interrupt()

    assert wait_until(lambda: held.players[0].terminated)


def test_an_interrupt_drops_sentences_that_had_not_started(held) -> None:
    held.engine.begin_turn()
    held.engine.speak("Holds the player open.")
    assert held.wait_for_play()
    held.engine.speak("Never reaches the player.")

    held.engine.interrupt()
    held.release()

    assert wait_until(lambda: held.engine.is_speaking() is False)
    assert len(held.played) == 1


# --------------------------------------------------------------------------
# Shutdown
# --------------------------------------------------------------------------


def test_closing_an_engine_stops_the_audio_it_was_playing(held) -> None:
    held.engine.begin_turn()
    held.engine.speak("Cut off by shutdown.")
    assert held.wait_for_play()

    held.engine.close()

    assert held.players[0].terminated


def test_speech_offered_after_close_is_refused(speech) -> None:
    speech.engine.begin_turn()
    speech.engine.close()

    speech.engine.speak("Arrived during shutdown.")

    assert speech.stayed_silent()


def test_closing_an_engine_twice_is_harmless(speech) -> None:
    """Shutdown runs from the interface and from the session teardown."""
    speech.engine.close()
    speech.engine.close()

    assert speech.engine.is_speaking() is False
