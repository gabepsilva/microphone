"""Provider selection and the switch that replaces one engine with another.

The engines themselves are faked here. What these cover is the boundary: that
each provider is built with its own default voice, and that a session can move
between them without the conversation holding a stale engine or losing the
device it plays through.
"""

from __future__ import annotations

import threading
import time

import pytest

from tagalong import piper_tts, tts
from tagalong.speech import (
    DEFAULT_PROVIDER,
    PROVIDER_LABELS,
    PROVIDERS,
    SwitchableSpeech,
    build_speech_engine,
    default_voice,
    provider_switch,
)

WAIT_SECONDS = 10


class FakeEngine:
    """Record the calls the runtime makes on whichever engine is installed."""

    def __init__(self, provider="fake", voice=None, output_sink=None):
        self.provider = provider
        self.voice = voice
        self.output_sink = output_sink
        self.spoken: list[str] = []
        self.turns = 0
        self.interrupts = 0
        self.enabled: list[bool] = []
        self.closed = False

    def begin_turn(self):
        self.turns += 1

    def speak(self, text):
        self.spoken.append(text)

    def interrupt(self):
        self.interrupts += 1

    def set_enabled(self, enabled):
        self.enabled.append(enabled)

    def is_likely_echo(self, text):
        return text == "echo"

    def is_speaking(self):
        return bool(self.spoken)

    def close(self):
        self.closed = True


def recording_builder(built=None, block=None):
    """Build fake engines, optionally holding the build open like a slow load."""
    built = [] if built is None else built

    def build(provider, voice, output_sink):
        if block is not None:
            block.wait(WAIT_SECONDS)
        engine = FakeEngine(provider, voice, output_sink)
        built.append(engine)
        return engine

    return build, built


def started(provider=DEFAULT_PROVIDER, output_sink=None, build=None):
    """A facade wrapping a fake engine, plus the list of engines built."""
    build, built = recording_builder() if build is None else (build, [])
    speech = SwitchableSpeech.start(provider, output_sink=output_sink, build=build)
    return speech, built


def wait_until(predicate):
    """Wait for the switch thread to land rather than for a fixed delay."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_every_provider_has_a_label_and_a_default_voice() -> None:
    assert set(PROVIDERS) == set(PROVIDER_LABELS)
    assert all(default_voice(name) for name in PROVIDERS)


def test_local_synthesis_is_the_default_provider() -> None:
    assert DEFAULT_PROVIDER == "piper"
    assert default_voice("piper") == "en_US-lessac-medium"


def test_an_unknown_provider_is_refused_by_name() -> None:
    with pytest.raises(RuntimeError, match="Unknown speech provider 'festival'"):
        build_speech_engine("festival")


@pytest.fixture
def engines(monkeypatch):
    """Replace both real engine classes, leaving the factory itself real."""
    built: list[FakeEngine] = []

    def engine_for(provider):
        def make(voice, output_sink=None):
            engine = FakeEngine(provider, voice, output_sink)
            built.append(engine)
            return engine

        return make

    monkeypatch.setattr(tts, "EdgeSentenceTTS", engine_for("edge"))
    monkeypatch.setattr(piper_tts, "PiperSentenceTTS", engine_for("piper"))
    return built


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_provider_is_built_with_its_own_default_voice(engines, provider) -> None:
    build_speech_engine(provider)

    assert len(engines) == 1
    assert engines[0].provider == provider
    assert engines[0].voice == default_voice(provider)


def test_a_chosen_voice_overrides_the_provider_default(engines) -> None:
    build_speech_engine("edge", "en-GB-RyanNeural")

    assert engines[0].voice == "en-GB-RyanNeural"


def test_the_chosen_output_reaches_the_engine(engines) -> None:
    build_speech_engine("piper", output_sink="meeting.sink")

    assert engines[0].output_sink == "meeting.sink"


def test_the_runtime_reaches_the_installed_engine() -> None:
    speech, _ = started()

    speech.begin_turn()
    speech.speak("Hello.")
    speech.interrupt()
    speech.set_enabled(False)

    assert speech.engine.turns == 1
    assert speech.engine.spoken == ["Hello."]
    assert speech.engine.interrupts == 1
    assert speech.engine.enabled == [False]
    assert speech.is_likely_echo("echo") is True
    assert speech.is_speaking() is True


def test_switching_replaces_the_engine_and_closes_the_old_one() -> None:
    speech, built = started("piper")

    assert speech.set_provider("edge") is True
    assert wait_until(lambda: speech.provider == "edge")

    assert [engine.provider for engine in built] == ["piper", "edge"]
    assert built[0].closed is True
    assert built[1].closed is False
    assert speech.engine is built[1]


def test_a_switch_speaks_with_the_new_provider_default_voice(engines) -> None:
    speech = SwitchableSpeech.start("piper")

    speech.set_provider("edge")

    assert wait_until(lambda: len(engines) == 2)
    assert engines[1].voice == default_voice("edge")


def test_a_switch_keeps_playing_through_the_session_output() -> None:
    speech, built = started("piper", output_sink="meeting.sink")

    speech.set_provider("edge")

    assert wait_until(lambda: len(built) == 2)
    assert built[1].output_sink == "meeting.sink"


def test_switching_to_the_provider_already_in_use_does_nothing() -> None:
    speech, built = started("piper")

    assert speech.set_provider("piper") is False
    assert built[0].closed is False
    assert len(built) == 1


def test_a_switch_already_running_is_not_joined_by_a_second() -> None:
    release = threading.Event()
    build, built = recording_builder(block=release)
    speech = SwitchableSpeech(DEFAULT_PROVIDER, FakeEngine("piper"), build=build)

    assert speech.set_provider("edge") is True
    assert speech.set_provider("edge") is False

    release.set()
    assert wait_until(lambda: speech.provider == "edge")
    assert len(built) == 1


def test_the_old_engine_keeps_speaking_until_the_new_one_is_ready() -> None:
    release = threading.Event()
    build, _ = recording_builder(block=release)
    original = FakeEngine("piper")
    speech = SwitchableSpeech(DEFAULT_PROVIDER, original, build=build)

    speech.set_provider("edge")
    speech.speak("Still the old engine.")

    assert original.spoken == ["Still the old engine."]
    release.set()
    assert wait_until(lambda: speech.provider == "edge")


def test_a_failed_switch_keeps_the_working_engine(capsys) -> None:
    def build(provider, voice, output_sink):  # noqa: ARG001 - matches the factory
        raise RuntimeError("no model")

    original = FakeEngine("piper")
    speech = SwitchableSpeech(DEFAULT_PROVIDER, original, build=build)

    speech.set_provider("edge")

    assert wait_until(lambda: "Could not switch speech" in capsys.readouterr().err)
    assert speech.provider == DEFAULT_PROVIDER
    assert speech.engine is original
    assert original.closed is False


def test_an_engine_built_after_the_session_closed_is_shut_down_too() -> None:
    """A switch that lands after ``close`` must not leave a live engine behind.

    The build is held open across the close so the two genuinely overlap;
    releasing it first would let the switch finish and test nothing.
    """
    release = threading.Event()
    build, built = recording_builder(block=release)
    original = FakeEngine("piper")
    speech = SwitchableSpeech(DEFAULT_PROVIDER, original, build=build)

    speech.set_provider("edge")
    closing = threading.Thread(target=speech.close, daemon=True)
    closing.start()
    release.set()
    closing.join(timeout=WAIT_SECONDS)

    assert original.closed is True
    assert wait_until(lambda: len(built) == 1 and built[0].closed is True)
    assert speech.provider == DEFAULT_PROVIDER
    assert speech.engine is original


def test_a_closed_session_refuses_to_switch() -> None:
    speech, _ = started()

    speech.close()

    assert speech.set_provider("edge") is False


def test_closing_shuts_the_installed_engine_down() -> None:
    speech, built = started()

    speech.close()

    assert built[0].closed is True


def test_the_interface_switch_reports_a_session_without_speech() -> None:
    assert provider_switch(None)("edge") is False


def test_the_interface_switch_drives_the_session_speech() -> None:
    speech, _ = started("piper")

    assert provider_switch(speech)("edge") is True
    assert wait_until(lambda: speech.provider == "edge")


def test_a_switch_reports_the_new_engine_speech_not_the_old(engines) -> None:
    """The sidebar must not keep describing an engine that has been retired."""
    speech = SwitchableSpeech.start("piper")
    speech.speak("From the first engine.")
    assert speech.is_speaking() is True

    speech.set_provider("edge")

    # Installed, not merely built: the old engine keeps answering until the
    # new one is swapped in, which is the whole point of the handover.
    assert wait_until(lambda: speech.provider == "edge")
    assert len(engines) == 2
    assert speech.is_speaking() is False
