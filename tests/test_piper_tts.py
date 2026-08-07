"""Local Piper synthesis, silence trimming, playback ordering, and shutdown.

Piper's model and ffplay are faked at their boundaries; the real worker thread
runs. Tests wait on an event the fake player sets rather than on a delay, so
they finish as fast as the pipeline does.
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

from tagalong.piper_tts import (
    SILENCE_THRESHOLD,
    PiperSentenceTTS,
    ensure_model,
    model_paths,
    trim_silence,
)

WAIT_SECONDS = 10
SAMPLE_RATE = 22050


def pcm(*sample_groups):
    """Build int16 PCM from runs of (amplitude, count)."""
    samples = numpy.concatenate(
        [
            numpy.full(count, amplitude, dtype=numpy.int16)
            for amplitude, count in sample_groups
        ]
    )
    return samples.tobytes()


def sample_count(audio):
    return len(audio) // 2


class FakeChunk:
    def __init__(self, audio):
        self.audio_int16_bytes = audio


SPEECH_SAMPLES = 100
PADDING_SAMPLES = 2205  # 0.1s, longer than either margin trimming keeps


class FakeVoice:
    """Stand in for a loaded PiperVoice without onnxruntime.

    The audio is padded with silence at both ends the way Piper's own output
    is, and arrives in two chunks, so the engine is seen to join the stream
    and then trim it.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, on_synthesize=None):
        self.config = SimpleNamespace(sample_rate=sample_rate)
        self.spoken: list[str] = []
        # Set by tests that need one sentence to fail synthesis.
        self.on_synthesize = on_synthesize

    def synthesize(self, text):
        self.spoken.append(text)
        if self.on_synthesize is not None:
            self.on_synthesize()
        silence = numpy.zeros(PADDING_SAMPLES, dtype=numpy.int16)
        speech = numpy.full(SPEECH_SAMPLES, SILENCE_THRESHOLD * 2, dtype=numpy.int16)
        yield FakeChunk(numpy.concatenate([silence, speech]).tobytes())
        yield FakeChunk(numpy.concatenate([speech, silence]).tobytes())


class FakePlayer:
    """Stand in for the ffplay process started by subprocess.Popen."""

    def __init__(self, recorder, returncode=0):
        self.recorder = recorder
        self.returncode = returncode
        self.terminated = False

    def communicate(self, input=None):
        self.recorder(input)
        return b"", b""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):  # noqa: ARG002 - matches Popen.wait
        return self.returncode

    def kill(self):
        self.terminated = True


class GatedPlayer(FakePlayer):
    """A player that stays open until a test releases it.

    Interruption and shutdown can only be observed against a player that is
    still running; the plain fake returns from ``communicate`` immediately and
    has already been reaped by the time either arrives.
    """

    def __init__(self, recorder, release):
        super().__init__(recorder)
        self.release = release

    def communicate(self, input=None):
        self.recorder(input)
        self.release.wait(WAIT_SECONDS)
        return b"", b""

    def terminate(self):
        # A terminated ffplay exits, so the blocked write returns. Modelling
        # that is what lets a test assert shutdown actually completes.
        super().terminate()
        self.release.set()


class Playback:
    """Collect what reached the player and signal when enough has arrived."""

    def __init__(self):
        self.audio: list[bytes] = []
        self.commands: list[list[str]] = []
        self.environments: list[dict | None] = []
        self.arrived = threading.Event()
        self.expected = 1
        self.players: list[FakePlayer] = []
        # Set by tests that need to hold a player open while they inspect it.
        self.gate = None

    def record(self, audio):
        self.audio.append(audio)
        if len(self.audio) >= self.expected:
            self.arrived.set()

    def popen(self, command, **kwargs):
        self.commands.append(command)
        self.environments.append(kwargs.get("env"))
        if self.gate is None:
            player = FakePlayer(self.record)
        else:
            player = GatedPlayer(self.record, self.gate)
        self.players.append(player)
        return player

    def wait_for(self, count, timeout=WAIT_SECONDS):
        self.expected = count
        self.arrived.clear()
        if len(self.audio) >= count:
            return True
        return self.arrived.wait(timeout)

    def stayed_silent(self):
        """Assert nothing played, without waiting out the full timeout.

        The pipeline is a fake generator feeding a fake player, so anything
        that was going to play has played long before this returns.
        """
        return not self.wait_for(1, timeout=1.5)


def wait_until(predicate):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def playback():
    return Playback()


@pytest.fixture
def voice():
    return FakeVoice()


def start_engine(monkeypatch, playback, voice, **kwargs):
    """Build a started engine with the model and the player faked."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: voice)),
    )
    monkeypatch.setattr(
        "tagalong.piper_tts.ensure_model", lambda _voice, _home: Path("model.onnx")
    )
    return PiperSentenceTTS("en_US-lessac-medium", **kwargs)


@pytest.fixture
def piper(monkeypatch, playback, voice):
    engine = start_engine(monkeypatch, playback, voice)
    yield engine
    engine.close()


# --------------------------------------------------------------------------
# Silence trimming
# --------------------------------------------------------------------------


def test_silence_is_cut_from_both_ends_of_a_sentence() -> None:
    quiet = SILENCE_THRESHOLD - 1
    loud = SILENCE_THRESHOLD * 2
    audio = pcm((quiet, 4410), (loud, 2205), (quiet, 4410))

    trimmed = trim_silence(audio, SAMPLE_RATE)

    # 0.02s of lead and 0.08s of tail are kept on purpose.
    assert sample_count(trimmed) == 2205 + int(0.02 * SAMPLE_RATE) + int(
        0.08 * SAMPLE_RATE
    )


def test_trimming_keeps_a_margin_rather_than_clipping_the_first_word() -> None:
    loud = SILENCE_THRESHOLD * 2
    audio = pcm((0, 4410), (loud, 100))

    trimmed = trim_silence(audio, SAMPLE_RATE)

    assert sample_count(trimmed) == 100 + int(0.02 * SAMPLE_RATE)


def test_a_margin_longer_than_the_silence_cannot_run_off_the_start() -> None:
    loud = SILENCE_THRESHOLD * 2
    audio = pcm((0, 10), (loud, 100))

    trimmed = trim_silence(audio, SAMPLE_RATE)

    assert sample_count(trimmed) == 110


def test_a_margin_longer_than_the_tail_cannot_run_off_the_end() -> None:
    loud = SILENCE_THRESHOLD * 2
    audio = pcm((loud, 100), (0, 10))

    trimmed = trim_silence(audio, SAMPLE_RATE)

    assert sample_count(trimmed) == 110


def test_a_sentence_that_is_silent_throughout_produces_nothing() -> None:
    assert trim_silence(pcm((0, 4410)), SAMPLE_RATE) == b""


def test_a_sample_exactly_at_the_threshold_counts_as_speech() -> None:
    audio = pcm((0, 100), (SILENCE_THRESHOLD, 1), (0, 100))

    assert trim_silence(audio, SAMPLE_RATE) != b""


def test_a_sample_just_below_the_threshold_counts_as_silence() -> None:
    audio = pcm((0, 100), (SILENCE_THRESHOLD - 1, 1), (0, 100))

    assert trim_silence(audio, SAMPLE_RATE) == b""


def test_negative_samples_are_as_loud_as_positive_ones() -> None:
    audio = pcm((0, 100), (-SILENCE_THRESHOLD * 2, 50), (0, 100))

    assert trim_silence(audio, SAMPLE_RATE) != b""


# --------------------------------------------------------------------------
# Model location and download
# --------------------------------------------------------------------------


def test_a_voice_names_both_the_model_and_its_config(tmp_path) -> None:
    model, config = model_paths("en_US-lessac-medium", tmp_path)

    assert model == tmp_path / "en_US-lessac-medium.onnx"
    assert config == tmp_path / "en_US-lessac-medium.onnx.json"


def test_an_already_downloaded_voice_is_not_downloaded_again(tmp_path) -> None:
    model, config = model_paths("voice", tmp_path)
    model.write_bytes(b"model")
    config.write_text("{}", encoding="utf-8")

    def download(*_args):
        raise AssertionError("downloaded a voice that was already present")

    assert ensure_model("voice", tmp_path, download=download) == model


def test_a_voice_missing_its_config_is_downloaded_again(tmp_path) -> None:
    model, config = model_paths("voice", tmp_path)
    model.write_bytes(b"model")
    downloads = []

    def download(voice, home):
        downloads.append((voice, home))
        config.write_text("{}", encoding="utf-8")

    ensure_model("voice", tmp_path, download=download, stream=None)

    assert downloads == [("voice", tmp_path)]


def test_a_missing_voice_is_downloaded_once(tmp_path, capsys) -> None:
    model, config = model_paths("voice", tmp_path)

    def download(_voice, _home):
        model.write_bytes(b"model")
        config.write_text("{}", encoding="utf-8")

    assert ensure_model("voice", tmp_path, download=download) == model
    assert "Downloading Piper voice voice" in capsys.readouterr().err


def test_a_download_that_fails_is_reported_with_the_voice_name(tmp_path) -> None:
    def download(_voice, _home):
        raise OSError("network down")

    with pytest.raises(RuntimeError, match="Could not download Piper voice 'voice'"):
        ensure_model("voice", tmp_path, download=download)


def test_a_download_that_produces_no_model_is_reported(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=r"downloaded without voice\.onnx"):
        ensure_model("voice", tmp_path, download=lambda *_: None)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


def test_a_missing_piper_package_explains_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "piper", None)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    with pytest.raises(RuntimeError, match="requires the piper-tts package"):
        PiperSentenceTTS("en_US-lessac-medium")


def test_a_missing_player_explains_which_package_provides_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=object()))
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires ffplay"):
        PiperSentenceTTS("en_US-lessac-medium")


def test_a_sentence_is_synthesized_and_played(piper, playback, voice) -> None:
    piper.begin_turn()
    piper.speak("Hello there.")

    assert playback.wait_for(1)
    assert voice.spoken == ["Hello there."]
    assert playback.audio[0]


def test_sentences_are_played_in_the_order_they_arrived(piper, playback, voice) -> None:
    piper.begin_turn()
    piper.speak("First.")
    piper.speak("Second.")
    piper.speak("Third.")

    assert playback.wait_for(3)
    assert voice.spoken == ["First.", "Second.", "Third."]


def test_the_player_is_told_the_raw_audio_format(piper, playback) -> None:
    piper.begin_turn()
    piper.speak("Hello there.")

    assert playback.wait_for(1)
    command = playback.commands[0]
    assert command[command.index("-f") + 1] == "s16le"
    assert command[command.index("-ar") + 1] == str(SAMPLE_RATE)
    # Mono is left unstated: no channel option spans every ffplay version.
    assert "-ac" not in command


def test_the_chosen_sink_reaches_the_player(monkeypatch, playback, voice) -> None:
    engine = start_engine(monkeypatch, playback, voice, output_sink="meeting.sink")
    try:
        engine.begin_turn()
        engine.speak("Hello there.")

        assert playback.wait_for(1)
        assert playback.environments[0]["PULSE_SINK"] == "meeting.sink"
    finally:
        engine.close()


def test_silence_is_trimmed_before_the_audio_reaches_the_player(
    piper, playback
) -> None:
    piper.begin_turn()
    piper.speak("Hello there.")

    assert playback.wait_for(1)
    played = sample_count(playback.audio[0])
    assert played == 2 * SPEECH_SAMPLES + int(0.02 * SAMPLE_RATE) + int(
        0.08 * SAMPLE_RATE
    )
    assert played < 2 * (PADDING_SAMPLES + SPEECH_SAMPLES)


def test_an_interrupted_turn_stops_the_player_and_drops_its_queue(
    monkeypatch, playback, voice
) -> None:
    playback.gate = threading.Event()
    piper = start_engine(monkeypatch, playback, voice)
    piper.begin_turn()
    piper.speak("First.")
    assert playback.wait_for(1)

    piper.speak("Second.")
    piper.interrupt()

    assert wait_until(lambda: playback.players[0].terminated)
    assert len(playback.audio) == 1
    piper.close()


def test_disabling_speech_silences_the_current_response(piper, playback, voice) -> None:
    piper.begin_turn()
    piper.set_enabled(False)
    piper.speak("Silenced.")

    assert playback.stayed_silent()
    assert voice.spoken == []


def test_re_enabling_speech_lets_the_next_turn_speak(piper, playback) -> None:
    piper.set_enabled(False)
    piper.set_enabled(True)
    piper.begin_turn()
    piper.speak("Audible again.")

    assert playback.wait_for(1)


def test_a_model_that_fails_to_load_is_reported_once(
    monkeypatch, playback, capsys
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: None)),
    )
    monkeypatch.setattr(
        "tagalong.piper_tts.ensure_model",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    engine = PiperSentenceTTS("en_US-lessac-medium")
    try:
        engine.begin_turn()
        engine.speak("First.")
        engine.speak("Second.")

        assert wait_until(lambda: engine.model_error is None)
        assert engine.model_ready.is_set()
        assert playback.audio == []
        assert capsys.readouterr().err.count("Piper TTS error: no model") == 1
    finally:
        engine.close()


def test_a_synthesis_failure_is_reported_without_stopping_the_worker(
    monkeypatch, playback, capsys
) -> None:
    def explode():
        raise RuntimeError("decoder failed")

    voice = FakeVoice(on_synthesize=explode)
    engine = start_engine(monkeypatch, playback, voice)
    try:
        engine.begin_turn()
        engine.speak("Breaks.")

        assert wait_until(lambda: "decoder failed" in capsys.readouterr().err)
        voice.on_synthesize = None
        engine.speak("Still works.")

        assert playback.wait_for(1)
    finally:
        engine.close()


def test_a_session_closed_while_the_model_loads_never_speaks(
    monkeypatch, playback, voice
) -> None:
    """Shutdown has to reach a worker parked on the load, not just on the queue.

    Piper alone can be asked to stop in this state: the worker waits on the
    model for up to two minutes, so a sentence accepted before the close is
    inside ``_await_model`` rather than inside ``sentences.get`` when the stop
    item arrives. Nothing may be synthesized or played once it comes back.
    """
    loaded = threading.Event()
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: voice)),
    )
    monkeypatch.setattr(
        "tagalong.piper_tts.ensure_model",
        lambda _voice, _home: (loaded.wait(WAIT_SECONDS), Path("model.onnx"))[1],
    )
    engine = PiperSentenceTTS("en_US-lessac-medium")
    engine.begin_turn()
    engine.speak("Queued while the model was still loading.")
    assert wait_until(lambda: not engine.model_ready.is_set())
    # Released from the side so the close below joins its worker rather than
    # waiting out the timeout; the sentence is already past the gate by then.
    threading.Timer(0.05, loaded.set).start()

    engine.close()

    assert engine.worker.is_alive() is False
    assert playback.stayed_silent()
    assert voice.spoken == []


def test_speech_spans_the_gap_between_two_sentences(monkeypatch, playback) -> None:
    """A poll landing between sentences must not read as silence."""
    playback.gate = threading.Event()
    piper = start_engine(monkeypatch, playback, FakeVoice())
    try:
        piper.begin_turn()
        piper.speak("First.")
        piper.speak("Second.")
        assert playback.wait_for(1)

        assert piper.is_speaking() is True
    finally:
        piper.close()


def test_wait_ready_returns_once_the_model_has_loaded(piper) -> None:
    piper.wait_ready(timeout=WAIT_SECONDS)
    assert piper.model is not None


def test_wait_ready_raises_when_the_model_fails_to_load(monkeypatch, playback) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: None)),
    )
    monkeypatch.setattr(
        "tagalong.piper_tts.ensure_model",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    engine = PiperSentenceTTS("en_US-lessac-medium")
    try:
        with pytest.raises(RuntimeError, match="failed to load"):
            engine.wait_ready(timeout=WAIT_SECONDS)
        assert engine.model_error is not None
    finally:
        engine.close()


def test_wait_ready_times_out_when_the_loader_never_finishes(
    monkeypatch, playback
) -> None:
    release = threading.Event()

    def hang_load(_self):
        release.wait(WAIT_SECONDS)

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: None)),
    )
    monkeypatch.setattr(PiperSentenceTTS, "_load_model", hang_load)
    engine = PiperSentenceTTS("en_US-lessac-medium")
    try:
        with pytest.raises(RuntimeError, match="did not become ready"):
            engine.wait_ready(timeout=0.05)
    finally:
        release.set()
        engine.close()


def test_wait_ready_raises_when_ready_without_a_model_or_error(
    monkeypatch, playback
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: None)),
    )

    def clear_load(self):
        self.model = None
        self.model_error = None
        self.model_ready.set()

    monkeypatch.setattr(PiperSentenceTTS, "_load_model", clear_load)
    engine = PiperSentenceTTS("en_US-lessac-medium")
    try:
        with pytest.raises(RuntimeError, match="failed to load"):
            engine.wait_ready(timeout=WAIT_SECONDS)
    finally:
        engine.close()
