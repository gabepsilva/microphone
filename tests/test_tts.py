"""Edge TTS synthesis, playback ordering, and interruption.

Edge's network client, ffmpeg, and ffplay are faked at their boundaries; the
real worker thread and asyncio pipeline run. Tests wait on an event the fake
player sets rather than on a delay, so they finish as fast as the pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from voice_codex.tts import (
    EdgeSentenceTTS,
    describe_tool_failure,
    player_environment,
    trim_command,
)

WAIT_SECONDS = 10


class FakeCommunicate:
    """Stand in for edge_tts.Communicate without a network call."""

    def __init__(self, text, voice_name):
        self.text = text
        self.voice_name = voice_name

    async def stream(self):
        yield {"type": "WordBoundary", "offset": 0}
        yield {"type": "audio", "data": f"audio:{self.text}".encode()}


class FakePlayer:
    """Stand in for the ffplay process started by subprocess.Popen."""

    def __init__(self, recorder, returncode=0, stderr=b""):
        self.recorder = recorder
        self.returncode = returncode
        self.stderr = stderr
        self.terminated = False

    def communicate(self, input=None):
        self.recorder(input)
        return b"", self.stderr

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):  # noqa: ARG002 - matches Popen.wait
        return self.returncode

    def kill(self):
        self.terminated = True


class Playback:
    """Collect what reached the player and signal when enough has arrived."""

    def __init__(self):
        self.audio: list[bytes] = []
        self.commands: list[list[str]] = []
        self.environments: list[dict | None] = []
        self.arrived = threading.Event()
        self.expected = 1
        self.returncode = 0
        self.stderr = b""

    def record(self, audio):
        self.audio.append(audio)
        if len(self.audio) >= self.expected:
            self.arrived.set()

    def popen(self, command, **kwargs):
        self.commands.append(command)
        self.environments.append(kwargs.get("env"))
        return FakePlayer(self.record, self.returncode, self.stderr)

    def wait_for(self, count):
        self.expected = count
        if len(self.audio) >= count:
            return True
        return self.arrived.wait(WAIT_SECONDS)


@pytest.fixture
def playback():
    return Playback()


@pytest.fixture
def tts(monkeypatch, playback):
    """A started Edge TTS pipeline with every external process faked."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=kwargs["input"] + b":trimmed"),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural")
    yield engine
    engine.close()


def test_edge_tts_requires_its_helper_binaries(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="requires ffplay"):
        EdgeSentenceTTS("en-US-AndrewNeural")


def test_edge_tts_requires_ffmpeg_for_silence_trimming(monkeypatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "ffmpeg" else "/usr/bin/x"
    )

    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        EdgeSentenceTTS("en-US-AndrewNeural")


def test_sentences_are_played_in_the_order_they_were_spoken(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("First sentence.")
    tts.speak("Second sentence.")
    tts.speak("Third sentence.")

    assert playback.wait_for(3)
    assert playback.audio == [
        b"audio:First sentence.:trimmed",
        b"audio:Second sentence.:trimmed",
        b"audio:Third sentence.:trimmed",
    ]


def test_silence_trimmed_audio_is_what_reaches_the_player(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("Hello there.")

    assert playback.wait_for(1)
    assert playback.audio == [b"audio:Hello there.:trimmed"]


def test_the_player_is_invoked_with_a_pipe_and_no_display(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("Hello there.")

    assert playback.wait_for(1)
    assert playback.commands[0] == [
        "/usr/bin/ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-i",
        "pipe:0",
    ]


def test_speech_is_dropped_while_tts_is_disabled(tts, playback) -> None:
    tts.begin_turn()
    tts.set_enabled(False)
    tts.speak("This must stay silent.")
    tts.set_enabled(True)
    tts.begin_turn()
    tts.speak("This one plays.")

    assert playback.wait_for(1)
    assert playback.audio == [b"audio:This one plays.:trimmed"]


def test_empty_text_is_never_queued(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("")
    tts.speak("Real speech.")

    assert playback.wait_for(1)
    assert playback.audio == [b"audio:Real speech.:trimmed"]


def test_an_interrupted_turn_is_no_longer_active(tts) -> None:
    tts.begin_turn()
    turn = tts.turns.current_turn
    tts.interrupt()

    assert not tts._turn_is_active(turn)


def test_speech_queued_after_an_interrupt_is_refused_until_a_new_turn(
    tts, playback
) -> None:
    tts.begin_turn()
    tts.interrupt()
    tts.speak("Ignored.")
    tts.begin_turn()
    tts.speak("Accepted.")

    assert playback.wait_for(1)
    assert playback.audio == [b"audio:Accepted.:trimmed"]


def test_queued_speech_is_remembered_as_a_likely_echo(tts) -> None:
    tts.begin_turn()
    tts.speak("Opening the settings panel.")

    assert tts.is_likely_echo("opening the settings panel")
    assert not tts.is_likely_echo("what time is the meeting")


def test_a_closed_engine_refuses_new_speech(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("Before closing.")

    assert playback.wait_for(1)

    tts.close()
    tts.speak("After closing.")

    assert playback.audio == [b"audio:Before closing.:trimmed"]


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (b"", "\nffplay broke."),
        (None, "\nffplay broke."),
        (b"   ", "\nffplay broke."),
        (b"codec not found", "\nffplay broke: codec not found"),
        (b"  spaced  ", "\nffplay broke: spaced"),
        (b"\xff\xfe bad bytes", "\nffplay broke: �� bad bytes"),
    ],
)
def test_a_failing_helper_process_is_explained_with_its_stderr(
    stderr, expected
) -> None:
    assert describe_tool_failure("ffplay broke", stderr) == expected


def test_the_trim_command_pipes_audio_through_the_silence_filter() -> None:
    command = trim_command("/usr/bin/ffmpeg", "silenceremove=x")

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-af") + 1] == "silenceremove=x"
    assert command[command.index("-i") + 1] == "pipe:0"
    assert command[-1] == "pipe:1"


def test_playback_is_routed_to_a_chosen_sink() -> None:
    environment = player_environment("alsa_output.pci", {"PATH": "/usr/bin"})

    assert environment == {"PATH": "/usr/bin", "PULSE_SINK": "alsa_output.pci"}


def test_playback_uses_the_default_sink_when_none_was_chosen() -> None:
    environment = player_environment(None, {"PATH": "/usr/bin"})

    assert environment == {"PATH": "/usr/bin"}


def test_the_chosen_sink_reaches_the_player(monkeypatch, playback) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=kwargs["input"]),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural", output_sink="meeting.sink")
    try:
        engine.begin_turn()
        engine.speak("Routed.")

        assert playback.wait_for(1)
        assert playback.environments[0]["PULSE_SINK"] == "meeting.sink"
    finally:
        engine.close()


def test_a_failing_trim_falls_back_to_the_untrimmed_audio(
    monkeypatch, playback, capsys
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)

    def failing_run(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"filter unavailable")

    monkeypatch.setattr(subprocess, "run", failing_run)
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural")
    try:
        engine.begin_turn()
        engine.speak("Untrimmed.")

        assert playback.wait_for(1)
        assert playback.audio == [b"audio:Untrimmed."]
        assert "silence trimming failed" in capsys.readouterr().err
    finally:
        engine.close()


def test_a_player_that_exits_badly_is_reported(monkeypatch, playback, capsys) -> None:
    playback.returncode = 3
    playback.stderr = b"no such device"
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=kwargs["input"]),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural")
    try:
        engine.begin_turn()
        engine.speak("Doomed.")

        assert playback.wait_for(1)
        engine.close()

        assert "player exited with code 3: no such device" in capsys.readouterr().err
    finally:
        engine.close()


def test_a_synthesis_failure_does_not_stop_later_sentences(
    monkeypatch, playback, capsys
) -> None:
    class ExplodingCommunicate(FakeCommunicate):
        async def stream(self):
            if "boom" in self.text:
                raise RuntimeError("edge refused the request")
            async for chunk in super().stream():
                yield chunk

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=kwargs["input"]),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=ExplodingCommunicate)
    )
    engine = EdgeSentenceTTS("en-US-AndrewNeural")
    try:
        engine.begin_turn()
        engine.speak("boom")
        engine.speak("Still speaking.")

        assert playback.wait_for(1)
        assert playback.audio == [b"audio:Still speaking."]
        assert "edge refused the request" in capsys.readouterr().err
    finally:
        engine.close()
