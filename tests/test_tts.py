"""Edge TTS synthesis, playback ordering, and interruption.

Edge's network client, ffmpeg, and ffplay are faked at their boundaries; the
real worker thread and asyncio pipeline run. Tests wait on an event the fake
player sets rather than on a delay, so they finish as fast as the pipeline.
"""

from __future__ import annotations

import asyncio
import queue
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest

from tagalong.domain import EchoMatcher
from tagalong.playback import describe_tool_failure, player_environment
from tagalong.tts import EdgeSentenceTTS, trim_command

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
        # Set by tests that need to hold a player open or inspect it.
        self.factory = None
        self.players: list[FakePlayer] = []

    def record(self, audio):
        self.audio.append(audio)
        if len(self.audio) >= self.expected:
            self.arrived.set()

    def popen(self, command, **kwargs):
        self.commands.append(command)
        self.environments.append(kwargs.get("env"))
        if self.factory is not None:
            player = self.factory(self.record)
        else:
            player = FakePlayer(self.record, self.returncode, self.stderr)
        self.players.append(player)
        return player

    def wait_for(self, count):
        self.expected = count
        self.arrived.clear()
        if len(self.audio) >= count:
            return True
        return self.arrived.wait(WAIT_SECONDS)


class GatedPlayer(FakePlayer):
    """A player that holds the pipeline open until a test releases it.

    Playback occupies a prefetch slot, so holding one lets a test observe the
    producer blocked on the next slot rather than racing it.
    """

    def __init__(self, recorder, release, running, **kwargs):
        super().__init__(recorder, **kwargs)
        self.release = release
        self.running = running
        self.killed = False
        self.polls = 0

    def communicate(self, input=None):
        self.recorder(input)
        self.release.wait(WAIT_SECONDS)
        return b"", self.stderr

    def poll(self):
        # ``running`` reports the process as alive on the first poll only, so
        # a test can exercise the terminate path without wedging shutdown.
        self.polls += 1
        if self.running and self.polls == 1:
            return None
        return self.returncode

    def terminate(self):
        # A terminated ffplay exits, so the blocked write returns. Modelling
        # that is what lets a test assert shutdown actually completes.
        super().terminate()
        self.release.set()

    def kill(self):
        self.killed = True
        self.release.set()
        super().kill()


def start_engine(monkeypatch, playback, communicate=FakeCommunicate, **kwargs):
    """Build a started engine with every external process faked."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "Popen", playback.popen)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **run_kwargs: SimpleNamespace(stdout=run_kwargs["input"]),
    )
    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=communicate)
    )
    return EdgeSentenceTTS("en-US-AndrewNeural", **kwargs)


def wait_until(predicate):
    """Wait for a pipeline state rather than for a fixed delay."""
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
        "error",
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

    assert environment["PULSE_SINK"] == "alsa_output.pci"
    assert environment["PATH"] == "/usr/bin"


def test_playback_uses_the_default_sink_when_none_was_chosen() -> None:
    environment = player_environment(None, {"PATH": "/usr/bin"})

    assert "PULSE_SINK" not in environment
    assert environment["PATH"] == "/usr/bin"


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


# --------------------------------------------------------------------------
# Abort paths
#
# Every stage of the pipeline re-checks whether its turn is still wanted, and
# shutdown has to stay bounded no matter where a sentence had reached. These
# are the paths a cancelled or closing turn takes, asserted on what the
# listener would hear rather than on the branch that carried it there.
# --------------------------------------------------------------------------


def test_edge_tts_requires_the_edge_tts_package(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setitem(sys.modules, "edge_tts", None)

    with pytest.raises(RuntimeError, match="requires the edge-tts package"):
        EdgeSentenceTTS("en-US-AndrewNeural")


def test_an_interrupt_stops_the_sentence_that_is_playing(monkeypatch, playback) -> None:
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Playing now.")
        assert playback.wait_for(1)

        engine.interrupt()

        assert playback.players[0].terminated
    finally:
        release.set()
        engine.close()


def test_an_interrupt_after_close_leaves_the_shutdown_request_alone(
    monkeypatch, playback
) -> None:
    engine = start_engine(monkeypatch, playback)
    engine.begin_turn()
    engine.close()
    cancelled_before = engine.turns.cancelled

    engine.interrupt()

    # A closed engine is already torn down. interrupt() must return without
    # touching the turn state or re-entering the queue drain.
    assert engine.turns.cancelled == cancelled_before
    assert not engine.worker.is_alive()


def test_draining_an_interrupt_puts_the_shutdown_item_back(
    monkeypatch, playback
) -> None:
    monkeypatch.setattr(EdgeSentenceTTS, "PREFETCH_COUNT", 1)
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Holds the only slot.")
        engine.speak("Blocks the producer.")
        assert playback.wait_for(1)
        # With the producer parked on a prefetch slot nothing drains the queue,
        # so the test controls exactly what the interrupt walks over.
        assert wait_until(engine.sentences.empty)
        engine.sentences.put_nowait(engine.stop_item)
        engine.sentences.put_nowait((engine.turns.current_turn, "after the stop."))

        engine.interrupt()

        # The drain stops at the stop item and puts it back, so a shutdown
        # already in flight is never swallowed by an interrupt.
        remaining = []
        with suppress(queue.Empty):
            while True:
                remaining.append(engine.sentences.get_nowait())
        assert engine.stop_item in remaining
    finally:
        release.set()
        engine.close()


def test_an_interrupt_stops_treating_dropped_speech_as_an_echo(
    monkeypatch, playback
) -> None:
    """A sentence the interrupt threw away must not filter the microphones.

    Queued speech is remembered for two minutes because it may not be said
    for a while. One that is dropped is never said at all, so its retention
    has to fall back to a spoken sentence's — otherwise the speaker's own
    words are filtered as echo of a reply they cut off and never heard.
    """
    monkeypatch.setattr(EdgeSentenceTTS, "PREFETCH_COUNT", 1)
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Holds the only slot.")
        engine.speak("Parks the producer.")
        assert playback.wait_for(1)
        assert wait_until(engine.sentences.empty)
        engine.speak("I can roll it back if you would rather not ship today.")

        engine.interrupt()

        elapsed = EdgeSentenceTTS.SPOKEN_RETENTION_SECONDS + 1
        engine.echo._clock = lambda: time.monotonic() + elapsed
        assert (
            engine.is_likely_echo(
                "i can roll it back if you would rather not ship today"
            )
            is False
        )
    finally:
        release.set()
        engine.close()


def test_a_prefetched_sentence_released_unspoken_stops_being_an_echo(
    monkeypatch, playback
) -> None:
    """The prefetch-release path skips the consumer, so it must forget too.

    A sentence the producer is holding when the turn is cancelled never
    reaches the ``finally`` that shortens retention, and it is never spoken
    either — the same two-minute filter would sit on it.
    """
    monkeypatch.setattr(EdgeSentenceTTS, "PREFETCH_COUNT", 1)
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    # A clock the test moves by hand: the release happens on the pipeline's
    # own schedule, and a real one would let it re-stamp the deadline from a
    # "now" the test had already moved past.
    now = [0.0]
    engine.echo._clock = lambda: now[0]
    held = "Let us roll the whole release back to yesterday."
    try:
        engine.begin_turn()
        engine.speak("Holds the only slot.")
        engine.speak(held)
        assert playback.wait_for(1)
        # The producer is parked on the prefetch slot holding the second
        # sentence, so the interrupt's queue drain never sees it.
        assert wait_until(engine.sentences.empty)

        engine.interrupt()
        # Freeing the player gives the slot back, which is what wakes the
        # producer into the release path under test.
        release.set()

        spoken_window = EdgeSentenceTTS.SPOKEN_RETENTION_SECONDS

        def shortened():
            deadline = engine.echo._expiry.get(EchoMatcher.normalize(held), 0)
            return deadline <= now[0] + spoken_window

        assert wait_until(shortened)

        now[0] += spoken_window + 1
        assert engine.is_likely_echo(held) is False
    finally:
        release.set()
        engine.close()


def test_a_turn_cancelled_during_synthesis_never_reaches_the_player(
    monkeypatch, playback
) -> None:
    started = threading.Event()
    release = threading.Event()

    class GatedCommunicate(FakeCommunicate):
        """Stall mid-stream, but only for the sentence the test cancels."""

        async def stream(self):
            if "Cancelled" not in self.text:
                async for chunk in super().stream():
                    yield chunk
                return
            yield {"type": "audio", "data": b"partial:"}
            started.set()
            await asyncio.to_thread(release.wait, WAIT_SECONDS)
            yield {"type": "audio", "data": f"audio:{self.text}".encode()}

    engine = start_engine(monkeypatch, playback, communicate=GatedCommunicate)
    try:
        engine.begin_turn()
        engine.speak("Cancelled mid-flight.")
        assert started.wait(WAIT_SECONDS)

        engine.interrupt()
        release.set()

        engine.begin_turn()
        engine.speak("The next turn.")
        assert playback.wait_for(1)

        # Only the new turn was heard; the half-synthesized sentence was
        # abandoned rather than played late.
        assert playback.audio == [b"audio:The next turn."]
    finally:
        release.set()
        engine.close()


def test_a_sentence_that_synthesizes_to_no_audio_never_starts_a_player(
    monkeypatch, playback
) -> None:
    class SilentCommunicate(FakeCommunicate):
        async def stream(self):
            yield {"type": "WordBoundary", "offset": 0}
            if "silent" not in self.text:
                yield {"type": "audio", "data": f"audio:{self.text}".encode()}

    engine = start_engine(monkeypatch, playback, communicate=SilentCommunicate)
    try:
        engine.begin_turn()
        engine.speak("a silent one.")
        engine.speak("An audible one.")
        assert playback.wait_for(1)

        assert playback.audio == [b"audio:An audible one."]
        assert len(playback.commands) == 1
    finally:
        engine.close()


def test_a_player_still_running_when_playback_ends_is_terminated(
    monkeypatch, playback
) -> None:
    release = threading.Event()
    release.set()
    playback.factory = lambda record: GatedPlayer(record, release, running=True)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Outlives its audio.")
        assert playback.wait_for(1)
        assert wait_until(lambda: playback.players[0].terminated)

        # A player that has not exited on its own is stopped rather than left
        # holding the output sink.
        assert playback.players[0].terminated
    finally:
        engine.close()


def test_a_player_that_ignores_termination_is_killed(monkeypatch, playback) -> None:
    release = threading.Event()
    release.set()

    class StubbornPlayer(GatedPlayer):
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("ffplay", timeout)
            return self.returncode

    playback.factory = lambda record: StubbornPlayer(record, release, running=True)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Refuses to stop.")
        assert playback.wait_for(1)
        assert wait_until(lambda: playback.players[0].killed)

        assert playback.players[0].killed
    finally:
        engine.close()


def test_speech_waiting_for_a_prefetch_slot_is_dropped_when_its_turn_is_cancelled(
    monkeypatch, playback
) -> None:
    monkeypatch.setattr(EdgeSentenceTTS, "PREFETCH_COUNT", 1)
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Holds the only slot.")
        engine.speak("Waiting for a slot.")
        assert playback.wait_for(1)
        # The second sentence has left the queue and is blocked on the slot the
        # held player owns, which is the state the cancellation has to cover.
        assert wait_until(engine.sentences.empty)

        engine.interrupt()
        release.set()

        engine.begin_turn()
        engine.speak("A fresh turn.")
        assert playback.wait_for(2)

        assert playback.audio == [b"audio:Holds the only slot.", b"audio:A fresh turn."]
    finally:
        release.set()
        engine.close()


def test_closing_while_speech_waits_for_a_slot_shuts_the_pipeline_down(
    monkeypatch, playback
) -> None:
    monkeypatch.setattr(EdgeSentenceTTS, "PREFETCH_COUNT", 1)
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Holds the only slot.")
        engine.speak("Never synthesized.")
        assert playback.wait_for(1)
        assert wait_until(engine.sentences.empty)

        release.set()
        engine.close()

        # close() must return, and the worker with it, even though a sentence
        # was still waiting on a prefetch slot.
        assert not engine.worker.is_alive()
        assert playback.audio == [b"audio:Holds the only slot."]
    finally:
        release.set()
        engine.close()


def test_closing_terminates_a_player_that_is_still_speaking(
    monkeypatch, playback
) -> None:
    release = threading.Event()
    playback.factory = lambda record: GatedPlayer(record, release, running=False)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Interrupted by shutdown.")
        assert playback.wait_for(1)

        engine.close()

        assert playback.players[0].terminated
        assert not engine.worker.is_alive()
    finally:
        release.set()
        engine.close()


def test_a_turn_cancelled_during_the_request_stagger_is_dropped(
    monkeypatch, playback
) -> None:
    # The pipeline staggers Edge requests, and a turn can be cancelled while a
    # sentence waits out that delay. Widening the stagger makes the window
    # observable instead of a race.
    monkeypatch.setattr(EdgeSentenceTTS, "REQUEST_STAGGER_SECONDS", 1.0)
    engine = start_engine(monkeypatch, playback)
    try:
        engine.begin_turn()
        engine.speak("Sent immediately.")
        engine.speak("Still waiting out the stagger.")
        assert playback.wait_for(1)

        engine.interrupt()
        engine.begin_turn()
        engine.speak("A fresh turn.")

        assert playback.wait_for(2)
        assert playback.audio == [b"audio:Sent immediately.", b"audio:A fresh turn."]
    finally:
        engine.close()


def test_an_engine_with_nothing_to_say_is_not_speaking(tts) -> None:
    assert tts.is_speaking() is False


def test_an_engine_is_speaking_from_the_moment_a_sentence_is_accepted(tts) -> None:
    tts.begin_turn()
    tts.speak("Hello there.")

    assert tts.is_speaking() is True


def test_an_engine_falls_quiet_once_the_last_sentence_has_played(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("Hello there.")
    assert playback.wait_for(1)

    assert wait_until(lambda: tts.is_speaking() is False)


def test_an_interrupted_engine_stops_reporting_speech(tts, playback) -> None:
    tts.begin_turn()
    tts.speak("First.")
    tts.speak("Second.")
    assert playback.wait_for(1)

    tts.interrupt()

    assert tts.is_speaking() is False
