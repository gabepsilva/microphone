"""The player process both speech engines share.

``subprocess.Popen`` is faked; everything the class does around it is real.
"""

from __future__ import annotations

import subprocess

import pytest

from voice_codex.playback import (
    AudioPlayer,
    describe_tool_failure,
    play_command,
    player_environment,
    raw_pcm_args,
)


class FakePlayer:
    """Stand in for the ffplay process started by subprocess.Popen."""

    def __init__(
        self, returncode=0, stderr=b"", alive=False, on_communicate=None, on_wait=None
    ):
        self.returncode = returncode
        self.stderr = stderr
        self.alive = alive
        self.on_communicate = on_communicate
        self.on_wait = on_wait
        self.received = None
        self.terminated = False
        self.killed = False
        self.waits: list[float | None] = []

    def communicate(self, input=None):
        self.received = input
        if self.on_communicate is not None:
            self.on_communicate()
        return b"", self.stderr

    def poll(self):
        # Reports itself as still running once, so the terminate path can be
        # exercised without wedging the test.
        if self.alive:
            self.alive = False
            return None
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.on_wait is not None:
            self.on_wait(len(self.waits), timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class Spawns:
    """Record every player command and hand back a prepared fake."""

    def __init__(self, player=None):
        self.commands: list[list[str]] = []
        self.environments: list[dict] = []
        self.player = FakePlayer() if player is None else player

    def popen(self, command, **kwargs):
        self.commands.append(command)
        self.environments.append(kwargs["env"])
        return self.player


@pytest.fixture
def spawns(monkeypatch):
    recorder = Spawns()
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)
    return recorder


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (b"no such device", "\nffplay broke: no such device"),
        (b"  padded  ", "\nffplay broke: padded"),
        (b"", "\nffplay broke."),
        (None, "\nffplay broke."),
    ],
)
def test_a_failing_helper_process_is_explained_with_its_stderr(
    stderr, expected
) -> None:
    assert describe_tool_failure("ffplay broke", stderr) == expected


def test_the_play_command_reads_the_sentence_from_standard_input() -> None:
    command = play_command("/usr/bin/ffplay")

    assert command[0] == "/usr/bin/ffplay"
    assert command[command.index("-i") + 1] == "pipe:0"
    assert "-nodisp" in command
    assert "-autoexit" in command


def test_encoded_audio_is_passed_without_a_format_hint() -> None:
    assert "-f" not in play_command("/usr/bin/ffplay")


def test_raw_audio_is_described_so_the_player_need_not_guess() -> None:
    command = play_command("/usr/bin/ffplay", raw_pcm_args(22050))

    assert command[command.index("-f") + 1] == "s16le"
    assert command[command.index("-ar") + 1] == "22050"
    assert command[command.index("-ac") + 1] == "1"
    # The format has to be stated before the input it describes.
    assert command.index("-f") < command.index("-i")


def test_playback_is_routed_to_a_chosen_sink() -> None:
    environment = player_environment("alsa_output.pci", {"PATH": "/usr/bin"})

    assert environment["PULSE_SINK"] == "alsa_output.pci"
    assert environment["PATH"] == "/usr/bin"


def test_playback_uses_the_default_sink_when_none_was_chosen() -> None:
    environment = player_environment(None, {"PATH": "/usr/bin"})

    assert "PULSE_SINK" not in environment
    assert environment["PATH"] == "/usr/bin"


def test_audio_reaches_the_player(spawns) -> None:
    AudioPlayer("/usr/bin/ffplay").play(b"sentence")

    assert spawns.player.received == b"sentence"
    assert spawns.commands[0][0] == "/usr/bin/ffplay"


def test_the_chosen_sink_reaches_the_player(spawns) -> None:
    AudioPlayer("/usr/bin/ffplay", output_sink="meeting.sink").play(b"sentence")

    assert spawns.environments[0]["PULSE_SINK"] == "meeting.sink"


def test_the_declared_input_format_reaches_the_player(spawns) -> None:
    AudioPlayer("/usr/bin/ffplay", input_args=raw_pcm_args(16000)).play(b"sentence")

    command = spawns.commands[0]
    assert command[command.index("-ar") + 1] == "16000"


def test_a_player_that_exits_badly_is_reported(monkeypatch, capsys) -> None:
    recorder = Spawns(FakePlayer(returncode=3, stderr=b"no such device"))
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    AudioPlayer("/usr/bin/ffplay").play(b"sentence")

    assert "player exited with code 3: no such device" in capsys.readouterr().err


def test_a_player_stopped_on_purpose_is_not_reported_as_a_failure(
    monkeypatch, capsys
) -> None:
    recorder = Spawns(FakePlayer(returncode=-15))
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    AudioPlayer("/usr/bin/ffplay").play(b"sentence", abandoned=lambda: True)

    assert capsys.readouterr().err == ""


def test_a_broken_pipe_does_not_escape_playback(monkeypatch) -> None:
    def explode():
        raise BrokenPipeError

    recorder = Spawns(FakePlayer(on_communicate=explode))
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    AudioPlayer("/usr/bin/ffplay").play(b"sentence")

    # A player that exits mid-sentence breaks the pipe. Playback has to
    # survive it, and the process still has to be reaped rather than left.
    assert recorder.player.received == b"sentence"
    assert recorder.player.waits == []


def test_a_player_still_running_afterwards_is_terminated(monkeypatch) -> None:
    recorder = Spawns(FakePlayer(alive=True))
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    AudioPlayer("/usr/bin/ffplay").play(b"sentence")

    assert recorder.player.terminated is True
    assert recorder.player.waits == [AudioPlayer.TERMINATE_TIMEOUT_SECONDS]


def test_a_player_that_ignores_termination_is_killed(monkeypatch) -> None:
    def refuse_to_exit(attempt, timeout):
        if attempt == 1:
            raise subprocess.TimeoutExpired("ffplay", timeout or 0.0)

    player = FakePlayer(alive=True, on_wait=refuse_to_exit)
    recorder = Spawns(player)
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    AudioPlayer("/usr/bin/ffplay").play(b"sentence")

    assert player.killed is True


def test_stopping_cuts_off_the_sentence_being_played(monkeypatch) -> None:
    speaker = AudioPlayer("/usr/bin/ffplay")
    player = FakePlayer(on_communicate=speaker.stop)
    recorder = Spawns(player)
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    speaker.play(b"sentence")

    assert player.terminated is True


def test_stopping_with_nothing_playing_does_nothing(spawns) -> None:
    AudioPlayer("/usr/bin/ffplay").stop()

    assert spawns.commands == []


def test_a_finished_sentence_is_no_longer_stoppable(spawns) -> None:
    speaker = AudioPlayer("/usr/bin/ffplay")

    speaker.play(b"sentence")
    speaker.stop()

    assert spawns.player.terminated is False
