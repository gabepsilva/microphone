"""The player process both speech engines share.

``subprocess.Popen`` is faked; everything the class does around it is real.
"""

from __future__ import annotations

import signal
import subprocess
import threading

import pytest

from tagalong.playback import (
    AudioPlayer,
    channel_args,
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
        self.sent_signals: list[int] = []
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

    def send_signal(self, sig):
        self.sent_signals.append(sig)

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
    # The format has to be stated before the input it describes.
    assert command.index("-f") < command.index("-i")


def test_mono_is_left_to_the_default_every_ffplay_shares() -> None:
    # -ac was dropped in ffmpeg 8 and -ch_layout arrived in 5.1; an ffplay
    # that rejects either one plays nothing, and mono is the default in all.
    command = play_command("/usr/bin/ffplay", raw_pcm_args(22050))

    assert "-ac" not in command
    assert "-ch_layout" not in command


def test_a_count_that_is_not_the_default_is_stated() -> None:
    assert channel_args(2) == ("-ch_layout", "stereo")
    assert channel_args(6) == ("-ch_layout", "6c")
    assert channel_args(1) == ()


def test_the_player_may_explain_itself_when_it_fails() -> None:
    # 'quiet' would leave play() with an exit code and no reason to quote.
    command = play_command("/usr/bin/ffplay")

    assert command[command.index("-loglevel") + 1] == "error"


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


def test_a_reap_does_not_clear_a_process_it_never_set(monkeypatch) -> None:
    """One player winding down must not retire a newer occupant's slot.

    The reap only ever excuses the process that took the slot; a previous
    player finishing underneath a new one leaves the new one untouched. The
    first play is still draining the pipe when the second claims the slot,
    so the reap of the first runs against a slot that is no longer its own.
    """
    release = threading.Event()
    first = FakePlayer(on_communicate=release.wait)
    second = FakePlayer(alive=True)
    recorder = Spawns(first)
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)
    speaker = AudioPlayer("/usr/bin/ffplay")

    drained = threading.Thread(target=speaker.play, args=(b"first",), daemon=True)
    drained.start()
    while first.received is None:
        assert drained.is_alive() is True
        threading.Event().wait(0.01)

    recorder.player = second
    speaker.play(b"second")
    release.set()
    drained.join(timeout=2)

    assert speaker.active is None
    assert first.terminated is False
    assert second.terminated is True


def test_stop_without_sigcont_never_signals_the_player(monkeypatch) -> None:
    """A platform without SIGCONT gets the terminate and nothing else."""
    monkeypatch.delattr(signal, "SIGCONT", raising=False)
    speaker = AudioPlayer("/usr/bin/ffplay")
    player = FakePlayer(on_communicate=speaker.stop)
    recorder = Spawns(player)
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)

    speaker.play(b"sentence")

    assert player.sent_signals == []
    assert player.terminated is True


class StoppedPlayer(FakePlayer):
    """A player frozen with SIGSTOP: ignores stdin and SIGTERM until SIGCONT.

    A stopped process does not read its input pipe, so the writes ``play``
    pushes into it back up and the call blocks; SIGTERM delivered to a
    stopped process stays pending until SIGCONT arrives, so ``stop`` alone
    cannot end the sentence. The player comes back to life — and reports the
    signals it finally received — only when something sends SIGCONT.
    """

    def __init__(self, pid=4242):
        super().__init__(alive=True)
        self.pid = pid
        self.continued = False
        self.sent: list[tuple[int, int]] = []
        self.wake = threading.Event()

    def send_signal(self, sig):
        self.sent.append((self.pid, sig))
        self.continued = True
        self.wake.set()

    def communicate(self, input=None):
        # The stopped player never consumed the pipe; only the SIGCONT that
        # resumes it lets the write and the wait complete.
        del input
        self.wake.wait(timeout=5)
        return b"", b""

    def poll(self):
        return None if not self.wake.is_set() else 0

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return 0 if self.wake.is_set() else None


def test_stop_unblocks_play_on_a_player_stopped_with_sigstop(
    monkeypatch,
) -> None:
    """SIGCONT before termination, or play() never returns (a real wedge)."""
    stopped = StoppedPlayer()
    recorder = Spawns(stopped)
    monkeypatch.setattr(subprocess, "Popen", recorder.popen)
    speaker = AudioPlayer("/usr/bin/ffplay")
    finished = threading.Event()
    errors = []

    def play_background():
        try:
            speaker.play(b"sentence")
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=play_background, daemon=True)
    thread.start()
    # Give the worker time to reach the blocking communicate call, then cut
    # it off the way a user pressing stop against a frozen player would.
    finished.wait(timeout=0.2)
    assert not finished.is_set()
    speaker.stop()
    assert finished.wait(timeout=2) is True
    thread.join(timeout=2)
    assert errors == []
    assert stopped.continued is True
    assert stopped.sent == [(stopped.pid, signal.SIGCONT)]
