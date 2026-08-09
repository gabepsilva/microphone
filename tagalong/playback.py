#!/usr/bin/env python3
"""Play one sentence of synthesized audio, and stop on demand.

Both speech engines end the same way: hand a finished sentence to ffplay and
be able to cut it off mid-word when the user starts talking. Only the audio
differs — Edge produces MP3 that ffplay must parse, Piper produces raw PCM
whose format ffplay has to be told. That difference is a command-line
argument, so it lives in the caller and the process handling lives here.

Interruption is why this is a class rather than a function. The process must
be reachable from the thread that decides to stop it, which is never the
thread that is blocked writing audio into it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from contextlib import suppress

from .session import tagged_environment


def describe_tool_failure(headline, stderr):
    """Explain a failed helper process, quoting its stderr when it wrote any."""
    message = stderr.decode(errors="replace").strip() if stderr else ""
    return f"\n{headline}{': ' + message if message else '.'}"


def play_command(player, input_args=()):
    """Build the ffplay command that plays one synthesized sentence.

    ``input_args`` describes audio ffplay cannot identify on its own. Raw PCM
    carries no header, so its format, rate, and channel count have to be
    stated; MP3 carries all three and passes nothing.

    The log level is ``error`` rather than ``quiet`` because ``play`` quotes
    the player's stderr when it exits badly, and under ``quiet`` there is
    never anything to quote. A player that rejects one of these arguments
    then reports only an exit code, which says a sentence was lost without
    saying which argument lost it. Nothing is written on a normal sentence.
    """
    return [
        player,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        "pipe:0",
    ]


CHANNEL_LAYOUTS = {1: "mono", 2: "stereo"}


def channel_layout(channels):
    """Name a channel count the way ffmpeg's layout syntax expects.

    Named layouts are what a reader recognizes; ``6c`` is ffmpeg's spelling
    for a count with no standard name behind it.
    """
    return CHANNEL_LAYOUTS.get(channels, f"{channels}c")


def channel_args(channels):
    """State a channel count only when it is not the one every ffplay assumes.

    There is no spelling of this that every ffplay accepts. ``-ac`` was an
    ffplay option until ffmpeg 8 removed it; ``-ch_layout`` is a demuxer
    option the pcm demuxers only grew in 5.1. Passing either one blind breaks
    playback on the versions that predate or postdate it, and an ffplay that
    rejects an argument plays nothing at all.

    What both ends agree on is the default: the raw pcm demuxers have read
    mono since long before either option existed. Mono is what this program
    synthesizes, so saying nothing is both the compatible answer and the
    correct one. A caller that wants something else gets ``-ch_layout``, and
    needs an ffplay from 5.1 or later to be heard.
    """
    return () if channels == 1 else ("-ch_layout", channel_layout(channels))


def raw_pcm_args(sample_rate, channels=1):
    """Describe headerless signed 16-bit PCM so ffplay does not have to guess."""
    return (
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        *channel_args(channels),
    )


def player_environment(output_sink, base_environment=None):
    """Copy the environment, routing playback to a specific sink when given.

    The session tag rides along so the graph can tell this program's own voice
    from a player someone is running for their own reasons — including after a
    hard stop leaves one behind with init for a parent.
    """
    environment = tagged_environment(base_environment)
    if output_sink is not None:
        environment["PULSE_SINK"] = output_sink
    return environment


class AudioPlayer:
    """Own the one player process a speech engine speaks through."""

    TERMINATE_TIMEOUT_SECONDS = 3

    def __init__(self, player, output_sink=None, input_args=(), stream=None):
        self.player = player
        self.output_sink = output_sink
        self.input_args = tuple(input_args)
        # Resolved on use, not here: binding sys.stderr at import time writes
        # past anything that replaces the stream later, tests included.
        self._stream = stream
        self.lock = threading.Lock()
        self.active = None

    def play(self, audio, abandoned=lambda: False):
        """Play one sentence, reporting a player that failed on its own.

        ``abandoned`` is asked again after playback rather than before it
        only: a player that was terminated on purpose exits non-zero, and
        reporting that as an error would turn every interruption into noise.
        """
        process = subprocess.Popen(
            play_command(self.player, self.input_args),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=player_environment(self.output_sink),
        )
        with self.lock:
            self.active = process
        player_error = b""
        try:
            with suppress(BrokenPipeError):
                _, player_error = process.communicate(input=audio)
        finally:
            self._reap(process)
        if process.returncode and not abandoned():
            print(
                describe_tool_failure(
                    f"Speech player exited with code {process.returncode}",
                    player_error,
                ),
                file=sys.stderr if self._stream is None else self._stream,
                flush=True,
            )

    def _reap(self, process):
        """Make sure the process is gone and stop tracking it."""
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        with self.lock:
            if self.active is process:
                self.active = None

    def stop(self):
        """Cut off whatever is playing now, if anything is.

        A player that was arrested with SIGSTOP — by a pause, a tracing tool,
        or a frozen desktop — ignores SIGTERM while it is stopped. The signal
        is only delivered after SIGCONT, so termination first reanimates the
        process; otherwise the child lingers as a wedged zombie until it is
        killed or the session ends.
        """
        with self.lock:
            if self.active is not None:
                self._release_suspended_player(self.active)
                self.active.terminate()

    @staticmethod
    def _release_suspended_player(process):
        """Let a stopped player receive the termination that follows."""
        pid = getattr(process, "pid", None)
        if not hasattr(signal, "SIGCONT") or pid is None:
            return
        with suppress(OSError):
            os.kill(pid, signal.SIGCONT)
