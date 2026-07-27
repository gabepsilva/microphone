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
import subprocess
import sys
import threading
from contextlib import suppress


def describe_tool_failure(headline, stderr):
    """Explain a failed helper process, quoting its stderr when it wrote any."""
    message = stderr.decode(errors="replace").strip() if stderr else ""
    return f"\n{headline}{': ' + message if message else '.'}"


def play_command(player, input_args=()):
    """Build the ffplay command that plays one synthesized sentence.

    ``input_args`` describes audio ffplay cannot identify on its own. Raw PCM
    carries no header, so its format, rate, and channel count have to be
    stated; MP3 carries all three and passes nothing.
    """
    return [
        player,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        *input_args,
        "-i",
        "pipe:0",
    ]


def raw_pcm_args(sample_rate, channels=1):
    """Describe headerless signed 16-bit PCM so ffplay does not have to guess."""
    return (
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
    )


def player_environment(output_sink, base_environment=None):
    """Copy the environment, routing playback to a specific sink when given."""
    environment = dict(os.environ if base_environment is None else base_environment)
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
        """Cut off whatever is playing now, if anything is."""
        with self.lock:
            if self.active is not None:
                self.active.terminate()
