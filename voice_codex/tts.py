#!/usr/bin/env python3
"""Speak Codex responses through Edge TTS, one sentence at a time.

The worker thread here follows the package's shutdown contract: it is a
daemon, and ``close`` joins it with a timeout. ``tools/worker_gate.py``
enforces both.
"""

from __future__ import annotations

import asyncio
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress

from .domain import EchoMemory, TurnGate


def describe_tool_failure(headline, stderr):
    """Explain a failed helper process, quoting its stderr when it wrote any."""
    message = stderr.decode(errors="replace").strip() if stderr else ""
    return f"\n{headline}{': ' + message if message else '.'}"


def trim_command(trimmer, silence_filter):
    """Build the ffmpeg command that trims leading and trailing silence."""
    return [
        trimmer,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-af",
        silence_filter,
        "-f",
        "wav",
        "pipe:1",
    ]


def play_command(player):
    """Build the ffplay command that plays one synthesized sentence."""
    return [player, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"]


def player_environment(output_sink, base_environment=None):
    """Copy the environment, routing playback to a specific sink when given."""
    environment = dict(os.environ if base_environment is None else base_environment)
    if output_sink is not None:
        environment["PULSE_SINK"] = output_sink
    return environment


class EdgeSentenceTTS:
    """Prefetch two Edge sentences and play them in their original order."""

    PREFETCH_COUNT = 2
    REQUEST_STAGGER_SECONDS = 0.1
    # A sentence stays recognizable as an echo for as long as it could still
    # be heard: longest while it waits in the queue, shortest once it has been
    # played and only room reverberation can still bring it back.
    QUEUED_RETENTION_SECONDS = 120
    SYNTHESIZING_RETENTION_SECONDS = 30
    SPOKEN_RETENTION_SECONDS = 12
    SILENCE_TRIM_FILTER = (
        "silenceremove="
        "start_periods=1:start_duration=0.02:start_threshold=-45dB:"
        "start_silence=0.08,"
        "areverse,"
        "silenceremove="
        "start_periods=1:start_duration=0.02:start_threshold=-45dB:"
        "start_silence=0.22,"
        "areverse"
    )

    def __init__(self, voice, output_sink=None):
        try:
            import edge_tts
        except ImportError as error:
            raise RuntimeError(
                "Edge TTS audio requires the edge-tts package. Install the "
                "locked project dependencies with 'uv sync --locked'."
            ) from error
        player = shutil.which("ffplay")
        if player is None:
            raise RuntimeError(
                "Edge TTS audio requires ffplay. Install the ffmpeg package."
            )
        trimmer = shutil.which("ffmpeg")
        if trimmer is None:
            raise RuntimeError(
                "Edge TTS audio requires ffmpeg. Install the ffmpeg package."
            )

        self.edge_tts = edge_tts
        self.voice = voice
        self.player = player
        self.trimmer = trimmer
        self.output_sink = output_sink
        self.sentences = queue.Queue()
        self.stop_item = object()
        self.shutdown_requested = threading.Event()
        self.turns = TurnGate()
        self.echo = EchoMemory()
        self.player_lock = threading.Lock()
        self.active_player = None
        self.worker = threading.Thread(
            target=self._worker,
            name="EdgeTTSWorker",
            daemon=True,
        )
        self.worker.start()

    def begin_turn(self):
        self.turns.begin_turn()

    def set_enabled(self, enabled):
        """Enable or silence future speech without rebuilding the pipeline."""
        if self.turns.set_enabled(enabled):
            self.interrupt()

    def _turn_is_active(self, turn):
        return self.turns.is_active(turn)

    def _abandoned(self, turn):
        """Report whether a turn's speech is no longer wanted.

        Synthesis, trimming, and playback each hand off to a thread or an
        event loop, so a turn can be interrupted or the engine closed between
        any two of them. Every stage re-asks rather than trusting the answer
        the stage before it got.
        """
        return self.shutdown_requested.is_set() or not self._turn_is_active(turn)

    def speak(self, text):
        turn, accepting = self.turns.accepting_turn()
        if text and accepting and not self.shutdown_requested.is_set():
            self.echo.remember(text, retention=self.QUEUED_RETENTION_SECONDS)
            self.sentences.put_nowait((turn, text))

    def interrupt(self):
        """Stop the current response and discard all of its queued speech."""
        if self.shutdown_requested.is_set():
            return
        self.turns.cancel()

        while True:
            try:
                queued = self.sentences.get_nowait()
            except queue.Empty:
                break
            if queued is self.stop_item:
                self.sentences.put_nowait(queued)
                break

        with self.player_lock:
            if self.active_player is not None:
                self.active_player.terminate()

    def is_likely_echo(self, text):
        """Return True when a transcript resembles recently queued TTS."""
        return self.echo.matches(text)

    async def _synthesize(self, turn, text):
        self.echo.remember(
            text, retention=self.SYNTHESIZING_RETENTION_SECONDS, replace=True
        )
        communicate = self.edge_tts.Communicate(text, self.voice)
        audio = bytearray()
        async for chunk in communicate.stream():
            if self._abandoned(turn):
                break
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        if not audio or self._abandoned(turn):
            return bytes(audio)
        return await asyncio.to_thread(self._trim_silence, bytes(audio))

    def _trim_silence(self, audio):
        try:
            result = subprocess.run(
                trim_command(self.trimmer, self.SILENCE_TRIM_FILTER),
                input=audio,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print(
                describe_tool_failure(
                    "Edge TTS silence trimming failed; playing original audio",
                    error.stderr,
                ),
                file=sys.stderr,
                flush=True,
            )
            return audio
        return result.stdout or audio

    def _play(self, turn, audio):
        if not audio or self._abandoned(turn):
            return
        process = subprocess.Popen(
            play_command(self.player),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=player_environment(self.output_sink),
        )
        with self.player_lock:
            self.active_player = process
        player_error = b""
        try:
            with suppress(BrokenPipeError):
                _, player_error = process.communicate(input=audio)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            with self.player_lock:
                if self.active_player is process:
                    self.active_player = None
        if process.returncode and not self._abandoned(turn):
            print(
                describe_tool_failure(
                    f"Edge TTS player exited with code {process.returncode}",
                    player_error,
                ),
                file=sys.stderr,
                flush=True,
            )

    def _release_if_abandoned(self, turn, available_slots):
        """Give back a prefetch slot held for speech nobody wants any more."""
        if not self._abandoned(turn):
            return False
        available_slots.release()
        return True

    async def _produce_synthesis_jobs(self, jobs, available_slots):
        last_request_started = None
        while True:
            item = await asyncio.to_thread(self.sentences.get)
            if item is self.stop_item:
                break
            turn, text = item

            await available_slots.acquire()
            # Checked twice around the stagger: acquiring a slot and waiting
            # out the delay can each span an interrupt or a close.
            if self._release_if_abandoned(turn, available_slots):
                if self.shutdown_requested.is_set():
                    break
                continue

            if last_request_started is not None:
                elapsed = time.monotonic() - last_request_started
                delay = self.REQUEST_STAGGER_SECONDS - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)

            if self._release_if_abandoned(turn, available_slots):
                if self.shutdown_requested.is_set():
                    break
                continue

            last_request_started = time.monotonic()
            synthesis = asyncio.create_task(self._synthesize(turn, text))
            await jobs.put((turn, text, synthesis))

        await jobs.put(self.stop_item)

    async def _consume_synthesis_jobs(self, jobs, available_slots):
        while True:
            job = await jobs.get()
            if job is self.stop_item:
                return

            turn, text, synthesis = job
            try:
                audio = await synthesis
                if not self._abandoned(turn):
                    await asyncio.to_thread(self._play, turn, audio)
            except Exception as error:
                if not self.shutdown_requested.is_set():
                    print(
                        f"\nEdge TTS error: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                self.echo.remember(
                    text, retention=self.SPOKEN_RETENTION_SECONDS, replace=True
                )
                available_slots.release()

    async def _run_pipeline(self):
        jobs = asyncio.Queue()
        available_slots = asyncio.Semaphore(self.PREFETCH_COUNT)
        producer = asyncio.create_task(
            self._produce_synthesis_jobs(jobs, available_slots)
        )
        try:
            await self._consume_synthesis_jobs(jobs, available_slots)
            await producer
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    def _worker(self):
        asyncio.run(self._run_pipeline())

    def close(self):
        self.shutdown_requested.set()
        with self.player_lock:
            if self.active_player is not None:
                self.active_player.terminate()
        self.sentences.put_nowait(self.stop_item)
        self.worker.join(timeout=3)
