#!/usr/bin/env python3
"""Speak Taga's responses through Edge TTS, one sentence at a time.

The worker thread here follows the package's shutdown contract: it is a
daemon, and ``close`` joins it with a timeout. ``tools/worker_gate.py``
enforces both.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import threading
import time

from .playback import AudioPlayer, describe_tool_failure
from .queued_tts import QueuedSentenceTTS


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


class EdgeSentenceTTS(QueuedSentenceTTS):
    """Prefetch two Edge sentences and play them in their original order."""

    PREFETCH_COUNT = 2
    REQUEST_STAGGER_SECONDS = 0.1
    # A sentence stays recognizable as an echo for as long as it could still
    # be heard: longest while it waits in the queue, shortest once it has been
    # played and only room reverberation can still bring it back.
    SYNTHESIZING_RETENTION_SECONDS = 30
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
        self.trimmer = trimmer
        super().__init__(AudioPlayer(player, output_sink=output_sink))
        self.worker = threading.Thread(
            target=self._worker,
            name="EdgeTTSWorker",
            daemon=True,
        )
        self.worker.start()

    def wait_ready(self, timeout: float | None = None) -> None:
        """Edge has no local model load; a built engine is already ready."""
        del timeout

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
        self.playback.play(audio, abandoned=lambda: self._abandoned(turn))

    def _release_if_abandoned(self, turn, text, available_slots):
        """Give back a prefetch slot held for speech nobody wants any more.

        The sentence is un-remembered on the way out for the same reason the
        interrupt drain does it: this path skips the consumer's ``finally``,
        so its long queued retention would otherwise never be shortened.
        """
        if not self._abandoned(turn):
            return False
        self._forget_unspoken(text)
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
            if self._release_if_abandoned(turn, text, available_slots):
                if self.shutdown_requested.is_set():
                    break
                continue

            if last_request_started is not None:
                elapsed = time.monotonic() - last_request_started
                delay = self.REQUEST_STAGGER_SECONDS - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)

            if self._release_if_abandoned(turn, text, available_slots):
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
                self.activity.finished()
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
