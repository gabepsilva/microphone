#!/usr/bin/env python3
"""Speak Taga's responses through Piper, locally, one sentence at a time.

Nothing here prefetches. Piper synthesizes a sentence about thirty times
faster than the sentence takes to say, so the queue is never the thing the
listener waits on and a second sentence rendered early would only sit in
memory. The Edge engine's prefetch depth, request stagger, and asyncio
pipeline all exist to hide a network round trip that this engine does not
make.

The silence Piper leaves around a sentence is trimmed from the samples in
memory. Edge has to shell out to ffmpeg for that because it receives encoded
MP3; Piper hands back raw PCM, so the same job is an array slice and stays off
the path to the first word.

The worker thread here follows the package's shutdown contract: it is a
daemon, and ``close`` joins it with a timeout. ``tools/worker_gate.py``
enforces both.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

import numpy

from .playback import AudioPlayer, raw_pcm_args
from .queued_tts import QueuedSentenceTTS

DEFAULT_MODEL_HOME = (
    Path(
        os.environ.get(
            "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
        )
    )
    / "tagalong/piper"
)

# -45 dBFS, the same floor the Edge engine's ffmpeg filter uses, expressed
# where this engine can apply it: a signed 16-bit sample magnitude.
SILENCE_THRESHOLD = int(32768 * (10 ** (-45 / 20)))


def model_paths(voice, home=DEFAULT_MODEL_HOME):
    """Locate the model and its config for one Piper voice."""
    return Path(home) / f"{voice}.onnx", Path(home) / f"{voice}.onnx.json"


def ensure_model(voice, home=DEFAULT_MODEL_HOME, download=None, stream=None):
    """Return the model path for a voice, downloading it the first time.

    Both files are required: a model whose config is missing loads and then
    fails on the first sentence, which would surface the problem long after
    the moment that could explain it.
    """
    model, config = model_paths(voice, home)
    if model.exists() and config.exists():
        return model
    if download is None:
        from piper.download_voices import download_voice as download

    Path(home).mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading Piper voice {voice}...",
        file=sys.stderr if stream is None else stream,
        flush=True,
    )
    try:
        download(voice, Path(home))
    except Exception as error:
        raise RuntimeError(
            f"Could not download Piper voice {voice!r}: {error}"
        ) from error
    if not model.exists():
        raise RuntimeError(f"Piper voice {voice!r} downloaded without {model.name}.")
    return model


def trim_silence(pcm, sample_rate, keep_lead=0.02, keep_tail=0.08):
    """Cut the silence around a sentence, keeping a short margin either side.

    The margins are what stop consecutive sentences running together. Cutting
    to the exact first loud sample clips the attack of a plosive, and cutting
    the tail exactly makes the next sentence sound like it interrupted this
    one.
    """
    samples = numpy.frombuffer(pcm, dtype=numpy.int16)
    loud = numpy.flatnonzero(numpy.abs(samples) >= SILENCE_THRESHOLD)
    if loud.size == 0:
        return b""
    start = max(0, int(loud[0]) - int(keep_lead * sample_rate))
    end = min(samples.size, int(loud[-1]) + 1 + int(keep_tail * sample_rate))
    return samples[start:end].tobytes()


class PiperSentenceTTS(QueuedSentenceTTS):
    """Synthesize each sentence locally and play it in the order it arrived."""

    LOAD_TIMEOUT_SECONDS = 120

    def __init__(self, voice, output_sink=None, home=DEFAULT_MODEL_HOME):
        try:
            import piper
        except ImportError as error:
            raise RuntimeError(
                "Piper TTS audio requires the piper-tts package. Install the "
                "locked project dependencies with 'uv sync --locked'."
            ) from error
        player = shutil.which("ffplay")
        if player is None:
            raise RuntimeError(
                "Piper TTS audio requires ffplay. Install the ffmpeg package."
            )

        self.piper = piper
        self.voice = voice
        self.home = home
        super().__init__(AudioPlayer(player, output_sink=output_sink))
        self.model = None
        self.model_error = None
        self.model_ready = threading.Event()
        # Loading is roughly a second, and it overlaps the transcription model
        # the session is already waiting on. Starting it here rather than on
        # the first sentence keeps that cost out of the first reply.
        self.loader = threading.Thread(
            target=self._load_model,
            name="PiperModelLoader",
            daemon=True,
        )
        self.loader.start()
        self.worker = threading.Thread(
            target=self._worker,
            name="PiperTTSWorker",
            daemon=True,
        )
        self.worker.start()

    def _load_model(self):
        try:
            self.model = self.piper.PiperVoice.load(
                str(ensure_model(self.voice, self.home))
            )
            self.playback.input_args = raw_pcm_args(self.model.config.sample_rate)
        except Exception as error:
            self.model_error = error
        finally:
            self.model_ready.set()

    def _await_model(self):
        """Return the loaded voice, or None once the failure has been reported.

        The error is printed once and then cleared: a model that failed to
        load fails for every sentence in the session, and repeating it would
        bury the transcript.
        """
        self.model_ready.wait(timeout=self.LOAD_TIMEOUT_SECONDS)
        if self.model is None:
            if self.model_error is not None:
                print(f"\nPiper TTS error: {self.model_error}", file=sys.stderr)
                self.model_error = None
            return None
        return self.model

    def _synthesize(self, model, text):
        """Render one sentence to trimmed PCM."""
        pcm = b"".join(chunk.audio_int16_bytes for chunk in model.synthesize(text))
        return trim_silence(pcm, model.config.sample_rate)

    def _speak_one(self, turn, text):
        model = self._await_model()
        if model is None or self._abandoned(turn):
            return
        audio = self._synthesize(model, text)
        if audio and not self._abandoned(turn):
            self.playback.play(audio, abandoned=lambda: self._abandoned(turn))

    def _worker(self):
        while True:
            item = self.sentences.get()
            if item is self.stop_item:
                return
            turn, text = item
            try:
                self._speak_one(turn, text)
            except Exception as error:
                if not self.shutdown_requested.is_set():
                    print(f"\nPiper TTS error: {error}", file=sys.stderr, flush=True)
            finally:
                self.echo.remember(
                    text, retention=self.SPOKEN_RETENTION_SECONDS, replace=True
                )
                self.activity.finished()
