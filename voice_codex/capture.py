#!/usr/bin/env python3
"""Feed microphone and sink-monitor audio into Moonshine transcription.

``PulseMonitorTranscriber`` runs a reader thread and a worker thread. Both are
daemons and both are joined with a timeout, because either can be blocked on
``parec`` or on the model when the interface quits. ``tools/worker_gate.py``
enforces that contract.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from moonshine_voice.transcriber import Transcriber


def audio_level(samples: np.ndarray) -> float:
    """Return a clipped, display-friendly RMS level for normalized samples."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return min(1.0, rms * 5.0)


class AudioLevelReporter:
    """Rate-limit live audio-level updates sent to the transcript display."""

    def __init__(self, display, channel, interval=0.1):
        self.display = display
        self.channel = channel
        self.interval = interval
        self.last_update = float("-inf")

    def update(self, samples: np.ndarray) -> None:
        now = time.monotonic()
        if now - self.last_update < self.interval:
            return
        self.last_update = now
        self.display.set_audio(self.channel, level=audio_level(samples))


def metered_mic_transcriber(*args, level_reporter: AudioLevelReporter, **kwargs):
    """Create microphone capture only when the audio runtime is starting."""
    from moonshine_voice import MicTranscriber

    class MeteredMicTranscriber(MicTranscriber):
        """Add a non-blocking level tap to Moonshine Voice microphone capture."""

        def __init__(self, *args, level_reporter: AudioLevelReporter, **kwargs):
            super().__init__(*args, **kwargs)
            self.level_reporter = level_reporter

        def _open_input_stream(self, samplerate, callback):
            def metered_callback(in_data, frames, time_info, status):
                self.level_reporter.update(in_data)
                callback(in_data, frames, time_info, status)

            return super()._open_input_stream(samplerate, metered_callback)

    return MeteredMicTranscriber(*args, level_reporter=level_reporter, **kwargs)


@dataclass(frozen=True)
class CaptureSettings:
    """Audio capture parameters for one transcription channel."""

    samplerate: int = 16000
    blocksize: int = 4096
    update_interval: float = 0.5


# A frozen default shared by every channel that does not override it.
DEFAULT_CAPTURE = CaptureSettings()


def parec_command(monitor, samplerate):
    """Build the parec command that streams a sink monitor as raw PCM."""
    return [
        "parec",
        "--record",
        "--raw",
        f"--device={monitor}",
        f"--rate={samplerate}",
        "--format=s16le",
        "--channels=1",
        "--client-name=voice-codex",
        "--stream-name=Them transcription",
    ]


def decode_pcm(raw_audio):
    """Convert little-endian signed 16-bit PCM into normalized float samples."""
    audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
    audio /= 32768.0
    return audio


def drain_audio_queue(audio_queue, stop_item, first):
    """Coalesce everything already queued behind ``first`` into one buffer.

    Batching keeps the transcriber's call rate independent of the capture
    block size, so a backlog is caught up in a single call rather than one
    call per block.
    """
    chunks = [first]
    stop_requested = False
    while True:
        try:
            queued = audio_queue.get_nowait()
        except queue.Empty:
            break
        if queued is stop_item:
            stop_requested = True
            break
        chunks.append(queued)
    return b"".join(chunks), stop_requested


class PulseMonitorTranscriber:
    """Feed a PulseAudio/PipeWire sink monitor into a Moonshine stream."""

    # parec exits rather than blocking when a monitor cannot be attached, so
    # startup waits briefly and checks, instead of discovering it at the first
    # silent read.
    STARTUP_GRACE_SECONDS = 0.05

    def __init__(
        self,
        model_path,
        model_arch,
        monitor,
        capture=DEFAULT_CAPTURE,
        level_reporter=None,
    ):
        if shutil.which("parec") is None:
            raise RuntimeError("parec is required to capture an audio-output monitor.")
        self.transcriber = Transcriber(model_path, model_arch)
        self.stream = self.transcriber.create_stream(capture.update_interval)
        self.monitor = monitor
        self.capture = capture
        self.level_reporter = level_reporter
        self.process = None
        self.reader = None
        self.worker = None
        self.audio_queue = queue.Queue()
        self.stop_item = object()
        self.started = False

    def add_listener(self, listener):
        self.stream.add_listener(listener)

    def _read_audio(self):
        try:
            while (process := self.process) is not None:
                if process.stdout is None:
                    break
                chunk = process.stdout.read(self.capture.blocksize * 2)
                if not chunk:
                    break
                self.audio_queue.put(chunk)
        finally:
            self.audio_queue.put(self.stop_item)

    def _process_audio(self):
        while True:
            item = self.audio_queue.get()
            if item is self.stop_item:
                return
            raw_audio, stop_requested = drain_audio_queue(
                self.audio_queue, self.stop_item, item
            )
            audio = decode_pcm(raw_audio)
            if self.level_reporter is not None:
                self.level_reporter.update(audio)
            try:
                self.stream.add_audio(audio.tolist(), self.capture.samplerate)
            except Exception as error:
                print(
                    f"Them transcription error: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            if stop_requested:
                return

    def start(self):
        if self.started:
            return
        self.stream.start()
        self.process = subprocess.Popen(
            parec_command(self.monitor, self.capture.samplerate),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(self.STARTUP_GRACE_SECONDS)
        if self.process.poll() is not None:
            self.stream.stop()
            raise RuntimeError(
                f"Could not capture audio-output monitor {self.monitor!r}."
            )
        self.worker = threading.Thread(
            target=self._process_audio,
            name="ThemTranscriptionWorker",
            daemon=True,
        )
        self.reader = threading.Thread(
            target=self._read_audio,
            name="ThemAudioReader",
            daemon=True,
        )
        self.worker.start()
        self.reader.start()
        self.started = True

    def stop(self):
        if not self.started:
            return
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.reader is not None:
            self.reader.join(timeout=3)
        if self.worker is not None:
            self.worker.join(timeout=10)
        self.stream.stop()
        self.started = False

    def close(self):
        self.stop()
        if self.process is not None and self.process.stdout is not None:
            self.process.stdout.close()
        self.stream.close()
        self.transcriber.close()
