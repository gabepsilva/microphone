#!/usr/bin/env python3
"""Feed microphone and application-stream audio into Moonshine transcription.

``ApplicationStreamTranscriber`` runs a reader thread and a worker thread. Both
are daemons and both are joined with a timeout, because either can be blocked
on the recorder or on the model when the interface quits.
``tools/worker_gate.py`` enforces that contract.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from moonshine_voice.transcriber import Transcriber

from .session import tagged_environment
from .streams import require_pipewire


def audio_level(samples: np.ndarray) -> float:
    """Return a clipped, display-friendly RMS level for normalized samples."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return min(1.0, rms * 5.0)


# A channel counts as hearing something at or above this display level, and
# keeps saying so for the release window after it drops back. Speech falls
# below any threshold between words, so without the release the indicator
# would flicker off in every syllable gap — reporting as often as the level
# meter it replaced, which is the cost this whole approach exists to avoid.
SOUND_THRESHOLD = 0.08
SOUND_RELEASE_SECONDS = 0.35


class SoundActivityReporter:
    """Tell the display when a channel starts and stops hearing sound.

    Only transitions are reported. A level changed on nearly every audio
    block, and each report forced the interface to lay itself out again;
    presence is stable, so a silent channel and a steadily-speaking one both
    cost nothing to draw.

    The same measurement answers a second question the silence timer needs —
    whether anyone is talking right now — which is why ``hearing_sound`` is a
    read rather than another report.
    """

    def __init__(
        self,
        display,
        channel,
        threshold: float = SOUND_THRESHOLD,
        release: float = SOUND_RELEASE_SECONDS,
        clock=time.monotonic,
    ):
        self.display = display
        self.channel = channel
        self.threshold = threshold
        self.release = release
        self.clock = clock
        self.active = False
        self.loud_until = float("-inf")

    def update(self, samples: np.ndarray) -> None:
        now = self.clock()
        if audio_level(samples) >= self.threshold:
            self.loud_until = now + self.release
        active = now < self.loud_until
        if active == self.active:
            return
        self.active = active
        self.display.set_audio(self.channel, active=active)

    @property
    def hearing_sound(self) -> bool:
        """Whether this channel is hearing sound at this moment.

        Recomputed against the release deadline rather than read off the last
        transition, because the reads come from the silence timer rather than
        from the audio thread: a release that expired between two blocks would
        otherwise report a channel as still busy for as long as the audio was
        quiet enough to stop reporting.

        Written on an audio thread and read on a timer thread with no lock. A
        float assignment is atomic, and the only cost of reading the previous
        one is deciding this question a block earlier than the next sample.
        """
        return self.clock() < self.loud_until


def metered_mic_transcriber(*args, level_reporter: SoundActivityReporter, **kwargs):
    """Create microphone capture only when the audio runtime is starting."""
    from moonshine_voice import MicTranscriber

    class MeteredMicTranscriber(MicTranscriber):
        """Add a non-blocking level tap to Moonshine Voice microphone capture."""

        def __init__(self, *args, level_reporter: SoundActivityReporter, **kwargs):
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


def decode_pcm(raw_audio):
    """Convert little-endian signed 16-bit PCM into normalized float samples."""
    audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
    audio /= 32768.0
    return audio


def to_mono(samples, channels):
    """Average interleaved channels into the single one Moonshine transcribes.

    Averaged rather than summed, and rather than one channel kept: summing
    doubles a stream that carries the same audio on both sides, which is most
    of them, and keeping one side alone loses a speaker panned to the other.

    A trailing partial frame is dropped. It only appears when the recorder is
    cut off mid-frame at shutdown, and half a frame of audio is worth less than
    every later frame being off by one channel.
    """
    if channels == 1:
        return samples
    usable = samples.size - samples.size % channels
    return samples[:usable].reshape(-1, channels).mean(axis=1)


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


class ApplicationStreamTranscriber:
    """Feed one application's PipeWire playback into a Moonshine stream.

    The recorder and the links are separate halves of the same tap and are
    started in that order: the capture node does not exist in the graph until
    the recorder has opened it, so nothing could be linked to it before.
    """

    # The recorder exits rather than blocking when it cannot reach the audio
    # server at all, so startup waits briefly and checks, instead of
    # discovering it at the first silent read. A tap with nothing linked yet is
    # a different thing entirely, and reads as silence on purpose.
    STARTUP_GRACE_SECONDS = 0.05

    def __init__(
        self,
        model_path,
        model_arch,
        tap,
        capture=DEFAULT_CAPTURE,
        level_reporter=None,
    ):
        require_pipewire()
        self.transcriber = Transcriber(model_path, model_arch)
        self.stream = self.transcriber.create_stream(capture.update_interval)
        self.tap = tap
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
                # Whole frames only: a read that split one would put every
                # later sample on the wrong channel.
                chunk = process.stdout.read(
                    self.capture.blocksize * 2 * self.tap.CHANNELS
                )
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
            audio = to_mono(decode_pcm(raw_audio), self.tap.CHANNELS)
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
            self.tap.command(self.capture.samplerate),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # Tagged so a recorder this session leaves behind can be recognized
            # and swept by the next one, whatever it ends up parented to.
            env=tagged_environment(),
        )
        time.sleep(self.STARTUP_GRACE_SECONDS)
        if self.process.poll() is not None:
            self.stream.stop()
            raise RuntimeError(
                f"Could not capture the audio of {self.tap.application!r}."
            )
        self.tap.start()
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
        # The linker stops first so it cannot rebuild the tap around a recorder
        # that is on its way out, which would leave a link behind pointing at a
        # node nobody owns.
        self.tap.stop()
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
