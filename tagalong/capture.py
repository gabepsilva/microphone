#!/usr/bin/env python3
"""Feed microphone and application-stream audio into Moonshine transcription.

``ApplicationStreamTranscriber`` runs a reader thread and a worker thread. Both
are daemons and both are joined with a timeout, because either can be blocked
on the recorder or on the model when the interface quits.
``tools/worker_gate.py`` enforces that contract.
"""

from __future__ import annotations

import ctypes
import queue
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from moonshine_voice.transcriber import Transcriber, check_error

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


# Moonshine hands back every line the stream has ever produced on every
# transcription update, and by default each one carries its audio. Parsing
# that means rebuilding one Python float per sample, for the whole session,
# four times per second of audio — 215ms per update at 52 lines, against 3ms
# for the recognition itself, and all of it holding the GIL while the
# interface is trying to repaint. Nothing here reads ``line.audio_data``, so
# the whole cost is waste, and turning it off keeps a long session as cheap
# as a fresh one.
TRANSCRIBER_OPTIONS = {"return_audio_data": "false"}


def transcriber_options(options=None):
    """Merge caller options over the defaults every channel shares."""
    merged = dict(TRANSCRIBER_OPTIONS)
    if options:
        merged.update(options)
    return merged


def release_microphone_input(transcriber) -> None:
    """Stop and drop the live PortAudio capture stream, if any.

    Moonshine's ``MicTranscriber.stop`` / ``close`` leave ``_sd_stream``
    running. An orphaned PortAudio callback into a closed Moonshine model
    crashes the process a few seconds later — exactly when a session switches
    microphones by rebuilding the channel, or when the interpreter reclaims
    the retired one.
    """
    stream = getattr(transcriber, "_sd_stream", None)
    if stream is None:
        return
    with suppress(Exception):
        stream.stop()
    with suppress(Exception):
        stream.close()
    transcriber._sd_stream = None


def switch_microphone_input(transcriber, device: int) -> None:
    """Move a live transcriber to another PortAudio input, rolling back errors."""
    previous_device = transcriber._device
    previous_samplerate = transcriber._samplerate
    previous_stream = transcriber._sd_stream

    transcriber._should_listen = False
    if previous_stream is not None:
        previous_stream.stop()
    transcriber._device = device
    transcriber._sd_stream = None
    try:
        transcriber._start_listening()
    except Exception:
        replacement = transcriber._sd_stream
        if replacement is not None:
            with suppress(Exception):
                replacement.close()
        transcriber._device = previous_device
        transcriber._samplerate = previous_samplerate
        transcriber._sd_stream = previous_stream
        if previous_stream is not None:
            previous_stream.start()
        transcriber._should_listen = True
        raise
    if previous_stream is not None:
        previous_stream.close()
    transcriber._should_listen = True


class MuteGate:
    """Make a channel silent to the recognizer while it is muted.

    Muting used to be applied after recognition, in the listener's transcript
    handlers: the words were decoded and then thrown away. A muted channel
    still paid full inference for speech nobody would read, and still held the
    GIL against the channel that was not muted.

    The gate replaces the captured audio with silence rather than dropping it.
    Dropping is cheaper by the few milliseconds silence costs to recognize, but
    it leaves the line that was open when the mute arrived open forever: no
    further audio means no further updates, so the recognizer never sees the
    gap that ends a line, and the half-sentence spoken before the mute is still
    there to be glued onto the first words spoken after the unmute — and
    submitted, because by then the listener is listening again. Feeding silence
    is what mute means physically, and it lets the recognizer close that line
    the same way it closes every other one.

    Written from the interface thread and read from an audio thread with no
    lock, like ``SoundActivityReporter``'s release deadline: the assignment is
    atomic, and the only cost of reading the previous value is gating one block
    later than the click.
    """

    def __init__(self, muted: bool = False):
        self.muted = muted

    def set_muted(self, muted) -> None:
        """Start or stop replacing this channel's audio with silence."""
        self.muted = bool(muted)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Return the audio the recognizer should hear for these samples."""
        if not self.muted:
            return samples
        return np.zeros_like(samples)


def metered_mic_transcriber(*args, level_reporter: SoundActivityReporter, **kwargs):
    """Create microphone capture only when the audio runtime is starting."""
    from moonshine_voice import MicTranscriber

    class MeteredMicTranscriber(MicTranscriber):
        """Add a non-blocking level tap and a mute gate to microphone capture."""

        def __init__(self, *args, level_reporter: SoundActivityReporter, **kwargs):
            super().__init__(*args, **kwargs)
            self.level_reporter = level_reporter
            self.mute_gate = MuteGate()

        def set_muted(self, muted) -> None:
            """Stop feeding microphone speech to the recognizer."""
            self.mute_gate.set_muted(muted)

        def switch_device(self, device: int) -> None:
            """Move capture to another input without reloading Moonshine."""
            switch_microphone_input(self, device)

        def _open_input_stream(self, samplerate, callback):
            def metered_callback(in_data, frames, time_info, status):
                # The level tap reads the microphone, not the gated audio: a
                # muted channel still has to show that it can hear you, which
                # is how you check the microphone is alive before unmuting.
                self.level_reporter.update(in_data)
                callback(self.mute_gate.apply(in_data), frames, time_info, status)

            return super()._open_input_stream(samplerate, metered_callback)

    kwargs["options"] = transcriber_options(kwargs.get("options"))
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
    return samples[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)


def feed_stream_audio(stream, samples, sample_rate):
    """Push float samples into a transcription stream without a Python list.

    Moonshine's ``Stream.add_audio`` builds its ``ctypes`` buffer by unpacking
    a ``list[float]`` — one Python float per sample, on every capture block.
    Real streams expose the same C entry point, so a contiguous float32 view
    can be handed over with ``from_buffer``. Fakes and any stream without the
    library handle keep the list path.
    """
    audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    lib = getattr(stream, "_lib", None)
    if lib is None:
        stream.add_audio(audio.tolist(), sample_rate)
        return
    count = int(audio.size)
    if count == 0:
        return
    error = lib.moonshine_transcribe_add_audio_to_stream(
        stream._transcriber._handle,
        stream._handle,
        (ctypes.c_float * count).from_buffer(audio),
        count,
        int(sample_rate),
        0,
    )
    check_error(error)
    # Mirror Stream.add_audio: advance the stream clock and refresh when the
    # update interval has elapsed, otherwise partials stall until the next
    # list-based call.
    stream._stream_time += count / sample_rate
    if stream._stream_time - stream._last_update_time >= stream._update_interval:
        stream.update_transcription(stream._transcribe_flags)
        stream._last_update_time = stream._stream_time


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


def queued_chunk_limit(capture, backlog_seconds):
    """How many capture blocks fit in ``backlog_seconds`` of audio."""
    seconds_per_chunk = capture.blocksize / capture.samplerate
    return max(1, int(backlog_seconds / seconds_per_chunk))


def offer_audio_chunk(audio_queue, chunk, stop_item):
    """Enqueue ``chunk``, dropping the oldest audio when the backlog is full.

    Preferring recent far-end audio over an unbounded backlog keeps
    transcription interactive when recognition falls behind. The stop
    sentinel is never discarded: if it is the oldest item, it is put back
    and the new chunk is abandoned so shutdown still wins.
    """
    while True:
        try:
            audio_queue.put_nowait(chunk)
            return True
        except queue.Full:
            try:
                dropped = audio_queue.get_nowait()
            except queue.Empty:
                continue
            if dropped is stop_item:
                audio_queue.put(stop_item)
                return False


def deliver_stop_item(audio_queue, stop_item):
    """Ensure the worker sees ``stop_item`` even when the queue is full."""
    while True:
        try:
            audio_queue.put_nowait(stop_item)
            return
        except queue.Full:
            with suppress(queue.Empty):
                audio_queue.get_nowait()


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
    # When recognition lags, keep only this much recent far-end audio. Older
    # blocks are dropped so the transcript stays near live speech instead of
    # growing a backlog nobody will wait for.
    MAX_BACKLOG_SECONDS = 2.0

    def __init__(
        self,
        model_path,
        model_arch,
        tap,
        capture=DEFAULT_CAPTURE,
        level_reporter=None,
    ):
        require_pipewire()
        self.transcriber = Transcriber(
            model_path, model_arch, options=transcriber_options()
        )
        self.stream = self.transcriber.create_stream(capture.update_interval)
        self.tap = tap
        self.capture = capture
        self.level_reporter = level_reporter
        self.process = None
        self.reader = None
        self.worker = None
        self.audio_queue = queue.Queue(
            maxsize=queued_chunk_limit(capture, self.MAX_BACKLOG_SECONDS)
        )
        self.stop_item = object()
        self.started = False
        self.mute_gate = MuteGate()

    def add_listener(self, listener):
        self.stream.add_listener(listener)

    def set_muted(self, muted):
        """Stop feeding far-end speech to the recognizer."""
        self.mute_gate.set_muted(muted)

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
                offer_audio_chunk(self.audio_queue, chunk, self.stop_item)
        finally:
            deliver_stop_item(self.audio_queue, self.stop_item)

    def _process_audio(self):
        while True:
            item = self.audio_queue.get()
            if item is self.stop_item:
                return
            raw_audio, stop_requested = drain_audio_queue(
                self.audio_queue, self.stop_item, item
            )
            audio = to_mono(decode_pcm(raw_audio), self.tap.CHANNELS)
            # As on the microphone, the level tap reads the tapped application
            # rather than the gated audio, so a muted speaker channel still
            # shows the far end talking.
            if self.level_reporter is not None:
                self.level_reporter.update(audio)
            audio = self.mute_gate.apply(audio)
            try:
                feed_stream_audio(self.stream, audio, self.capture.samplerate)
            except Exception as error:
                print(
                    f"Audio transcription error: {error}",
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
