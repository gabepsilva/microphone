"""PCM decoding, level metering, and the monitor capture lifecycle.

``parec`` and the Moonshine transcriber are faked at their boundaries; the
real reader and worker threads run.
"""

from __future__ import annotations

import queue
import shutil
import struct
import subprocess
import threading

import numpy as np
import pytest

from voice_codex.capture import (
    CaptureSettings,
    PulseMonitorTranscriber,
    SoundActivityReporter,
    audio_level,
    decode_pcm,
    drain_audio_queue,
    parec_command,
)

WAIT_SECONDS = 10


def pcm(*samples):
    """Encode samples as the little-endian signed 16-bit PCM parec emits."""
    return struct.pack(f"<{len(samples)}h", *samples)


class FakeStream:
    """Stand in for a Moonshine transcription stream."""

    def __init__(self):
        self.audio: list[tuple[list[float], int]] = []
        self.listeners: list[object] = []
        self.events: list[str] = []
        self.received = threading.Event()
        self.failure = None

    def add_listener(self, listener):
        self.listeners.append(listener)

    def add_audio(self, samples, samplerate):
        if self.failure is not None:
            raise self.failure
        self.audio.append((samples, samplerate))
        self.received.set()

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class FakeTranscriber:
    def __init__(self, model_path, model_arch):
        self.model_path = model_path
        self.model_arch = model_arch
        self.stream = FakeStream()
        self.closed = False
        self.update_interval = None

    def create_stream(self, update_interval):
        self.update_interval = update_interval
        return self.stream

    def close(self):
        self.closed = True


class FakeParec:
    """Stand in for the parec process, serving a fixed script of reads."""

    def __init__(self, reads, exit_code=None):
        self.stdout = FakePipe(reads)
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15
        self.stdout.unblock()

    def kill(self):
        self.killed = True
        self._exit_code = -9
        self.stdout.unblock()

    def wait(self, timeout=None):  # noqa: ARG002 - matches Popen.wait
        return self._exit_code


class FakePipe:
    def __init__(self, reads):
        self.reads = list(reads)
        self.closed = False

    def read(self, size):  # noqa: ARG002 - matches BufferedReader.read
        if self.reads:
            return self.reads.pop(0)
        return b""

    def unblock(self):
        self.reads = []

    def close(self):
        self.closed = True


class RecordedMonitor(PulseMonitorTranscriber):
    """A monitor transcriber that keeps a handle on its faked parec process."""

    def __init__(self, process, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_process = process


@pytest.fixture
def capture(monkeypatch):
    """Build a monitor transcriber with parec and Moonshine faked."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("voice_codex.capture.Transcriber", FakeTranscriber)
    monkeypatch.setattr(PulseMonitorTranscriber, "STARTUP_GRACE_SECONDS", 0)

    def build(reads=(), exit_code=None, **kwargs):
        process = FakeParec(list(reads), exit_code)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
        return RecordedMonitor(process, "model", "arch", "sink.monitor", **kwargs)

    return build


def test_silence_reports_no_level() -> None:
    assert audio_level(np.zeros(0, dtype=np.float32)) == 0.0
    assert audio_level(np.zeros(64, dtype=np.float32)) == 0.0


def test_a_louder_signal_reports_a_higher_level() -> None:
    quiet = audio_level(np.full(64, 0.02, dtype=np.float32))
    loud = audio_level(np.full(64, 0.15, dtype=np.float32))

    assert 0.0 < quiet < loud < 1.0


def test_the_level_is_clipped_to_one() -> None:
    assert audio_level(np.full(64, 0.9, dtype=np.float32)) == 1.0


def _recording_display(reports):
    return type(
        "Display",
        (),
        {"set_audio": staticmethod(lambda channel, active: reports.append(active))},
    )()


def test_continuing_sound_is_reported_once() -> None:
    """Steady speech is one report, not one per audio block."""
    reports: list[bool] = []
    reporter = SoundActivityReporter(_recording_display(reports), "them")

    reporter.update(np.full(64, 0.1, dtype=np.float32))
    reporter.update(np.full(64, 0.1, dtype=np.float32))
    reporter.update(np.full(64, 0.1, dtype=np.float32))

    assert reports == [True]


def test_continuing_silence_is_never_reported() -> None:
    reports: list[bool] = []
    reporter = SoundActivityReporter(_recording_display(reports), "them")

    reporter.update(np.zeros(64, dtype=np.float32))
    reporter.update(np.zeros(64, dtype=np.float32))

    assert reports == []


def test_a_gap_shorter_than_the_release_does_not_report_silence() -> None:
    """A pause between words must not drop the indicator."""
    reports: list[bool] = []
    clock = iter([0.0, 0.2, 0.3]).__next__
    reporter = SoundActivityReporter(
        _recording_display(reports), "mic", release=0.35, clock=clock
    )

    reporter.update(np.full(64, 0.1, dtype=np.float32))  # t=0.0, sound
    reporter.update(np.zeros(64, dtype=np.float32))  # t=0.2, within release
    reporter.update(np.zeros(64, dtype=np.float32))  # t=0.3, still within

    assert reports == [True]


def test_silence_past_the_release_reports_the_channel_quiet() -> None:
    reports: list[bool] = []
    clock = iter([0.0, 0.5]).__next__
    reporter = SoundActivityReporter(
        _recording_display(reports), "mic", release=0.35, clock=clock
    )

    reporter.update(np.full(64, 0.1, dtype=np.float32))  # t=0.0, sound
    reporter.update(np.zeros(64, dtype=np.float32))  # t=0.5, past release

    assert reports == [True, False]


def test_sound_below_the_threshold_is_not_sound() -> None:
    reports: list[bool] = []
    reporter = SoundActivityReporter(_recording_display(reports), "mic", threshold=0.5)

    reporter.update(np.full(64, 0.02, dtype=np.float32))

    assert reports == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (pcm(0), [0.0]),
        (pcm(16384), [0.5]),
        (pcm(-16384), [-0.5]),
        (pcm(-32768), [-1.0]),
        (pcm(0, 8192, -8192), [0.0, 0.25, -0.25]),
        (b"", []),
    ],
)
def test_pcm_decodes_to_normalized_samples(raw, expected) -> None:
    assert decode_pcm(raw).tolist() == pytest.approx(expected)


def test_decoded_samples_stay_within_the_normalized_range() -> None:
    decoded = decode_pcm(pcm(32767, -32768, 0, 12345))

    assert decoded.dtype == np.float32
    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0


def test_a_backlog_is_coalesced_into_one_buffer() -> None:
    audio_queue = queue.Queue()
    stop = object()
    audio_queue.put(b"second")
    audio_queue.put(b"third")

    raw, stop_requested = drain_audio_queue(audio_queue, stop, b"first")

    assert raw == b"firstsecondthird"
    assert stop_requested is False


def test_a_stop_item_ends_the_backlog_and_is_reported() -> None:
    audio_queue = queue.Queue()
    stop = object()
    audio_queue.put(b"second")
    audio_queue.put(stop)
    audio_queue.put(b"never read")

    raw, stop_requested = drain_audio_queue(audio_queue, stop, b"first")

    assert raw == b"firstsecond"
    assert stop_requested is True


def test_an_empty_queue_yields_only_the_first_chunk() -> None:
    raw, stop_requested = drain_audio_queue(queue.Queue(), object(), b"only")

    assert (raw, stop_requested) == (b"only", False)


def test_the_parec_command_requests_raw_mono_pcm() -> None:
    command = parec_command("sink.monitor", 16000)

    assert command[0] == "parec"
    assert "--device=sink.monitor" in command
    assert "--rate=16000" in command
    assert "--format=s16le" in command
    assert "--channels=1" in command


def test_capture_requires_parec(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="parec is required"):
        PulseMonitorTranscriber("model", "arch", "sink.monitor")


def test_captured_audio_reaches_the_transcription_stream(capture) -> None:
    monitor = capture(reads=[pcm(16384, -16384)])
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)
        samples, samplerate = monitor.stream.audio[0]

        assert samples == pytest.approx([0.5, -0.5])
        assert samplerate == 16000
    finally:
        monitor.stop()


def test_a_capture_level_is_reported_while_transcribing(capture) -> None:
    levels: list[float] = []
    reporter = type("Reporter", (), {"update": staticmethod(levels.append)})()
    monitor = capture(reads=[pcm(16384, -16384)], level_reporter=reporter)
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)
        assert len(levels) == 1
    finally:
        monitor.stop()


def test_a_parec_that_exits_immediately_is_reported(capture) -> None:
    monitor = capture(exit_code=1)

    with pytest.raises(RuntimeError, match="Could not capture audio-output monitor"):
        monitor.start()

    assert monitor.stream.events == ["start", "stop"]


def test_starting_twice_does_not_start_a_second_capture(capture) -> None:
    monitor = capture(reads=[pcm(0)])
    monitor.start()
    try:
        monitor.start()

        assert monitor.stream.events == ["start"]
    finally:
        monitor.stop()


def test_stopping_before_starting_does_nothing(capture) -> None:
    monitor = capture()

    monitor.stop()

    assert monitor.stream.events == []


def test_stopping_terminates_parec_and_stops_the_stream(capture) -> None:
    monitor = capture(reads=[pcm(0)], exit_code=None)
    monitor.start()

    monitor.stop()

    assert monitor.fake_process.terminated
    assert monitor.stream.events == ["start", "stop"]


def test_closing_releases_the_pipe_the_stream_and_the_model(capture) -> None:
    monitor = capture(reads=[pcm(0)])
    monitor.start()

    monitor.close()

    assert monitor.fake_process.stdout.closed
    assert monitor.stream.events == ["start", "stop", "close"]
    assert monitor.transcriber.closed


def test_a_transcription_failure_is_reported_without_ending_capture(
    capture, capsys
) -> None:
    monitor = capture(reads=[pcm(0, 1)])
    monitor.stream.failure = RuntimeError("model is busy")
    monitor.start()
    try:
        monitor.stop()

        assert "Them transcription error: model is busy" in capsys.readouterr().err
    finally:
        monitor.close()


def test_a_listener_is_registered_with_the_stream(capture) -> None:
    monitor = capture()
    listener = object()

    monitor.add_listener(listener)

    assert monitor.stream.listeners == [listener]


def test_capture_settings_reach_the_stream_and_the_reader(capture) -> None:
    monitor = capture(
        capture=CaptureSettings(samplerate=8000, blocksize=256, update_interval=1)
    )

    assert monitor.transcriber.update_interval == 1
    assert monitor.capture.samplerate == 8000
    assert monitor.capture.blocksize == 256
