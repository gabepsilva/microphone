"""PCM decoding, level metering, and the stream capture lifecycle.

The recorder process, the PipeWire tap, and the Moonshine transcriber are
faked at their boundaries; the real reader and worker threads run.
"""

from __future__ import annotations

import queue
import shutil
import struct
import subprocess
import sys
import threading
import types

import numpy as np
import pytest

from tagalong.capture import (
    ApplicationStreamTranscriber,
    CaptureSettings,
    MuteGate,
    SoundActivityReporter,
    audio_level,
    decode_pcm,
    deliver_stop_item,
    drain_audio_queue,
    feed_stream_audio,
    metered_mic_transcriber,
    offer_audio_chunk,
    queued_chunk_limit,
    to_mono,
    transcriber_options,
)

WAIT_SECONDS = 10


def pcm(*samples):
    """Encode samples as the little-endian signed 16-bit PCM the tap emits."""
    return struct.pack(f"<{len(samples)}h", *samples)


def stereo(*samples):
    """The same, as the interleaved stereo frames the tap actually captures."""
    return pcm(*[value for sample in samples for value in (sample, sample)])


class FakeStream:
    """Stand in for a Moonshine transcription stream."""

    def __init__(self):
        self.audio: list[tuple[list[float], int]] = []
        self.listeners: list[object] = []
        self.events: list[str] = []
        self.received = threading.Event()
        self.failed = threading.Event()
        self.failure = None

    def add_listener(self, listener):
        self.listeners.append(listener)

    def add_audio(self, samples, samplerate):
        if self.failure is not None:
            self.failed.set()
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
    def __init__(self, model_path, model_arch, options=None):
        self.model_path = model_path
        self.model_arch = model_arch
        self.options = options
        self.stream = FakeStream()
        self.closed = False
        self.update_interval = None

    def create_stream(self, update_interval):
        self.update_interval = update_interval
        return self.stream

    def close(self):
        self.closed = True


class FakeRecorder:
    """Stand in for the pw-record process, serving a fixed script of reads."""

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


class FakeTap:
    """Stand in for the PipeWire tap, recording when it was told to follow."""

    CHANNELS = 2

    def __init__(self):
        self.application = "ZOOM VoiceEngine"
        self.events: list[str] = []

    def command(self, samplerate):
        self.events.append(f"command {samplerate}")
        return ["pw-record", str(samplerate)]

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


class RecordedCapture(ApplicationStreamTranscriber):
    """A stream transcriber that keeps a handle on its faked recorder."""

    def __init__(self, process, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_process = process


@pytest.fixture
def capture(monkeypatch):
    """Build a stream transcriber with the recorder, tap, and Moonshine faked."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("tagalong.capture.Transcriber", FakeTranscriber)
    monkeypatch.setattr(ApplicationStreamTranscriber, "STARTUP_GRACE_SECONDS", 0)

    def build(reads=(), exit_code=None, **kwargs):
        process = FakeRecorder(list(reads), exit_code)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
        return RecordedCapture(process, "model", "arch", FakeTap(), **kwargs)

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
    reporter = SoundActivityReporter(_recording_display(reports), "audio")

    reporter.update(np.full(64, 0.1, dtype=np.float32))
    reporter.update(np.full(64, 0.1, dtype=np.float32))
    reporter.update(np.full(64, 0.1, dtype=np.float32))

    assert reports == [True]


def test_activity_transitions_notify_an_optional_hook() -> None:
    """Energy-aware turn closure arms from these edges, not from every block."""
    transitions: list[bool] = []
    clock = iter([0.0, 1.0]).__next__
    reporter = SoundActivityReporter(
        _recording_display([]),
        "mic",
        release=0.35,
        clock=clock,
    )
    reporter.on_transition = transitions.append

    reporter.update(np.full(64, 0.1, dtype=np.float32))  # loud
    reporter.update(np.zeros(64, dtype=np.float32))  # quiet after release

    assert transitions == [True, False]


def test_continuing_silence_is_never_reported() -> None:
    reports: list[bool] = []
    reporter = SoundActivityReporter(_recording_display(reports), "audio")

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


# --------------------------------------------------------------------------
# Asking the tap directly
#
# The silence timer reads this to decide whether a speaker is still talking,
# so what matters is that a read is answered from the release deadline rather
# than from whatever the last block happened to report.
# --------------------------------------------------------------------------


def test_a_quiet_channel_is_hearing_nothing() -> None:
    reporter = SoundActivityReporter(_recording_display([]), "mic")

    assert reporter.hearing_sound is False


def test_a_channel_with_sound_on_it_is_hearing_something() -> None:
    reporter = SoundActivityReporter(_recording_display([]), "mic")

    reporter.update(np.full(64, 0.1, dtype=np.float32))

    assert reporter.hearing_sound is True


def test_the_channel_is_still_heard_through_a_gap_between_words() -> None:
    clock = iter([0.0, 0.2]).__next__
    reporter = SoundActivityReporter(
        _recording_display([]), "mic", release=0.35, clock=clock
    )

    reporter.update(np.full(64, 0.1, dtype=np.float32))  # t=0.0, sound

    assert reporter.hearing_sound is True  # t=0.2, inside the release


def test_a_release_that_expired_between_blocks_reads_quiet() -> None:
    """The read is recomputed, not taken from the last block's verdict.

    No audio arrives while a timer is deciding whether to wait longer, so a
    read that trusted the last transition would report a speaker as still
    talking for as long as the channel stayed quiet enough to stop reporting.
    """
    clock = iter([0.0, 0.5]).__next__
    reporter = SoundActivityReporter(
        _recording_display([]), "mic", release=0.35, clock=clock
    )

    reporter.update(np.full(64, 0.1, dtype=np.float32))  # t=0.0, sound

    assert reporter.active is True  # what the last block reported
    assert reporter.hearing_sound is False  # t=0.5, the release has run out


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


def test_the_chunk_limit_covers_the_configured_backlog() -> None:
    capture = CaptureSettings(samplerate=16000, blocksize=4096)

    assert queued_chunk_limit(capture, 2.0) == 7


def test_a_short_backlog_still_keeps_one_chunk() -> None:
    capture = CaptureSettings(samplerate=16000, blocksize=4096)

    assert queued_chunk_limit(capture, 0.1) == 1


def test_a_full_queue_drops_the_oldest_audio() -> None:
    audio_queue = queue.Queue(maxsize=2)
    stop = object()

    assert offer_audio_chunk(audio_queue, b"old", stop) is True
    assert offer_audio_chunk(audio_queue, b"mid", stop) is True
    assert offer_audio_chunk(audio_queue, b"new", stop) is True

    assert audio_queue.get_nowait() == b"mid"
    assert audio_queue.get_nowait() == b"new"


def test_offering_never_discards_the_stop_sentinel() -> None:
    audio_queue = queue.Queue(maxsize=1)
    stop = object()
    audio_queue.put(stop)

    assert offer_audio_chunk(audio_queue, b"late", stop) is False
    assert audio_queue.get_nowait() is stop


def test_offer_retries_when_a_slot_opens_between_full_and_drop() -> None:
    """Full then Empty is a race with the worker; the put must be retried."""

    class RaceQueue:
        def __init__(self):
            self._full_once = True
            self._empty_once = True
            self.items: list[bytes] = []

        def put_nowait(self, item):
            if self._full_once:
                self._full_once = False
                raise queue.Full
            self.items.append(item)

        def get_nowait(self):
            if self._empty_once:
                self._empty_once = False
                raise queue.Empty
            raise AssertionError("drop should not run after Empty")

    audio_queue = RaceQueue()

    assert offer_audio_chunk(audio_queue, b"chunk", object()) is True
    assert audio_queue.items == [b"chunk"]


def test_stop_delivery_clears_a_full_queue() -> None:
    audio_queue = queue.Queue(maxsize=1)
    stop = object()
    audio_queue.put(b"stale")

    deliver_stop_item(audio_queue, stop)

    assert audio_queue.get_nowait() is stop


def test_interleaved_channels_are_averaged_rather_than_summed() -> None:
    """Summing doubles a stream carrying the same audio on both sides."""
    samples = decode_pcm(stereo(16384, -16384))

    mono = to_mono(samples, 2)

    assert mono.dtype == np.float32
    assert mono.tolist() == pytest.approx([0.5, -0.5])


def test_a_speaker_panned_to_one_side_survives_the_downmix() -> None:
    """Keeping one channel alone would drop them entirely."""
    samples = decode_pcm(pcm(16384, 0))

    assert to_mono(samples, 2).tolist() == pytest.approx([0.25])


def test_a_mono_tap_is_left_exactly_as_it_arrived() -> None:
    samples = decode_pcm(pcm(16384, -16384))

    assert to_mono(samples, 1) is samples


def test_a_frame_cut_in_half_at_shutdown_is_dropped() -> None:
    """Keeping it would put every later sample on the wrong channel."""
    samples = decode_pcm(pcm(16384, 16384, 8192))

    assert to_mono(samples, 2).tolist() == pytest.approx([0.5])


def test_a_fake_stream_still_receives_a_python_list() -> None:
    stream = FakeStream()

    feed_stream_audio(stream, np.array([0.5, -0.5], dtype=np.float32), 16000)

    assert stream.audio == [([0.5, -0.5], 16000)]


def test_a_moonshine_stream_is_fed_from_a_float32_buffer() -> None:
    """Avoid building one Python float per sample on every capture block."""
    calls: list[tuple] = []

    class Lib:
        @staticmethod
        def moonshine_transcribe_add_audio_to_stream(
            _transcriber, _stream, audio, count, rate, _flags
        ):
            calls.append((list(audio), count, rate))
            return 0

    stream = types.SimpleNamespace(
        _lib=Lib(),
        _transcriber=types.SimpleNamespace(_handle=object()),
        _handle=object(),
        _stream_time=0.0,
        _last_update_time=0.0,
        _update_interval=0.5,
        _transcribe_flags=0,
        update_transcription=lambda *_a, **_k: calls.append(("update",)),
        add_audio=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("list path must not run")
        ),
    )

    feed_stream_audio(stream, np.array([0.25, -0.5], dtype=np.float32), 16000)

    assert calls == [([0.25, -0.5], 2, 16000)]
    assert stream._stream_time == pytest.approx(2 / 16000)


def test_feeding_enough_audio_refreshes_transcription() -> None:
    updates: list[str] = []

    class Lib:
        @staticmethod
        def moonshine_transcribe_add_audio_to_stream(*_args):
            return 0

    stream = types.SimpleNamespace(
        _lib=Lib(),
        _transcriber=types.SimpleNamespace(_handle=object()),
        _handle=object(),
        _stream_time=0.0,
        _last_update_time=0.0,
        _update_interval=0.01,
        _transcribe_flags=0,
        update_transcription=lambda *_a, **_k: updates.append("tick"),
        add_audio=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("list path must not run")
        ),
    )
    samples = np.zeros(320, dtype=np.float32)  # 0.02s at 16kHz

    feed_stream_audio(stream, samples, 16000)

    assert updates == ["tick"]
    assert stream._last_update_time == stream._stream_time


def test_a_moonshine_api_mismatch_falls_back_to_the_list_path() -> None:
    """Private stream shape can change; audio must still reach the recognizer."""
    received: list[tuple] = []

    class Lib:
        @staticmethod
        def moonshine_transcribe_add_audio_to_stream(*_args):
            raise AttributeError("unexpected stream layout")

    stream = types.SimpleNamespace(
        _lib=Lib(),
        _transcriber=types.SimpleNamespace(_handle=object()),
        _handle=object(),
        add_audio=lambda samples, rate: received.append((samples, rate)),
    )

    feed_stream_audio(stream, np.array([0.5], dtype=np.float32), 16000)

    assert received == [([0.5], 16000)]


def test_bookkeeping_failure_after_a_native_feed_does_not_double_feed() -> None:
    feeds = {"native": 0, "list": 0}

    class Lib:
        @staticmethod
        def moonshine_transcribe_add_audio_to_stream(*_args):
            feeds["native"] += 1
            return 0

    stream = types.SimpleNamespace(
        _lib=Lib(),
        _transcriber=types.SimpleNamespace(_handle=object()),
        _handle=object(),
        # Missing _stream_time forces AttributeError after the native add.
        add_audio=lambda *_a, **_k: feeds.__setitem__("list", feeds["list"] + 1),
    )

    feed_stream_audio(stream, np.array([0.25], dtype=np.float32), 16000)

    assert feeds == {"native": 1, "list": 0}


def test_an_empty_queue_yields_only_the_first_chunk() -> None:
    raw, stop_requested = drain_audio_queue(queue.Queue(), object(), b"only")

    assert (raw, stop_requested) == (b"only", False)


def test_capture_requires_the_pipewire_tools(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="pw-record is required"):
        ApplicationStreamTranscriber("model", "arch", FakeTap())


def test_the_recorder_is_asked_for_the_capture_samplerate(capture) -> None:
    monitor = capture(reads=[stereo(0)])
    monitor.start()
    try:
        assert "command 16000" in monitor.tap.events
    finally:
        monitor.stop()


def test_the_tap_follows_the_application_only_once_the_recorder_is_up(
    capture,
) -> None:
    """Nothing can be linked to a capture node that is not in the graph yet."""
    monitor = capture(reads=[stereo(0)])
    monitor.start()
    try:
        assert monitor.tap.events == ["command 16000", "start"]
    finally:
        monitor.stop()


def test_a_recorder_that_exits_immediately_leaves_the_tap_unstarted(capture) -> None:
    monitor = capture(exit_code=1)

    with pytest.raises(RuntimeError, match="Could not capture the audio"):
        monitor.start()

    assert "start" not in monitor.tap.events


def test_monitor_capture_leaves_line_audio_in_the_recognizer(capture) -> None:
    """Every update re-parses the whole transcript, so lines must stay cheap.

    Moonshine returns each line's audio as one Python float per sample on
    every update, for every line the session has produced. Nothing reads it,
    and parsing it is what made a long session cost more than a fresh one.
    """
    monitor = capture()

    assert monitor.transcriber.options["return_audio_data"] == "false"


def test_capture_options_extend_the_shared_defaults() -> None:
    """A channel that needs its own option keeps the shared ones too."""
    assert transcriber_options({"spelling_model_path": "/models/spelling.ort"}) == {
        "return_audio_data": "false",
        "spelling_model_path": "/models/spelling.ort",
    }


def test_microphone_capture_leaves_line_audio_in_the_recognizer(monkeypatch) -> None:
    """The microphone channel carries the same cost, and the same fix."""
    recorded = {}

    class FakeMicTranscriber:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "moonshine_voice",
        types.SimpleNamespace(MicTranscriber=FakeMicTranscriber),
    )

    reporter = SoundActivityReporter(display=None, channel="mic")
    metered_mic_transcriber(model_path="model", level_reporter=reporter)

    assert recorded["options"]["return_audio_data"] == "false"


def test_captured_audio_reaches_the_transcription_stream(capture) -> None:
    monitor = capture(reads=[stereo(16384, -16384)])
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
    monitor = capture(reads=[stereo(16384, -16384)], level_reporter=reporter)
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)
        assert len(levels) == 1
    finally:
        monitor.stop()


def test_a_recorder_that_exits_immediately_is_reported(capture) -> None:
    monitor = capture(exit_code=1)

    with pytest.raises(RuntimeError, match="Could not capture the audio"):
        monitor.start()

    assert monitor.stream.events == ["start", "stop"]


def test_starting_twice_does_not_start_a_second_capture(capture) -> None:
    monitor = capture(reads=[stereo(0)])
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


def test_stopping_terminates_the_recorder_and_stops_the_stream(capture) -> None:
    monitor = capture(reads=[stereo(0)], exit_code=None)
    monitor.start()

    monitor.stop()

    assert monitor.fake_process.terminated
    assert monitor.stream.events == ["start", "stop"]


def test_stopping_retires_the_linker_before_the_recorder_it_wires(capture) -> None:
    """A relink after the recorder goes leaves a link to a node nobody owns."""
    monitor = capture(reads=[stereo(0)], exit_code=None)
    monitor.start()
    monitor.tap.events.clear()

    monitor.stop()

    assert monitor.tap.events == ["stop"]
    assert monitor.fake_process.terminated


def test_closing_releases_the_pipe_the_stream_and_the_model(capture) -> None:
    monitor = capture(reads=[stereo(0)])
    monitor.start()

    monitor.close()

    assert monitor.fake_process.stdout.closed
    assert monitor.stream.events == ["start", "stop", "close"]
    assert monitor.transcriber.closed


def test_a_transcription_failure_is_reported_without_ending_capture(
    capture, capsys
) -> None:
    monitor = capture(reads=[stereo(0, 1)])
    monitor.stream.failure = RuntimeError("model is busy")
    monitor.start()
    try:
        assert monitor.stream.failed.wait(WAIT_SECONDS)
        monitor.stop()

        assert "Audio transcription error: model is busy" in capsys.readouterr().err
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


class RecordingReporter(SoundActivityReporter):
    """A level tap that keeps every block it was handed."""

    def __init__(self):
        super().__init__(display=None, channel="mic")
        self.blocks: list[np.ndarray] = []

    def update(self, samples: np.ndarray) -> None:
        self.blocks.append(samples)


def test_an_unmuted_gate_passes_the_audio_through() -> None:
    samples = np.array([0.5, -0.5], dtype=np.float32)

    assert MuteGate().apply(samples) is samples


def test_a_muted_gate_replaces_speech_with_silence() -> None:
    """Mute has to be applied before recognition, not to its results."""
    gate = MuteGate()
    gate.set_muted(True)

    gated = gate.apply(np.array([0.5, -0.5], dtype=np.float32))

    assert gated.tolist() == [0.0, 0.0]


def test_a_muted_gate_keeps_the_block_it_silences() -> None:
    """Silence rather than nothing, so the open line still meets a gap."""
    gate = MuteGate()
    gate.set_muted(True)

    gated = gate.apply(np.zeros(7, dtype=np.float32))

    assert gated.shape == (7,)
    assert gated.dtype == np.float32


def test_a_muted_gate_leaves_the_captured_block_alone() -> None:
    """The level tap reads the same buffer, and reads it after the gate."""
    samples = np.array([0.5, -0.5], dtype=np.float32)
    gate = MuteGate()
    gate.set_muted(True)

    gate.apply(samples)

    assert samples.tolist() == [0.5, -0.5]


def test_unmuting_lets_the_audio_through_again() -> None:
    gate = MuteGate()
    gate.set_muted(True)
    gate.set_muted(False)

    assert gate.apply(np.array([0.5], dtype=np.float32)).tolist() == [0.5]


def test_a_muted_far_end_sends_no_speech_to_the_recognizer(capture) -> None:
    """The regression: a muted channel must not pay for words nobody reads."""
    monitor = capture(reads=[stereo(16384, -16384)])
    monitor.set_muted(True)
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)
        samples, _ = monitor.stream.audio[0]

        assert samples == [0.0, 0.0]
    finally:
        monitor.stop()


def test_a_muted_far_end_still_reports_what_it_is_doing(capture) -> None:
    """Mute stops the transcription, not the activity the sidebar shows."""
    reporter = RecordingReporter()
    monitor = capture(reads=[stereo(16384, -16384)], level_reporter=reporter)
    monitor.set_muted(True)
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)

        assert [block.tolist() for block in reporter.blocks] == [[0.5, -0.5]]
    finally:
        monitor.stop()


def test_an_unmuted_far_end_reaches_the_recognizer_again(capture) -> None:
    """Unmuting has to restore capture, not just the transcript."""
    monitor = capture(reads=[stereo(16384, -16384)])
    monitor.set_muted(True)
    monitor.set_muted(False)
    monitor.start()
    try:
        assert monitor.stream.received.wait(WAIT_SECONDS)
        samples, _ = monitor.stream.audio[0]

        assert samples == pytest.approx([0.5, -0.5])
    finally:
        monitor.stop()


class FakeMicBase:
    """Stand in for Moonshine's ``MicTranscriber`` at the capture boundary."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opened_callback = None
        self._device = kwargs.get("device")
        self._samplerate = kwargs.get("samplerate", 16000)
        self._sd_stream = None
        self._should_listen = False

    def _open_input_stream(self, samplerate, callback):
        self.samplerate = samplerate
        self.opened_callback = callback
        return object()


def metered_mic(monkeypatch, reporter):
    """Build the microphone channel with the Moonshine base class faked."""
    monkeypatch.setitem(
        sys.modules,
        "moonshine_voice",
        types.SimpleNamespace(MicTranscriber=FakeMicBase),
    )
    return metered_mic_transcriber(model_path="model", level_reporter=reporter)


def recorded_microphone_block(transcriber, block):
    """Push one captured block through the transcriber's capture callback."""
    heard: list[np.ndarray] = []
    transcriber._open_input_stream(16000, lambda in_data, *_: heard.append(in_data))
    transcriber.opened_callback(block, len(block), None, None)
    return heard


def test_an_unmuted_microphone_reaches_the_recognizer(monkeypatch) -> None:
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    block = np.array([0.5, -0.5], dtype=np.float32)

    heard = recorded_microphone_block(transcriber, block)

    assert [samples.tolist() for samples in heard] == [[0.5, -0.5]]


def test_a_muted_microphone_sends_no_speech_to_the_recognizer(monkeypatch) -> None:
    """The regression on the channel the user speaks on."""
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    transcriber.set_muted(True)

    heard = recorded_microphone_block(
        transcriber, np.array([0.5, -0.5], dtype=np.float32)
    )

    assert [samples.tolist() for samples in heard] == [[0.0, 0.0]]


def test_a_muted_microphone_still_shows_that_it_can_hear_you(monkeypatch) -> None:
    """Checking the microphone is alive is exactly what you do before unmuting."""
    reporter = RecordingReporter()
    transcriber = metered_mic(monkeypatch, reporter)
    transcriber.set_muted(True)

    recorded_microphone_block(transcriber, np.array([0.5, -0.5], dtype=np.float32))

    assert [samples.tolist() for samples in reporter.blocks] == [[0.5, -0.5]]


def test_an_unmuted_microphone_is_heard_again(monkeypatch) -> None:
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    transcriber.set_muted(True)
    transcriber.set_muted(False)

    heard = recorded_microphone_block(
        transcriber, np.array([0.5, -0.5], dtype=np.float32)
    )

    assert [samples.tolist() for samples in heard] == [[0.5, -0.5]]


class FakeInputStream:
    def __init__(self):
        self.events: list[str] = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


def test_switching_microphones_reopens_only_the_input_stream(monkeypatch) -> None:
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    previous = FakeInputStream()
    replacement = FakeInputStream()
    transcriber._device = 1
    transcriber._sd_stream = previous
    transcriber._should_listen = True

    def open_replacement():
        transcriber._sd_stream = replacement
        replacement.start()

    transcriber._start_listening = open_replacement

    transcriber.switch_device(2)

    assert transcriber._device == 2
    assert transcriber._sd_stream is replacement
    assert previous.events == ["stop", "close"]
    assert replacement.events == ["start"]
    assert transcriber._should_listen is True


def test_a_microphone_that_cannot_open_restores_the_previous_stream(
    monkeypatch,
) -> None:
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    previous = FakeInputStream()
    rejected = FakeInputStream()
    transcriber._device = 1
    transcriber._sd_stream = previous
    transcriber._should_listen = True

    def reject_replacement():
        transcriber._sd_stream = rejected
        raise RuntimeError("device busy")

    transcriber._start_listening = reject_replacement

    with pytest.raises(RuntimeError, match="device busy"):
        transcriber.switch_device(2)

    assert transcriber._device == 1
    assert transcriber._sd_stream is previous
    assert previous.events == ["stop", "start"]
    assert rejected.events == ["close"]
    assert transcriber._should_listen is True


def test_releasing_a_microphone_stops_and_forgets_its_portaudio_stream(
    monkeypatch,
) -> None:
    """An abandoned stream is what made microphone switches crash later."""
    from tagalong.capture import release_microphone_input

    transcriber = metered_mic(monkeypatch, RecordingReporter())
    stream = FakeInputStream()
    transcriber._sd_stream = stream

    release_microphone_input(transcriber)

    assert stream.events == ["stop", "close"]
    assert transcriber._sd_stream is None


def test_releasing_a_microphone_without_a_stream_is_a_no_op(monkeypatch) -> None:
    from tagalong.capture import release_microphone_input

    transcriber = metered_mic(monkeypatch, RecordingReporter())
    transcriber._sd_stream = None

    release_microphone_input(transcriber)

    assert transcriber._sd_stream is None


def test_switching_without_a_prior_stream_opens_the_new_one(monkeypatch) -> None:
    """A channel that never opened still has to land on the requested device."""
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    replacement = FakeInputStream()
    transcriber._device = 1
    transcriber._sd_stream = None
    transcriber._should_listen = False

    def open_replacement():
        transcriber._sd_stream = replacement
        replacement.start()

    transcriber._start_listening = open_replacement

    transcriber.switch_device(2)

    assert transcriber._device == 2
    assert transcriber._sd_stream is replacement
    assert replacement.events == ["start"]
    assert transcriber._should_listen is True


def test_a_failed_switch_without_a_prior_stream_stays_closed(monkeypatch) -> None:
    transcriber = metered_mic(monkeypatch, RecordingReporter())
    transcriber._device = 1
    transcriber._sd_stream = None
    transcriber._should_listen = False

    def reject_replacement():
        raise RuntimeError("device busy")

    transcriber._start_listening = reject_replacement

    with pytest.raises(RuntimeError, match="device busy"):
        transcriber.switch_device(2)

    assert transcriber._device == 1
    assert transcriber._sd_stream is None
    assert transcriber._should_listen is True
