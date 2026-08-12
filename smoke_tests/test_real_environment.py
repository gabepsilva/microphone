"""Opt-in smoke tests for hardware and services that deterministic CI fakes."""

from __future__ import annotations

import asyncio
import fcntl
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress

import numpy as np
import pytest
import sounddevice

from tagalong.catalog import probe_codex_models
from tagalong.choosers import audio_outputs, input_devices
from tagalong.piper_tts import ensure_model
from tagalong.speech import EDGE, PIPER, default_voice
from tagalong.streams import ALL_APPLICATIONS

_FFPLAY_ARGS = [
    "-nodisp",
    "-autoexit",
    "-loglevel",
    "quiet",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=880:duration=8",
]
_DETACHED_FFPLAY = """
import os
import sys

if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
with open(sys.argv[1], "w", encoding="ascii") as pid_file:
    pid_file.write(str(os.getpid()))
os.execv(sys.argv[2], [sys.argv[2], *sys.argv[3:]])
"""


def _wait_for_playing_process(streams_coreaudio, pid: int, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(
            obj.get("pid") == pid and obj.get("playing")
            for obj in streams_coreaudio.graph()
        ):
            return
        time.sleep(0.1)
    raise AssertionError(f"process {pid} did not become a Core Audio output")


def _drain_tap(process) -> None:
    """Discard the helper's backlog so the next read samples current audio.

    The parent normally drains this pipe continuously. A test that reads it in
    bursts does not, so the pipe fills to its capacity -- 65536 bytes here,
    about a second at 16 kHz stereo int16 -- and every read then returns audio
    from a second ago. Worse, once ``_write_pcm`` blocks the stream advances
    only as fast as the test reads, so waiting longer moves the sample further
    into the past rather than closer to now.
    """
    assert process.stdout is not None
    descriptor = process.stdout.fileno()
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    fcntl.fcntl(descriptor, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        while process.stdout.read(65536):
            pass
    except BlockingIOError:
        pass
    finally:
        fcntl.fcntl(descriptor, fcntl.F_SETFL, flags)


def _read_tap_pcm(process, timeout: float = 5) -> np.ndarray:
    """Read one 50 ms window of live PCM, discarding any backlog first."""
    assert process.stdout is not None
    _drain_tap(process)
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout), "Core Audio IOProc produced no PCM"
    finally:
        selector.close()
    data = process.stdout.read(3200)
    assert data
    assert len(data) % 4 == 0
    return np.frombuffer(data, dtype="<i2")


def _stop_process(process) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=3)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _stop_pid(pid: int | None) -> None:
    if pid is None:
        return
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def _start_all_tap(streams_coreaudio, poll: float = 0.2):
    tap = streams_coreaudio.StreamTap(ALL_APPLICATIONS, poll=poll)
    process = subprocess.Popen(
        tap.command(16000),
        stdout=subprocess.PIPE,
        **tap.process_options(),
    )
    error = tap.wait_ready(process, timeout=5)
    assert error is None, error
    tap.attach(process)
    tap.start()
    return tap, process


def _start_detached_ffplay(player_path: str, pid_file) -> tuple[subprocess.Popen, int]:
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _DETACHED_FFPLAY,
            str(pid_file),
            player_path,
            *_FFPLAY_ARGS,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    launcher.wait(timeout=3)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if pid_file.exists():
            return launcher, int(pid_file.read_text(encoding="ascii"))
        time.sleep(0.1)
    raise AssertionError("detached ffplay did not publish its pid")


def test_the_default_microphone_records_real_samples() -> None:
    devices = input_devices()

    assert devices, "PortAudio found no microphone input"
    recording = sounddevice.rec(
        frames=1600,
        samplerate=16000,
        channels=1,
        dtype="float32",
        blocking=True,
    )
    assert recording.shape == (1600, 1)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="PipeWire output smoke test only applies to Linux",
)
def test_pipewire_exposes_at_least_one_playback_output() -> None:
    outputs = audio_outputs()

    assert outputs, "pactl found no output sink with a monitor"
    assert all(output["name"] and output["monitor"] for output in outputs)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Core Audio process-tap smoke test only applies to macOS",
)
def test_core_audio_process_tap_captures_a_real_playing_process() -> None:
    from tagalong import streams_coreaudio

    player_path = shutil.which("ffplay")
    if player_path is None:
        pytest.skip("ffplay is required to provide a real playing process")
    player = subprocess.Popen(
        [
            player_path,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process = None
    tap = None
    try:
        target = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            target = next(
                (
                    obj["application"]
                    for obj in streams_coreaudio.graph()
                    if obj.get("pid") == player.pid and obj.get("playing")
                ),
                None,
            )
            if target is not None:
                break
            time.sleep(0.1)
        assert target is not None, "ffplay did not become a Core Audio output"

        tap = streams_coreaudio.StreamTap(target)
        process = subprocess.Popen(
            tap.command(16000),
            stdout=subprocess.PIPE,
            **tap.process_options(),
        )
        error = tap.wait_ready(process, timeout=5)
        assert error is None, error
        tap.attach(process)
        tap.start()
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(5), "Core Audio IOProc produced no PCM"
        finally:
            selector.close()
        data = process.stdout.read(3200)
        assert data
        assert len(data) % 4 == 0
    finally:
        if tap is not None:
            tap.stop()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        if player.poll() is None:
            player.terminate()
            player.wait(timeout=3)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Core Audio global-tap smoke test only applies to macOS",
)
@pytest.mark.xfail(
    reason=(
        "Known failure, and the reason SUPPORTS_ALL is False in "
        "tagalong/streams_coreaudio.py. A tap's exclusion list names "
        "AudioObjectIDs, which do not exist until a player registers an "
        "output, so a player shorter than one control tick is captured "
        "before it can be excluded. Measured at 685.3 mean absolute PCM "
        "against a 128.0 floor on macOS 26.5.2. Kept running so a future "
        "attempt has the instrument that produced that number; an XPASS "
        "here means the exclusion held and the capability can be revisited."
    ),
    strict=False,
)
def test_core_audio_all_tap_captures_external_audio_and_excludes_own_audio(
    tmp_path,
) -> None:
    from tagalong import streams_coreaudio

    player_path = shutil.which("ffplay")
    if player_path is None:
        pytest.skip("ffplay is required to provide real playing processes")

    detached_pid = None
    launcher = None
    external_pid_file = tmp_path / "detached-ffplay.pid"
    self_player = None
    process = None
    tap = None
    try:
        launcher, detached_pid = _start_detached_ffplay(player_path, external_pid_file)
        assert not streams_coreaudio.started_here(detached_pid, own_pid=os.getpid())
        _wait_for_playing_process(streams_coreaudio, detached_pid)

        tap, process = _start_all_tap(streams_coreaudio, poll=1.0)

        external_pcm = _read_tap_pcm(process)
        assert np.mean(np.abs(external_pcm)) > 1000

        _stop_pid(detached_pid)
        time.sleep(1.2)
        quiet_pcm = _read_tap_pcm(process)
        quiet_level = float(np.mean(np.abs(quiet_pcm)))

        # A player shorter than one control tick, so it can be born, register
        # an output and be heard before the exclusion list is next rebuilt.
        # Sampled while it is still playing: reading after it exits measures
        # the silence that follows it and passes whatever the tap did.
        short_args = [*_FFPLAY_ARGS]
        short_args[-1] = "sine=frequency=880:duration=0.5"
        self_player = subprocess.Popen(
            [player_path, *short_args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert streams_coreaudio.started_here(self_player.pid, own_pid=os.getpid())
        self_pcm = _read_tap_pcm(process)
        assert self_player.poll() is None, "self player exited before it was sampled"
        self_level = float(np.mean(np.abs(self_pcm)))
        assert self_level <= max(128.0, quiet_level * 2 + 64)
    finally:
        if tap is not None:
            tap.stop()
        _stop_process(process)
        _stop_process(self_player)
        _stop_pid(detached_pid)
        _stop_process(launcher)


def test_the_installed_codex_cli_exposes_a_usable_model_catalog() -> None:
    options = probe_codex_models()

    assert options, "`codex debug models` exposed no usable models"
    assert all(option.slug and option.efforts for option in options)


def test_the_default_piper_voice_synthesizes_audio() -> None:
    import piper

    model = piper.PiperVoice.load(str(ensure_model(default_voice(PIPER))))
    chunks = list(model.synthesize("TagAlong smoke test."))

    assert chunks
    assert any(chunk.audio_int16_bytes for chunk in chunks)


async def _edge_audio() -> bytes:
    import edge_tts

    audio = bytearray()
    communicate = edge_tts.Communicate(
        "TagAlong smoke test.",
        default_voice(EDGE),
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def test_the_default_edge_voice_synthesizes_audio_over_the_network() -> None:
    assert asyncio.run(_edge_audio())
