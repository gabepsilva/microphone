"""Opt-in smoke tests for hardware and services that deterministic CI fakes."""

from __future__ import annotations

import asyncio
import selectors
import shutil
import subprocess
import sys
import time

import pytest
import sounddevice

from tagalong.catalog import probe_codex_models
from tagalong.choosers import audio_outputs, input_devices
from tagalong.piper_tts import ensure_model
from tagalong.speech import EDGE, PIPER, default_voice


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
        if player.poll() is None:
            player.terminate()
            player.wait(timeout=3)


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
