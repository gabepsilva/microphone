"""Opt-in smoke tests for hardware and services that deterministic CI fakes."""

from __future__ import annotations

import asyncio

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


def test_pipewire_exposes_at_least_one_playback_output() -> None:
    outputs = audio_outputs()

    assert outputs, "pactl found no output sink with a monitor"
    assert all(output["name"] and output["monitor"] for output in outputs)


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
