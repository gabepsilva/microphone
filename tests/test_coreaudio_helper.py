"""Pure boundaries around the Darwin Core Audio helper process."""

from __future__ import annotations

import ctypes
from typing import Any, cast

import numpy as np

from tagalong.coreaudio_helper import (
    AudioBuffer,
    AudioBufferList,
    AudioTimeStamp,
    CoreAudioTap,
    LinearResampler,
    sample_time_is_contiguous,
)


def test_sample_time_continuity_is_explicit() -> None:
    assert sample_time_is_contiguous(None, 10.0, 4)
    assert sample_time_is_contiguous(10.0, 14.0, 4)
    assert not sample_time_is_contiguous(10.0, 15.0, 4)


def test_resampler_converts_hal_rate_to_the_capture_rate() -> None:
    resampler = LinearResampler(48000, 16000)
    source = np.column_stack(
        (np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 100)
    )

    result = resampler.process(source)

    assert result.tolist() == [[0.0, 100.0], [3.0, 103.0], [6.0, 106.0]]

    continuation = resampler.process(
        np.column_stack(
            (np.arange(8, 14, dtype=np.float32), np.arange(108, 114, dtype=np.float32))
        )
    )
    assert continuation.tolist() == [[9.0, 109.0], [12.0, 112.0]]


def test_the_io_proc_converts_interleaved_float_audio_to_s16le() -> None:
    tap = CoreAudioTap(16000, "Browser")
    tap.running = True
    written: list[bytes] = []

    def write(data: bytes) -> None:
        written.append(data)

    tap._write_pcm = cast(Any, write)

    values = (ctypes.c_float * 4)(1.0, -1.0, 0.5, -0.5)
    buffers = AudioBufferList()
    buffers.mNumberBuffers = 1
    buffers.mBuffers[0] = AudioBuffer(
        2,
        ctypes.sizeof(values),
        ctypes.cast(values, ctypes.c_void_p),
    )
    timestamp = AudioTimeStamp()
    timestamp.mSampleTime = 20.0

    assert (
        tap._io_proc(
            0,
            ctypes.pointer(timestamp),
            ctypes.pointer(buffers),
            None,
            None,
            None,
            None,
        )
        == 0
    )

    result = np.frombuffer(written[0], dtype="<i2")
    assert result.tolist() == [32767, -32767, 16383, -16383]
    assert tap.previous_sample_time == 20.0
    assert tap.discontinuities == 0


def test_reconcile_without_an_application_releases_audio() -> None:
    tap = CoreAudioTap(16000, "Browser")
    released = []
    tap._destroy_audio = cast(Any, lambda: released.append(True))

    tap.reconcile(None)

    assert tap.application is None
    assert tap.object_ids == []
    assert released == [True]
