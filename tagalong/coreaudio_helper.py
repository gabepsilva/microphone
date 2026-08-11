"""Darwin helper process for a private Core Audio process tap.

This module is intentionally a subprocess boundary.  Core Audio invokes the
IOProc from a native audio thread; the helper converts its interleaved float32
buffers to the s16le frames the shared capture reader already consumes and
writes them to stdout.  The parent owns shutdown and continuously drains that
pipe, so a slow recognizer cannot stop the HAL callback from making progress.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import sys
import uuid
from contextlib import suppress
from typing import Any

import numpy as np

from . import streams_coreaudio


class AudioBuffer(ctypes.Structure):
    """The part of Core Audio's interleaved AudioBuffer we consume."""

    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class AudioBufferList(ctypes.Structure):
    """A one-buffer view; additional buffers are read by address arithmetic."""

    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", AudioBuffer * 1),
    ]


class AudioTimeStamp(ctypes.Structure):
    """Enough of an AudioTimeStamp to check the sample-time field."""

    _fields_ = [
        ("mSampleTime", ctypes.c_double),
        ("mHostTime", ctypes.c_uint64),
        ("mRateScalar", ctypes.c_double),
        ("mWordClockTime", ctypes.c_uint64),
        ("mSMPTETime", ctypes.c_byte * 24),
        ("mSMPTETimeFlags", ctypes.c_uint32),
        ("mFlags", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


IOProc = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.POINTER(AudioTimeStamp),
    ctypes.POINTER(AudioBufferList),
    ctypes.c_void_p,
    ctypes.POINTER(AudioBufferList),
    ctypes.POINTER(AudioTimeStamp),
    ctypes.c_void_p,
)


class LinearResampler:
    """Convert interleaved float frames between the HAL and capture rates."""

    def __init__(self, source_rate: float, target_rate: int, channels: int = 2):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.channels = channels
        self.step = source_rate / target_rate
        self.buffer = np.empty((0, channels), dtype=np.float32)
        self.position = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return as many linearly interpolated target-rate frames as available."""
        if self.source_rate == self.target_rate:
            return samples
        self.buffer = np.concatenate((self.buffer, samples), axis=0)
        if len(self.buffer) < 2:
            return np.empty((0, self.channels), dtype=np.float32)
        positions = np.arange(
            self.position, len(self.buffer) - 1, self.step, dtype=np.float64
        )
        if len(positions) == 0:
            return np.empty((0, self.channels), dtype=np.float32)
        indexes = positions.astype(np.intp)
        fraction = (positions - indexes).astype(np.float32)[:, None]
        output = self.buffer[indexes] * (1.0 - fraction)
        output += self.buffer[indexes + 1] * fraction
        self.position = positions[-1] + self.step
        consumed = min(int(self.position), len(self.buffer) - 1)
        if consumed:
            self.buffer = self.buffer[consumed:]
            self.position -= consumed
        return output


def _bindings() -> tuple[Any, Any, Any]:
    """Load Core Audio and configure the C signatures used by this helper."""
    streams_coreaudio.require_stream_capture()
    CoreAudio: Any = importlib.import_module("CoreAudio")
    objc: Any = importlib.import_module("objc")

    library = streams_coreaudio._framework()
    library.AudioHardwareCreateProcessTap.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.AudioHardwareCreateProcessTap.restype = ctypes.c_int32
    library.AudioHardwareDestroyProcessTap.argtypes = [ctypes.c_uint32]
    library.AudioHardwareDestroyProcessTap.restype = ctypes.c_int32
    library.AudioHardwareCreateAggregateDevice.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.AudioHardwareCreateAggregateDevice.restype = ctypes.c_int32
    library.AudioHardwareDestroyAggregateDevice.argtypes = [ctypes.c_uint32]
    library.AudioHardwareDestroyAggregateDevice.restype = ctypes.c_int32
    library.AudioObjectSetPropertyData.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(streams_coreaudio.AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    library.AudioObjectSetPropertyData.restype = ctypes.c_int32
    library.AudioDeviceCreateIOProcID.argtypes = [
        ctypes.c_uint32,
        IOProc,
        ctypes.c_void_p,
        ctypes.POINTER(IOProc),
    ]
    library.AudioDeviceCreateIOProcID.restype = ctypes.c_int32
    library.AudioDeviceStart.argtypes = [ctypes.c_uint32, IOProc]
    library.AudioDeviceStart.restype = ctypes.c_int32
    library.AudioDeviceStop.argtypes = [ctypes.c_uint32, IOProc]
    library.AudioDeviceStop.restype = ctypes.c_int32
    library.AudioDeviceDestroyIOProcID.argtypes = [ctypes.c_uint32, IOProc]
    library.AudioDeviceDestroyIOProcID.restype = ctypes.c_int32
    return CoreAudio, objc, library


def _key(value: bytes) -> str:
    """Decode a Core Audio dictionary key before handing it to Foundation."""
    return value.decode("ascii")


def sample_time_is_contiguous(previous, current, frames) -> bool:
    """Whether an IOProc callback follows the previous sample-time exactly."""
    return previous is None or current == previous + frames


def _default_output_uid(library) -> str:
    """Return the UID of the device that supplies the aggregate clock."""
    address = streams_coreaudio.AudioObjectPropertyAddress(
        streams_coreaudio._fourcc("dOut"),
        streams_coreaudio.AUDIO_OBJECT_SCOPE_GLOBAL,
        streams_coreaudio.AUDIO_OBJECT_ELEMENT_MAIN,
    )
    device = ctypes.c_uint32()
    size = ctypes.c_uint32(ctypes.sizeof(device))
    status = library.AudioObjectGetPropertyData(
        streams_coreaudio.AUDIO_OBJECT_SYSTEM,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(device),
    )
    if status:
        raise RuntimeError(
            f"Core Audio default output device could not be read (OSStatus {status})."
        )
    uid = streams_coreaudio._property_string(
        device.value, streams_coreaudio._fourcc("uid "), library
    )
    if not uid:
        raise RuntimeError("Core Audio default output device has no UID.")
    return uid


def _nominal_rate(device: int, library) -> float:
    """Read the aggregate's actual rate; hardware may reject 16 kHz."""
    address = streams_coreaudio.AudioObjectPropertyAddress(
        streams_coreaudio._fourcc("nsrt"),
        streams_coreaudio.AUDIO_OBJECT_SCOPE_GLOBAL,
        streams_coreaudio.AUDIO_OBJECT_ELEMENT_MAIN,
    )
    value = ctypes.c_double()
    size = ctypes.c_uint32(ctypes.sizeof(value))
    status = library.AudioObjectGetPropertyData(
        device,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(value),
    )
    if status or value.value <= 0:
        raise RuntimeError(
            "Core Audio aggregate device has no usable nominal sample rate "
            f"(OSStatus {status})."
        )
    return value.value


class CoreAudioTap:
    """Own one process tap, aggregate device, and native IOProc."""

    def __init__(self, samplerate: int, application: str):
        self.samplerate = samplerate
        self.application = application
        self.process_tap = 0
        self.aggregate_device = 0
        self.io_proc_id = None
        self.io_proc = None
        self.running = False
        self.previous_sample_time = None
        self.discontinuities = 0
        self.input_samplerate = None
        self.resampler = None
        self._CoreAudio: Any = None
        self._objc: Any = None
        self._library: Any = None

    def start(self):
        """Create the process tap and start its aggregate-device IOProc."""
        self._CoreAudio, self._objc, self._library = _bindings()
        self._rebuild()

    def reconcile(self, application: str | None):
        """Retarget only when the process-object set changed."""
        self.application = application
        if not application:
            self.object_ids = []
            self._destroy_audio()
            return
        object_ids = self._object_ids(application)
        current = getattr(self, "object_ids", None)
        if current == object_ids:
            return
        self._destroy_audio()
        self.object_ids = object_ids
        self._create_audio(object_ids)

    def stop(self):
        """Stop the IOProc before destroying the aggregate and tap."""
        self._destroy_audio()

    def _object_ids(self, application):
        return [
            obj["id"]
            for obj in streams_coreaudio._process_objects(self._library)
            if obj.get("application") == application
        ]

    def _rebuild(self):
        object_ids = self._object_ids(self.application)
        if not object_ids:
            raise RuntimeError(
                f"no active Core Audio process was found for {self.application!r}"
            )
        self.object_ids = object_ids
        self._create_audio(object_ids)

    def _create_audio(self, object_ids):
        CoreAudio, objc, library = self._CoreAudio, self._objc, self._library
        description = CoreAudio.CATapDescription.alloc().initStereoMixdownOfProcesses_(
            object_ids
        )
        description.setPrivate_(True)
        description.setMuteBehavior_(CoreAudio.CATapUnmuted)
        description.setName_(f"tagalong_tap_{os.getpid()}")
        tap_id = ctypes.c_uint32()
        status = library.AudioHardwareCreateProcessTap(
            ctypes.c_void_p(objc.pyobjc_id(description)), ctypes.byref(tap_id)
        )
        if status:
            raise RuntimeError(
                "Core Audio process-tap capture permission was denied or is "
                f"unavailable (AudioHardwareCreateProcessTap OSStatus {status})."
            )

        def key(name):
            return _key(getattr(CoreAudio, name))

        try:
            output_uid = _default_output_uid(library)
            config = {
                key("kAudioAggregateDeviceNameKey"): (
                    f"tagalong_aggregate_{os.getpid()}"
                ),
                key("kAudioAggregateDeviceUIDKey"): (
                    f"com.tagalong.aggregate.{os.getpid()}.{uuid.uuid4()}"
                ),
                key("kAudioAggregateDeviceMainSubDeviceKey"): output_uid,
                key("kAudioAggregateDeviceIsPrivateKey"): True,
                key("kAudioAggregateDeviceTapAutoStartKey"): True,
                key("kAudioAggregateDeviceTapListKey"): [
                    {
                        key("kAudioSubTapUIDKey"): str(description.UUID()),
                        key("kAudioSubTapDriftCompensationKey"): True,
                    }
                ],
                key("kAudioAggregateDeviceSubDeviceListKey"): [
                    {key("kAudioSubDeviceUIDKey"): output_uid}
                ],
            }
            foundation = __import__("Foundation", fromlist=["NSDictionary"])
            dictionary_class: Any = foundation.NSDictionary

            dictionary = dictionary_class.dictionaryWithDictionary_(config)
            aggregate_id = ctypes.c_uint32()
            status = library.AudioHardwareCreateAggregateDevice(
                ctypes.c_void_p(objc.pyobjc_id(dictionary)), ctypes.byref(aggregate_id)
            )
            if status:
                raise RuntimeError(
                    "Core Audio aggregate capture device could not be created "
                    f"(OSStatus {status})."
                )

            self.process_tap = tap_id.value
            self.aggregate_device = aggregate_id.value
            self.input_samplerate = _nominal_rate(self.aggregate_device, library)
            self.resampler = LinearResampler(
                self.input_samplerate, self.samplerate, channels=2
            )
            self.io_proc = IOProc(self._io_proc)
            self.io_proc_id = IOProc()
            status = library.AudioDeviceCreateIOProcID(
                self.aggregate_device,
                self.io_proc,
                None,
                ctypes.byref(self.io_proc_id),
            )
            if status:
                raise RuntimeError(
                    f"Core Audio IOProc could not be created (OSStatus {status})."
                )
            status = library.AudioDeviceStart(self.aggregate_device, self.io_proc_id)
            if status:
                raise RuntimeError(
                    f"Core Audio aggregate device could not start (OSStatus {status})."
                )
            self.running = True
        except Exception:
            if not self.process_tap:
                with suppress(Exception):
                    library.AudioHardwareDestroyProcessTap(tap_id.value)
            self._destroy_audio()
            raise

    @staticmethod
    def _interleaved_samples(input_data):
        """Normalize interleaved or non-interleaved HAL buffers to stereo."""
        channel_buffers = []
        number = input_data.contents.mNumberBuffers
        for index in range(number):
            address = ctypes.addressof(
                input_data.contents.mBuffers
            ) + index * ctypes.sizeof(AudioBuffer)
            buffer = AudioBuffer.from_address(address)
            if buffer.mData is None or buffer.mDataByteSize == 0:
                continue
            samples = buffer.mDataByteSize // ctypes.sizeof(ctypes.c_float)
            channels = max(1, buffer.mNumberChannels)
            values = np.ctypeslib.as_array(
                ctypes.cast(buffer.mData, ctypes.POINTER(ctypes.c_float)),
                shape=(samples,),
            )
            channel_buffers.append(values.reshape(-1, channels))
        if not channel_buffers:
            return np.empty((0, 2), dtype=np.float32), 0
        frames = min(len(buffer) for buffer in channel_buffers)
        if len(channel_buffers) == 1:
            interleaved = channel_buffers[0][:frames, :2]
        else:
            interleaved = np.column_stack(
                [buffer[:frames, 0] for buffer in channel_buffers[:2]]
            )
        if interleaved.shape[1] == 1:
            interleaved = np.repeat(interleaved, 2, axis=1)
        return interleaved, frames

    def _io_proc(
        self,
        _device,
        timestamp,
        input_data,
        _input_time,
        _output_data,
        _output_time,
        _client_data,
    ):
        """Convert one interleaved float32 callback into frame-aligned s16le."""
        if not self.running or not input_data:
            return 0
        try:
            interleaved, frames = self._interleaved_samples(input_data)
            resampler = self.resampler
            if resampler is not None:
                interleaved = resampler.process(interleaved)
            if len(interleaved):
                pcm = np.clip(interleaved, -1.0, 1.0) * 32767.0
                self._write_pcm(pcm.astype("<i2", copy=False).tobytes())
            if timestamp and frames:
                current = timestamp.contents.mSampleTime
                if not sample_time_is_contiguous(
                    self.previous_sample_time, current, frames
                ):
                    self.discontinuities += 1
                self.previous_sample_time = current
        except (BrokenPipeError, OSError):
            self.running = False
        return 0

    @staticmethod
    def _write_pcm(data: bytes):
        """Write every frame, because one os.write may accept a short prefix."""
        file_descriptor = sys.stdout.fileno()
        view = memoryview(data)
        while view:
            view = view[os.write(file_descriptor, view) :]

    def _destroy_audio(self):
        """Make all cleanup idempotent for startup failures and SIGTERM."""
        if self._library is None:
            return
        self.running = False
        if self.aggregate_device and self.io_proc_id is not None:
            with suppress(Exception):
                self._library.AudioDeviceStop(self.aggregate_device, self.io_proc_id)
            with suppress(Exception):
                self._library.AudioDeviceDestroyIOProcID(
                    self.aggregate_device, self.io_proc_id
                )
        self.io_proc_id = None
        self.io_proc = None
        if self.aggregate_device:
            with suppress(Exception):
                self._library.AudioHardwareDestroyAggregateDevice(self.aggregate_device)
            self.aggregate_device = 0
        if self.process_tap:
            with suppress(Exception):
                self._library.AudioHardwareDestroyProcessTap(self.process_tap)
            self.process_tap = 0


def _redirect_stderr():
    """Discard post-readiness diagnostics so the helper cannot fill a pipe."""
    sys.stderr.flush()
    descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--samplerate", type=int, required=True)
    parser.add_argument("--application", required=True)
    return parser.parse_args(argv)


def run(argv=None) -> int:
    """Run until stdin closes, returning a named nonzero startup/runtime code."""
    args = _arguments(argv)
    tap = CoreAudioTap(args.samplerate, args.application)
    try:
        tap.start()
        print("READY", file=sys.stderr, flush=True)
        _redirect_stderr()
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            application = request.get("application")
            if application != tap.application:
                tap.reconcile(application)
        return 0
    except Exception as error:
        print(f"ERROR {error}", file=sys.stderr, flush=True)
        return 64
    finally:
        tap.stop()


def main() -> None:
    """Run the helper as a module entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
