"""Pure boundaries around the Darwin Core Audio helper process."""

from __future__ import annotations

import ctypes
import io
import sys
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest

import tagalong.coreaudio_helper as helper
from tagalong import streams_coreaudio
from tagalong.coreaudio_helper import (
    AudioBuffer,
    AudioBufferList,
    AudioTimeStamp,
    CoreAudioTap,
    LinearResampler,
    sample_time_is_contiguous,
)
from tagalong.streams import ALL_APPLICATIONS


class NativeSymbol:
    """Callable fake with the attributes ctypes assigns to native symbols."""

    def __init__(self, function=None):
        self.function = function or (lambda *_args: 0)
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


class FakeHAL:
    """A fake Core Audio library at the HAL boundary used by CoreAudioTap."""

    def __init__(
        self,
        *,
        tap_status=0,
        aggregate_status=0,
        io_status=0,
        start_status=0,
        rate=48000.0,
    ):
        self.tap_status = tap_status
        self.aggregate_status = aggregate_status
        self.io_status = io_status
        self.start_status = start_status
        self.rate = rate
        self.AudioHardwareCreateProcessTap = NativeSymbol(self._create_tap)
        self.AudioHardwareDestroyProcessTap = NativeSymbol()
        self.AudioHardwareCreateAggregateDevice = NativeSymbol(self._create_aggregate)
        self.AudioHardwareDestroyAggregateDevice = NativeSymbol()
        self.AudioObjectSetPropertyData = NativeSymbol()
        self.AudioDeviceCreateIOProcID = NativeSymbol(lambda *_args: self.io_status)
        self.AudioDeviceStart = NativeSymbol(lambda *_args: self.start_status)
        self.AudioDeviceStop = NativeSymbol()
        self.AudioDeviceDestroyIOProcID = NativeSymbol()
        self.AudioObjectGetPropertyData = NativeSymbol(self._read_property)
        self.AudioObjectGetPropertyDataSize = NativeSymbol()

    def _create_tap(self, _description, result):
        result._obj.value = 501
        return self.tap_status

    def _create_aggregate(self, _dictionary, result):
        result._obj.value = 601
        return self.aggregate_status

    def _read_property(
        self, _object_id, address, _qualifier_size, _qualifier, size, data
    ):
        selector = address._obj.mSelector
        if selector == streams_coreaudio._fourcc("dOut"):
            data._obj.value = 701
        elif selector == streams_coreaudio._fourcc("nsrt"):
            data._obj.value = self.rate
        size._obj.value = ctypes.sizeof(data._obj)
        return 0


class FakeDescription:
    def initStereoGlobalTapButExcludeProcesses_(self, object_ids):
        self.initializer = "global"
        self.object_ids = object_ids
        return self

    def initStereoMixdownOfProcesses_(self, object_ids):
        self.initializer = "named"
        self.object_ids = object_ids
        return self

    def setPrivate_(self, value):
        self.private = value

    def setMuteBehavior_(self, value):
        self.mute_behavior = value

    def setName_(self, value):
        self.name = value

    def UUID(self):
        return "tap-uuid"


class FakeDescriptionClass:
    instances: ClassVar[list[FakeDescription]] = []

    @classmethod
    def alloc(cls):
        instance = FakeDescription()
        cls.instances.append(instance)
        return instance


class FakeDictionary:
    @staticmethod
    def dictionaryWithDictionary_(value):
        return value


def fake_native_objects():
    names = (
        "kAudioAggregateDeviceNameKey",
        "kAudioAggregateDeviceUIDKey",
        "kAudioAggregateDeviceMainSubDeviceKey",
        "kAudioAggregateDeviceIsPrivateKey",
        "kAudioAggregateDeviceTapAutoStartKey",
        "kAudioAggregateDeviceTapListKey",
        "kAudioAggregateDeviceSubDeviceListKey",
        "kAudioSubTapUIDKey",
        "kAudioSubTapDriftCompensationKey",
        "kAudioSubDeviceUIDKey",
    )
    core_audio = SimpleNamespace(
        CATapDescription=FakeDescriptionClass,
        CATapUnmuted=1,
        **{name: name.encode("ascii") for name in names},
    )
    objc = SimpleNamespace(pyobjc_id=lambda value: id(value))
    return core_audio, objc


def install_fake_tap_environment(monkeypatch):
    core_audio, objc = fake_native_objects()
    monkeypatch.setattr(
        helper.streams_coreaudio,
        "_property_string",
        lambda *_args: "BuiltInOutput",
    )
    monkeypatch.setitem(
        sys.modules,
        "Foundation",
        SimpleNamespace(NSDictionary=FakeDictionary),
    )
    monkeypatch.setattr(
        helper.streams_coreaudio,
        "_process_objects",
        lambda _library: [
            {"id": 11, "application": "Browser"},
            {"id": 12, "application": "Other"},
        ],
    )
    return core_audio, objc


def test_sample_time_continuity_is_explicit() -> None:
    assert sample_time_is_contiguous(None, 10.0, 4)
    assert sample_time_is_contiguous(10.0, 14.0, 4)
    assert not sample_time_is_contiguous(10.0, 15.0, 4)


def test_bindings_configures_the_native_symbols(monkeypatch) -> None:
    library = FakeHAL()
    core_audio, objc = fake_native_objects()
    modules = {"CoreAudio": core_audio, "objc": objc}

    monkeypatch.setattr(
        helper.streams_coreaudio, "require_stream_capture", lambda: None
    )
    monkeypatch.setattr(
        helper.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(helper.streams_coreaudio, "_framework", lambda: library)

    result = helper._bindings()

    assert result == (core_audio, objc, library)
    assert library.AudioHardwareCreateProcessTap.restype is ctypes.c_int32
    assert cast(Any, library.AudioDeviceCreateIOProcID.argtypes)[1] is helper.IOProc


def test_tap_lifecycle_reconciles_ids_and_cleans_up(monkeypatch) -> None:
    library = FakeHAL()
    core_audio, objc = install_fake_tap_environment(monkeypatch)
    monkeypatch.setattr(helper, "_bindings", lambda: (core_audio, objc, library))

    tap = CoreAudioTap(16000, "Browser")
    tap.start()

    assert tap.running
    assert tap.object_ids == [11]
    assert tap.input_samplerate == 48000.0
    assert tap.resampler is not None
    assert len(library.AudioHardwareCreateProcessTap.calls) == 1

    tap.reconcile("Browser")
    assert len(library.AudioHardwareCreateProcessTap.calls) == 1

    monkeypatch.setattr(
        helper.streams_coreaudio,
        "_process_objects",
        lambda _library: [{"id": 13, "application": "Browser"}],
    )
    tap.reconcile("Browser")
    assert tap.object_ids == [13]
    assert len(library.AudioHardwareCreateProcessTap.calls) == 2

    tap.stop()
    tap.stop()
    assert not tap.running
    assert tap.aggregate_device == 0
    assert tap.process_tap == 0
    assert library.AudioDeviceStop.calls
    assert library.AudioHardwareDestroyAggregateDevice.calls
    assert library.AudioHardwareDestroyProcessTap.calls


def test_all_tap_uses_global_initializer_and_refreshes_own_exclusions(
    monkeypatch,
) -> None:
    library = FakeHAL()
    core_audio, objc = install_fake_tap_environment(monkeypatch)
    monkeypatch.setattr(helper, "_bindings", lambda: (core_audio, objc, library))
    processes = [
        {"id": 11, "pid": 101},
        {"id": 12, "pid": 202},
    ]
    monkeypatch.setattr(
        helper.streams_coreaudio, "_process_objects", lambda _library: processes
    )
    monkeypatch.setattr(helper.os, "getppid", lambda: 900)
    monkeypatch.setattr(
        helper.streams_coreaudio,
        "started_here",
        lambda pid, *, own_pid: own_pid == 900 and pid == 101,
    )

    tap = CoreAudioTap(16000, ALL_APPLICATIONS)
    tap.start()

    assert tap.object_ids == [11]
    assert FakeDescriptionClass.instances[-1].initializer == "global"
    assert FakeDescriptionClass.instances[-1].object_ids == [11]
    assert len(library.AudioHardwareCreateProcessTap.calls) == 1

    processes.append({"id": 13, "pid": 303})
    tap.reconcile(ALL_APPLICATIONS)
    assert len(library.AudioHardwareCreateProcessTap.calls) == 1

    processes[:] = [{"id": 14, "pid": 101}]
    tap.reconcile(ALL_APPLICATIONS)
    assert tap.object_ids == [14]
    assert len(library.AudioHardwareCreateProcessTap.calls) == 2
    tap.stop()


def test_all_tap_allows_an_empty_exclusion_list(monkeypatch) -> None:
    library = FakeHAL()
    core_audio, objc = install_fake_tap_environment(monkeypatch)
    monkeypatch.setattr(helper, "_bindings", lambda: (core_audio, objc, library))
    monkeypatch.setattr(helper.streams_coreaudio, "_process_objects", lambda _: [])

    tap = CoreAudioTap(16000, ALL_APPLICATIONS)
    tap.start()

    assert tap.object_ids == []
    assert FakeDescriptionClass.instances[-1].initializer == "global"
    assert FakeDescriptionClass.instances[-1].object_ids == []
    tap.stop()


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("tap_status", "process-tap capture permission"),
        ("aggregate_status", "aggregate capture device"),
        ("io_status", "IOProc could not be created"),
        ("start_status", "aggregate device could not start"),
    ],
)
def test_tap_reports_startup_failures_and_releases_native_handles(
    monkeypatch, attribute: str, message: str
) -> None:
    library = FakeHAL()
    setattr(library, attribute, 1)
    core_audio, objc = install_fake_tap_environment(monkeypatch)
    tap = CoreAudioTap(16000, "Browser")
    tap._CoreAudio = core_audio
    tap._objc = objc
    tap._library = library

    with pytest.raises(RuntimeError, match=message):
        tap._create_audio([11])

    assert not tap.running
    assert tap.aggregate_device == 0
    assert tap.process_tap == 0


def test_default_output_and_nominal_rate_report_native_errors(monkeypatch) -> None:
    library = FakeHAL()
    monkeypatch.setattr(
        helper.streams_coreaudio,
        "_property_string",
        lambda *_args: "",
    )
    with pytest.raises(RuntimeError, match="has no UID"):
        helper._default_output_uid(library)

    library.AudioObjectGetPropertyData = NativeSymbol(lambda *_args: 9)
    with pytest.raises(RuntimeError, match="could not be read"):
        helper._default_output_uid(library)

    bad_rate = FakeHAL(rate=0)
    with pytest.raises(RuntimeError, match="no usable nominal sample rate"):
        helper._nominal_rate(601, bad_rate)


def test_rebuild_requires_a_matching_process() -> None:
    tap = CoreAudioTap(16000, "Browser")
    tap._object_ids = cast(Any, lambda _application: [])

    with pytest.raises(RuntimeError, match="no active Core Audio process"):
        tap._rebuild()


def buffer_list(rows):
    """Allocate a variable-length AudioBufferList for the callback normalizer."""
    size = ctypes.sizeof(AudioBufferList) + max(0, len(rows) - 1) * ctypes.sizeof(
        AudioBuffer
    )
    storage = ctypes.create_string_buffer(size)
    pointer = ctypes.cast(storage, ctypes.POINTER(AudioBufferList))
    pointer.contents.mNumberBuffers = len(rows)
    data_arrays = []
    for index, row in enumerate(rows):
        address = ctypes.addressof(pointer.contents.mBuffers) + index * ctypes.sizeof(
            AudioBuffer
        )
        target = AudioBuffer.from_address(address)
        if row is None:
            target.mNumberChannels = 1
            target.mDataByteSize = 0
            target.mData = None
            continue
        channels, values = row
        data = (ctypes.c_float * len(values))(*values)
        data_arrays.append(data)
        target.mNumberChannels = channels
        target.mDataByteSize = ctypes.sizeof(data)
        target.mData = ctypes.cast(data, ctypes.c_void_p)
    return pointer, data_arrays, storage


def test_buffer_normalizer_handles_empty_mono_and_noninterleaved_audio() -> None:
    tap = CoreAudioTap(16000, "Browser")
    empty, _, empty_storage = buffer_list([None])
    samples, frames = tap._interleaved_samples(empty)
    assert empty_storage
    assert samples.shape == (0, 2)
    assert frames == 0

    mono, _, mono_storage = buffer_list([(1, [0.25, -0.25])])
    samples, frames = tap._interleaved_samples(mono)
    assert mono_storage
    assert frames == 2
    assert samples.tolist() == [[0.25, 0.25], [-0.25, -0.25]]

    left_right, _, multi_storage = buffer_list([(1, [0.25, 0.5]), (1, [-0.25, -0.5])])
    samples, frames = tap._interleaved_samples(left_right)
    assert multi_storage
    assert frames == 2
    assert samples.tolist() == [[0.25, -0.25], [0.5, -0.5]]


def test_io_proc_handles_discontinuity_and_broken_output() -> None:
    tap = CoreAudioTap(16000, "Browser")
    tap.running = True
    tap.resampler = None
    written = []
    tap._write_pcm = cast(Any, lambda data: written.append(data))
    data, _, storage = buffer_list([(2, [0.25, -0.25, 0.5, -0.5])])
    first = AudioTimeStamp()
    first.mSampleTime = 20.0
    second = AudioTimeStamp()
    second.mSampleTime = 25.0

    assert tap._io_proc(0, ctypes.pointer(first), data, None, None, None, None) == 0
    assert tap._io_proc(0, ctypes.pointer(second), data, None, None, None, None) == 0
    assert storage
    assert len(written) == 2
    assert tap.discontinuities == 1

    def broken(_data):
        raise BrokenPipeError

    tap._write_pcm = cast(Any, broken)
    assert tap._io_proc(0, ctypes.pointer(second), data, None, None, None, None) == 0
    assert not tap.running

    assert tap._io_proc(0, ctypes.pointer(second), None, None, None, None, None) == 0


def test_write_pcm_retries_short_os_writes(monkeypatch) -> None:
    monkeypatch.setattr(helper.sys, "stdout", SimpleNamespace(fileno=lambda: 9))
    writes = []

    def write(_fd, view):
        writes.append(bytes(view))
        return 1

    monkeypatch.setattr(helper.os, "write", write)
    CoreAudioTap._write_pcm(b"abc")

    assert writes == [b"abc", b"bc", b"c"]


def test_redirect_stderr_closes_the_devnull_descriptor(monkeypatch) -> None:
    events = []
    stderr = SimpleNamespace(flush=lambda: events.append("flush"), fileno=lambda: 2)
    monkeypatch.setattr(helper.sys, "stderr", stderr)
    monkeypatch.setattr(
        helper.os, "open", lambda path, flags: events.append((path, flags)) or 17
    )
    monkeypatch.setattr(
        helper.os, "dup2", lambda source, target: events.append((source, target))
    )
    monkeypatch.setattr(
        helper.os, "close", lambda descriptor: events.append(("close", descriptor))
    )

    helper._redirect_stderr()

    assert events == [
        "flush",
        (helper.os.devnull, helper.os.O_WRONLY),
        (17, 2),
        ("close", 17),
    ]


def test_arguments_and_run_retarget_the_helper(monkeypatch) -> None:
    assert helper._arguments(
        ["--samplerate", "16000", "--application", "Browser"]
    ).application == ("Browser")

    class RunningTap:
        instances: ClassVar[list] = []

        def __init__(self, _samplerate, application):
            self.samplerate = _samplerate
            self.application = application
            self.retargeted = []
            self.started = False
            self.stopped = False
            self.instances.append(self)

        def start(self):
            self.started = True

        def reconcile(self, application):
            self.retargeted.append(application)
            self.application = application

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(helper, "CoreAudioTap", RunningTap)
    monkeypatch.setattr(helper, "_redirect_stderr", lambda: None)
    monkeypatch.setattr(
        helper.sys,
        "stdin",
        io.StringIO('\n{"application":"Other"}\n{"application":"Other"}\n'),
    )
    monkeypatch.setattr(helper.sys, "stderr", io.StringIO())

    assert helper.run(["--samplerate", "16000", "--application", "Browser"]) == 0
    tap = RunningTap.instances[-1]
    assert tap.started
    assert tap.retargeted == ["Other"]
    assert tap.stopped


def test_run_reconciles_all_selection_on_every_control_tick(monkeypatch) -> None:
    class RunningTap:
        instances: ClassVar[list] = []

        def __init__(self, _samplerate, application):
            self.application = application
            self.retargeted = []
            self.stopped = False
            self.instances.append(self)

        def start(self):
            pass

        def reconcile(self, application):
            self.retargeted.append(application)
            self.application = application

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(helper, "CoreAudioTap", RunningTap)
    monkeypatch.setattr(helper, "_redirect_stderr", lambda: None)
    monkeypatch.setattr(
        helper.sys,
        "stdin",
        io.StringIO(
            f'{{"application":"{ALL_APPLICATIONS}"}}\n'
            f'{{"application":"{ALL_APPLICATIONS}"}}\n'
        ),
    )
    monkeypatch.setattr(helper.sys, "stderr", io.StringIO())

    assert helper.run(["--samplerate", "16000", "--application", ALL_APPLICATIONS]) == 0
    tap = RunningTap.instances[-1]
    assert tap.retargeted == [ALL_APPLICATIONS, ALL_APPLICATIONS]
    assert tap.stopped


def test_run_reports_startup_failure_and_main_exits(monkeypatch) -> None:
    class FailingTap:
        def __init__(self, _samplerate, _application):
            self.stopped = False

        def start(self):
            raise RuntimeError("TCC denied")

        def stop(self):
            self.stopped = True

    tap = FailingTap(16000, "Browser")
    monkeypatch.setattr(helper, "CoreAudioTap", lambda *_args: tap)
    stderr = io.StringIO()
    monkeypatch.setattr(helper.sys, "stderr", stderr)

    assert helper.run(["--samplerate", "16000", "--application", "Browser"]) == 64
    assert "ERROR TCC denied" in stderr.getvalue()
    assert tap.stopped

    monkeypatch.setattr(helper, "run", lambda: 7)
    with pytest.raises(SystemExit) as raised:
        helper.main()
    assert raised.value.code == 7


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
