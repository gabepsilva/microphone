"""Deterministic tests for the injectable Core Audio property boundary."""

from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

from tagalong import coreaudio_native


class PropertyLibrary:
    """Small property reader fake at the ctypes boundary, not in the adapter."""

    def __init__(self, payload: bytes, *, size_status: int = 0, data_status: int = 0):
        self.payload = payload
        self.size_status = size_status
        self.data_status = data_status

    def AudioObjectGetPropertyDataSize(
        self, _object_id, _address, _qualifier_size, _qualifier, size
    ):
        if self.size_status:
            return self.size_status
        size._obj.value = len(self.payload)
        return 0

    def AudioObjectGetPropertyData(
        self,
        _object_id,
        _address,
        _qualifier_size,
        _qualifier,
        actual,
        data,
    ):
        if self.data_status:
            return self.data_status
        ctypes.memmove(data, self.payload, len(self.payload))
        actual._obj.value = len(self.payload)
        return 0


class NativeFunction:
    """Callable object with the attributes ctypes assigns to native symbols."""

    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def test_fourcc_and_core_audio_error_are_named() -> None:
    assert coreaudio_native.fourcc("glob") == int.from_bytes(b"glob", "big")

    error = coreaudio_native.CoreAudioError("read", -42)

    assert error.operation == "read"
    assert error.status == -42
    assert str(error) == "read failed with OSStatus -42"


def test_framework_fails_closed_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(coreaudio_native.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match=r"macOS 14\.2"):
        coreaudio_native.framework()


def test_property_readers_use_the_injectable_library() -> None:
    library = PropertyLibrary(b"*\x00\x00\x00")

    assert coreaudio_native.property_bytes(7, 8, library) == b"*\x00\x00\x00"
    assert coreaudio_native.property_uint32(7, 8, library) == 42
    assert coreaudio_native.property_bool(7, 8, library)

    false_library = PropertyLibrary(b"\x00\x00\x00\x00")
    assert not coreaudio_native.property_bool(7, 8, false_library)


def test_property_readers_report_native_errors_and_short_values() -> None:
    with pytest.raises(coreaudio_native.CoreAudioError, match="OSStatus 9"):
        coreaudio_native.property_bytes(7, 8, PropertyLibrary(b"", size_status=9))

    with pytest.raises(coreaudio_native.CoreAudioError, match="OSStatus 10"):
        coreaudio_native.property_bytes(7, 8, PropertyLibrary(b"", data_status=10))

    with pytest.raises(coreaudio_native.CoreAudioError, match="short property"):
        coreaudio_native.property_uint32(7, 8, PropertyLibrary(b"\x01"))


def test_property_string_handles_null_and_foundation_conversion(monkeypatch) -> None:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    assert (
        coreaudio_native.property_string(
            7, 8, PropertyLibrary(b"\x00" * (pointer_size - 1))
        )
        == ""
    )
    assert (
        coreaudio_native.property_string(7, 8, PropertyLibrary(b"\x00" * pointer_size))
        == ""
    )

    backing = ctypes.create_string_buffer(b"Foundation string")
    pointer = ctypes.addressof(backing).to_bytes(pointer_size, sys.byteorder)

    class Foundation:
        def __init__(self):
            self.result = True
            self.CFStringGetCString = NativeFunction(self._get_c_string)

        def _get_c_string(self, _value, output, _length, _encoding):
            if self.result:
                output.value = b"Foundation string"
            return self.result

    foundation = Foundation()
    monkeypatch.setattr(coreaudio_native.ctypes, "CDLL", lambda _path: foundation)

    assert coreaudio_native.property_string(7, 8, PropertyLibrary(pointer)) == (
        "Foundation string"
    )
    foundation.result = False
    assert coreaudio_native.property_string(7, 8, PropertyLibrary(pointer)) == ""


def test_process_name_handles_platform_and_libproc_results(monkeypatch) -> None:
    monkeypatch.setattr(coreaudio_native.sys, "platform", "linux")
    assert coreaudio_native.process_name(7) == ""

    monkeypatch.setattr(coreaudio_native.sys, "platform", "darwin")

    class Libproc:
        def __init__(self):
            self.result = 7
            self.proc_name = NativeFunction(self._proc_name)

        def _proc_name(self, _pid, output, _length):
            if self.result:
                output.value = b"Browser"
            return self.result

    libproc = Libproc()
    monkeypatch.setattr(coreaudio_native.ctypes, "CDLL", lambda _path: libproc)
    assert coreaudio_native.process_name(7) == "Browser"

    libproc.result = 0
    assert coreaudio_native.process_name(7) == ""

    def missing_library(_path):
        raise OSError("libproc missing")

    monkeypatch.setattr(coreaudio_native.ctypes, "CDLL", missing_library)
    assert coreaudio_native.process_name(7) == ""


def test_running_application_name_uses_appkit_when_available(monkeypatch) -> None:
    monkeypatch.setattr(coreaudio_native.sys, "platform", "linux")
    assert coreaudio_native.running_application_name(7) == ""

    monkeypatch.setattr(coreaudio_native.sys, "platform", "darwin")

    class Application:
        def localizedName(self):
            return "Browser"

    class RunningApplication:
        @staticmethod
        def runningApplicationWithProcessIdentifier_(_pid):
            return Application()

    monkeypatch.setitem(
        sys.modules,
        "AppKit",
        SimpleNamespace(NSRunningApplication=RunningApplication),
    )
    assert coreaudio_native.running_application_name(7) == "Browser"

    class NoApplication:
        @staticmethod
        def runningApplicationWithProcessIdentifier_(_pid):
            return None

    monkeypatch.setitem(
        sys.modules,
        "AppKit",
        SimpleNamespace(NSRunningApplication=NoApplication),
    )
    assert coreaudio_native.running_application_name(7) == ""

    monkeypatch.delitem(sys.modules, "AppKit")
    assert coreaudio_native.running_application_name(7) == ""


@pytest.mark.parametrize(
    ("bundle_id", "name", "binary", "expected"),
    [
        ("com.example.helper.renderer", "Example", "renderer", "com.example"),
        ("com.example", "Example", "renderer", "com.example"),
        ("", "Example", "renderer", "Example"),
        ("", "", "renderer", "renderer"),
    ],
)
def test_application_key_groups_helpers_and_falls_back(
    bundle_id: str, name: str, binary: str, expected: str
) -> None:
    assert coreaudio_native.application_key(bundle_id, name, binary) == expected


def test_process_objects_normalizes_and_skips_unusable_objects(monkeypatch) -> None:
    monkeypatch.setattr(coreaudio_native.sys, "platform", "linux")
    assert coreaudio_native.process_objects(PropertyLibrary(b"")) == []

    monkeypatch.setattr(coreaudio_native.sys, "platform", "darwin")
    ids = (11).to_bytes(4, "little") + (22).to_bytes(4, "little")
    ids += (33).to_bytes(4, "little") + (44).to_bytes(4, "little")

    def read_bytes(object_id, _selector, _library):
        assert object_id == coreaudio_native.AUDIO_OBJECT_SYSTEM
        return ids

    def read_pid(object_id, _selector, _library):
        if object_id == 22:
            raise coreaudio_native.CoreAudioError("pid", -1)
        return {11: 101, 33: 303, 44: 404}[object_id]

    def read_bundle(object_id, _selector, _library):
        return {
            11: "com.example.helper.renderer",
            33: "",
            44: "com.example",
        }[object_id]

    def read_playing(object_id, _selector, _library):
        return object_id == 11

    monkeypatch.setattr(coreaudio_native, "property_bytes", read_bytes)
    monkeypatch.setattr(coreaudio_native, "property_uint32", read_pid)
    monkeypatch.setattr(coreaudio_native, "property_string", read_bundle)
    monkeypatch.setattr(coreaudio_native, "property_bool", read_playing)
    monkeypatch.setattr(
        coreaudio_native,
        "process_name",
        lambda pid: {101: "renderer", 303: "", 404: "app"}[pid],
    )
    monkeypatch.setattr(coreaudio_native, "running_application_name", lambda _pid: "")

    assert coreaudio_native.process_objects(PropertyLibrary(b"")) == [
        {
            "id": 11,
            "pid": 101,
            "bundle_id": "com.example.helper.renderer",
            "application": "com.example",
            "binary": "renderer",
            "playing": True,
        },
        {
            "id": 44,
            "pid": 404,
            "bundle_id": "com.example",
            "application": "com.example",
            "binary": "app",
            "playing": False,
        },
    ]
