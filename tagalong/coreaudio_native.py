"""Low-level Core Audio property and process-object bindings.

This module contains the ctypes/PyObjC boundary only.  The deterministic
stream adapter remains in ``streams_coreaudio``; the real binding is exercised
by the opt-in macOS smoke test because a Linux fake cannot prove the CFString
dictionary and HAL ABI contract.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

AUDIO_OBJECT_SYSTEM = 1
AUDIO_OBJECT_SCOPE_GLOBAL = int.from_bytes(b"glob", "big")
AUDIO_OBJECT_ELEMENT_MAIN = 0
PROCESS_OBJECT_LIST = int.from_bytes(b"prs#", "big")
PROCESS_BUNDLE_ID = int.from_bytes(b"pbid", "big")
PROCESS_PID = int.from_bytes(b"ppid", "big")
PROCESS_RUNNING_OUTPUT = int.from_bytes(b"piro", "big")


def fourcc(value: str) -> int:
    """Encode an AudioObject selector or scope."""
    return int.from_bytes(value.encode("latin1"), "big")


class AudioObjectPropertyAddress(ctypes.Structure):
    """ctypes spelling of Core Audio's property address."""

    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class CoreAudioError(RuntimeError):
    """A named error from the Core Audio HAL boundary."""

    def __init__(self, operation: str, status: int):
        self.operation = operation
        self.status = int(status)
        super().__init__(f"{operation} failed with OSStatus {self.status}")


def framework():
    """Load Core Audio lazily so Linux can import the process port."""
    if sys.platform != "darwin":
        raise RuntimeError("Core Audio process-tap capture requires macOS 26 or newer.")
    library = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    library.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    library.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(AudioObjectPropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    library.AudioObjectGetPropertyData.restype = ctypes.c_int32
    return library


def property_bytes(object_id: int, selector: int, library=None) -> bytes:
    """Read one Core Audio property as its raw byte representation."""
    library = framework() if library is None else library
    address = AudioObjectPropertyAddress(
        selector, AUDIO_OBJECT_SCOPE_GLOBAL, AUDIO_OBJECT_ELEMENT_MAIN
    )
    size = ctypes.c_uint32()
    status = library.AudioObjectGetPropertyDataSize(
        object_id, ctypes.byref(address), 0, None, ctypes.byref(size)
    )
    if status:
        raise CoreAudioError(f"read property {selector}", status)
    data = ctypes.create_string_buffer(size.value)
    actual = ctypes.c_uint32(size.value)
    status = library.AudioObjectGetPropertyData(
        object_id,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(actual),
        data,
    )
    if status:
        raise CoreAudioError(f"read property {selector}", status)
    return data.raw[: actual.value]


def property_uint32(object_id: int, selector: int, library=None) -> int:
    """Read a little-endian ``AudioObjectID`` or process id."""
    data = property_bytes(object_id, selector, library)
    if len(data) < 4:
        raise CoreAudioError(f"short property {selector}", -1)
    return int.from_bytes(data[:4], "little")


def property_bool(object_id: int, selector: int, library=None) -> bool:
    """Read Core Audio's Boolean process state."""
    return bool(property_uint32(object_id, selector, library))


def property_string(object_id: int, selector: int, library=None) -> str:
    """Convert a Core Foundation string returned by a property query."""
    data = property_bytes(object_id, selector, library)
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    if len(data) < pointer_size:
        return ""
    value = ctypes.c_void_p.from_buffer_copy(data[:pointer_size])
    if not value.value:
        return ""
    foundation = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    foundation.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    foundation.CFStringGetCString.restype = ctypes.c_bool
    output = ctypes.create_string_buffer(256)
    if not foundation.CFStringGetCString(value, output, len(output), 0x08000100):
        return ""
    return output.value.decode("utf-8", "replace")


def process_name(pid: int) -> str:
    """Return a process name through libproc, or an empty string."""
    if sys.platform != "darwin" or pid <= 0:
        return ""
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_name.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        library.proc_name.restype = ctypes.c_int
        output = ctypes.create_string_buffer(256)
        if library.proc_name(pid, output, len(output)) <= 0:
            return ""
        return output.value.decode("utf-8", "replace")
    except OSError:
        return ""


def running_application_name(pid: int) -> str:
    """Resolve a GUI process name when AppKit can see one."""
    if sys.platform != "darwin" or pid <= 0:
        return ""
    try:
        appkit = __import__("AppKit", fromlist=["NSRunningApplication"])
        running_application: Any = appkit.NSRunningApplication
        application = running_application.runningApplicationWithProcessIdentifier_(pid)
    except (ImportError, OSError):
        return ""
    return str(application.localizedName() or "") if application else ""


def application_key(bundle_id: str, name: str, binary: str) -> str:
    """Group an app's Core Audio helper processes under its base bundle id."""
    if bundle_id:
        helper = bundle_id.find(".helper")
        if helper >= 0:
            return bundle_id[:helper]
        return bundle_id
    return name or binary


def process_objects(library=None):
    """Read the Core Audio process-object list into plain dictionaries."""
    if sys.platform != "darwin":
        return []
    library = framework() if library is None else library
    raw = property_bytes(AUDIO_OBJECT_SYSTEM, PROCESS_OBJECT_LIST, library)
    objects = []
    for offset in range(0, len(raw) - 3, 4):
        object_id = int.from_bytes(raw[offset : offset + 4], "little")
        try:
            pid = property_uint32(object_id, PROCESS_PID, library)
            bundle_id = property_string(object_id, PROCESS_BUNDLE_ID, library)
            playing = property_bool(object_id, PROCESS_RUNNING_OUTPUT, library)
        except CoreAudioError:
            continue
        binary = process_name(pid)
        application = application_key(bundle_id, running_application_name(pid), binary)
        if not application:
            continue
        objects.append(
            {
                "id": object_id,
                "pid": pid,
                "bundle_id": bundle_id,
                "application": application,
                "binary": binary,
                "playing": playing,
            }
        )
    return objects
