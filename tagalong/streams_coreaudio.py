"""macOS Core Audio process-object discovery and tap control.

The parent process owns only the small control surface.  A Darwin helper
process owns the Core Audio tap, private aggregate device, and IOProc so a
blocked HAL callback cannot wedge the session that is driving the UI.  The
helper's stdout remains the same frame-aligned s16le stream as PipeWire;
startup readiness and diagnostics use stderr, and selection changes use
newline-delimited JSON on stdin.
"""

from __future__ import annotations

import json
import platform
import selectors
import subprocess
import sys
import threading

from . import coreaudio_native
from .session_darwin import started_here
from .streams import ApplicationStream

CHANNELS = 2
POLL_SECONDS = 1.0
HELPER_MODULE = "tagalong.coreaudio_helper"
# Apple's process-tap API is available from macOS 14.2. This floor is higher on
# purpose: 26.x is the only version this backend has been run on, and the gate
# claims what was measured rather than what the API docs allow. Lower it to
# (14, 2) once a real 14.x host has passed the smoke probe in smoke_tests.
MINIMUM_MACOS = (26, 0)
READY = "READY"


# These selectors are stable C constants.  Keeping their values here means
# importing the platform adapter on Linux remains safe; the framework is
# loaded only by a Darwin process that actually needs a graph or a tap.
AudioObjectPropertyAddress = coreaudio_native.AudioObjectPropertyAddress
CoreAudioError = coreaudio_native.CoreAudioError
_framework = coreaudio_native.framework
_property_bytes = coreaudio_native.property_bytes
_property_uint32 = coreaudio_native.property_uint32
_property_bool = coreaudio_native.property_bool
_property_string = coreaudio_native.property_string
process_name = coreaudio_native.process_name
_running_application_name = coreaudio_native.running_application_name
_application_key = coreaudio_native.application_key
_process_objects = coreaudio_native.process_objects
_fourcc = coreaudio_native.fourcc

AUDIO_OBJECT_SYSTEM = coreaudio_native.AUDIO_OBJECT_SYSTEM
AUDIO_OBJECT_SCOPE_GLOBAL = coreaudio_native.AUDIO_OBJECT_SCOPE_GLOBAL
AUDIO_OBJECT_ELEMENT_MAIN = coreaudio_native.AUDIO_OBJECT_ELEMENT_MAIN
PROCESS_OBJECT_LIST = coreaudio_native.PROCESS_OBJECT_LIST
PROCESS_BUNDLE_ID = coreaudio_native.PROCESS_BUNDLE_ID
PROCESS_PID = coreaudio_native.PROCESS_PID
PROCESS_RUNNING_OUTPUT = coreaudio_native.PROCESS_RUNNING_OUTPUT


def graph(read_objects=_process_objects):
    """Return one opaque Core Audio process snapshot."""
    try:
        return read_objects()
    except (CoreAudioError, OSError, RuntimeError):
        return []


def application_streams(objects, mine=started_here):
    """Normalize process objects into the shared application-stream shape."""
    streams = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        pid = obj.get("pid")
        if isinstance(pid, int) and mine(pid):
            continue
        application = obj.get("application") or obj.get("bundle_id")
        node_id = obj.get("id")
        if not application or not isinstance(node_id, int):
            continue
        streams.append(
            ApplicationStream(
                node_id=node_id,
                application=str(application),
                title=str(obj.get("title") or application),
                binary=str(obj.get("binary") or ""),
                playing=bool(obj.get("playing")),
            )
        )
    return streams


def applications(streams):
    """Collapse helper processes into one selectable application."""
    collapsed = {}
    for stream in streams:
        found = collapsed.get(stream.application)
        collapsed[stream.application] = ApplicationStream(
            node_id=stream.node_id,
            application=stream.application,
            title=stream.title,
            binary=stream.binary,
            playing=stream.playing or (found is not None and found.playing),
        )
    return sorted(
        collapsed.values(), key=lambda stream: (not stream.playing, stream.application)
    )


def offered_applications(objects, mine=started_here):
    """Return picker labels and stable application names."""
    from .streams import stream_label

    return [
        (stream_label(stream), stream.application)
        for stream in applications(application_streams(objects, mine=mine))
    ]


def _macos_version() -> tuple[int, ...]:
    """Return the host's numeric macOS version for the process-tap gate."""
    version = platform.mac_ver()[0]
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def require_stream_capture():
    """Fail at point of use with the platform's actual capability message."""
    if sys.platform != "darwin":
        raise RuntimeError("Core Audio process-tap capture requires macOS 26 or newer.")
    if _macos_version() < MINIMUM_MACOS:
        raise RuntimeError(
            "Core Audio process-tap capture is only verified on macOS 26 or "
            "newer; this host reports macOS "
            f"{platform.mac_ver()[0] or 'an unknown version'}."
        )
    try:
        _framework()
    except OSError as error:
        raise RuntimeError(
            "Core Audio process-tap capture needs the macOS Core Audio framework."
        ) from error


class StreamTap:
    """Control one helper process that owns a private Core Audio tap."""

    CHANNELS = CHANNELS
    POLL_SECONDS = POLL_SECONDS

    def __init__(self, application=None, poll=POLL_SECONDS, executable=sys.executable):
        self.application = application
        self.poll = poll
        self.executable = executable
        self.process = None
        self.control = None
        self.stopping = threading.Event()
        self.watcher = None

    def command(self, samplerate):
        """Build the helper command without importing macOS bindings."""
        command = [
            self.executable,
            "-m",
            HELPER_MODULE,
            "--samplerate",
            str(samplerate),
        ]
        if self.application is not None:
            command.extend(("--application", self.application))
        return command

    def process_options(self):
        """Request control stdin and startup-only diagnostics stderr."""
        return {"stdin": subprocess.PIPE, "stderr": subprocess.PIPE}

    def wait_ready(self, process, timeout):
        """Read one bounded READY/error line, then close stderr forever."""
        if process.stderr is None:
            return "Core Audio helper did not provide a readiness channel"
        selector = selectors.DefaultSelector()
        line = ""
        timeout_error = None
        try:
            selector.register(process.stderr, selectors.EVENT_READ)
            events = selector.select(timeout)
            if not events:
                timeout_error = (
                    "Core Audio system-audio permission was denied or the HAL "
                    f"did not respond within {timeout:g}s"
                )
            else:
                line = process.stderr.readline().decode("utf-8", "replace").strip()
        finally:
            selector.close()
        process.stderr.close()
        if timeout_error is not None:
            return timeout_error
        if line == READY:
            return (
                f"Core Audio helper exited with code {process.poll()}"
                if process.poll() is not None
                else None
            )
        if line:
            return line.removeprefix("ERROR ")
        return f"Core Audio helper exited with code {process.poll()}"

    def attach(self, process):
        """Retain the helper's control pipe; selection writes stay off the UI lock."""
        self.process = process
        self.control = process.stdin

    def follow(self, application):
        """Record a desired application; the bounded watcher sends it later."""
        self.application = application

    def start(self):
        """Start the bounded selection reconciler."""
        if self.watcher is not None:
            return
        self.stopping.clear()
        self.watcher = threading.Thread(
            target=self._follow,
            name="CoreAudioTapLinker",
            daemon=True,
        )
        self.watcher.start()

    def _follow(self):
        while True:
            self._send_selection()
            if self.stopping.wait(self.poll):
                return

    def _send_selection(self):
        """Send the latest selection every pass so restarted processes rebind."""
        if self.control is None:
            return
        payload = json.dumps({"application": self.application}, ensure_ascii=False)
        try:
            self.control.write(f"{payload}\n".encode())
            self.control.flush()
        except (BrokenPipeError, OSError):
            return

    def stop(self):
        """Stop the reconciler and close stdin so a parent death exits the helper."""
        if self.watcher is not None:
            self.stopping.set()
            self.watcher.join(timeout=5)
            self.watcher = None
        if self.control is not None:
            self.control.close()
            self.control = None
