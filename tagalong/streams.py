"""Platform-neutral application-stream capture port.

The application picker and its polling lifecycle are shared by every backend.
Only the graph snapshot, process identity, and tap implementation vary by
platform; the live backend is selected from the host platform and tests can
select a backend explicitly.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol


@dataclass(frozen=True)
class ApplicationStream:
    """One application stream in a backend-neutral shape."""

    node_id: int
    application: str
    title: str
    binary: str
    playing: bool


class StreamCaptureBackend(Protocol):
    """The platform-specific operations the shared application layer needs."""

    SUPPORTS_ALL: bool

    def graph(self, *args, **kwargs): ...

    def application_streams(self, objects, *args, **kwargs): ...

    def applications(self, streams, *args, **kwargs): ...

    def require_stream_capture(self, *args, **kwargs): ...

    StreamTap: type[StreamTapPort]


class StreamTapPort(Protocol):
    """The tap lifecycle shared by the transcriber and the CLI."""

    CHANNELS: int
    application: str | None

    def command(self, samplerate: int): ...

    def process_options(self) -> dict[str, object]: ...

    def wait_ready(self, process, timeout: float) -> str | None: ...

    def attach(self, process): ...

    def follow(self, application: str | None): ...

    def start(self): ...

    def stop(self): ...


_DEFAULT_PLATFORM = sys.platform


def default_stream_backend(platform: str | None = None) -> ModuleType:
    """Return the stream backend for ``platform`` or this host."""
    selected = _DEFAULT_PLATFORM if platform is None else platform
    if selected == "darwin":
        from . import streams_coreaudio

        return streams_coreaudio
    from . import streams_pipewire

    return streams_pipewire


def _backend() -> ModuleType:
    return default_stream_backend()


def graph(*args, **kwargs):
    """Read one platform backend's opaque graph snapshot."""
    return _backend().graph(*args, **kwargs)


def application_streams(objects, *args, **kwargs):
    """Normalize one backend graph into application streams."""
    return _backend().application_streams(objects, *args, **kwargs)


def applications(streams, *args, **kwargs):
    """Collapse backend streams into offered applications."""
    return _backend().applications(streams, *args, **kwargs)


def supports_all() -> bool:
    """Whether the selected backend can capture every external application."""
    return bool(getattr(_backend(), "SUPPORTS_ALL", False))


def require_stream_capture(*args, **kwargs):
    """Raise a named platform error when far-end capture is unavailable."""
    return _backend().require_stream_capture(*args, **kwargs)


class StreamTap:
    """Construct the selected platform's tap adapter."""

    def __new__(cls, *args, **kwargs) -> StreamTapPort:
        return _backend().StreamTap(*args, **kwargs)


TITLE_LIMIT = 32
ALL_APPLICATIONS = "__all__"
ALL_APPLICATIONS_LABEL = "All"


def stream_label(stream: ApplicationStream) -> str:
    """Describe one application the way the picker offers it."""
    state = "playing" if stream.playing else "idle"
    if not stream.title or stream.title == stream.application:
        return f"{stream.application} ({state})"
    title = stream.title
    if len(title) > TITLE_LIMIT:
        title = f"{title[: TITLE_LIMIT - 1]}…"
    return f"{stream.application}: {title} ({state})"


def offered_entries(
    streams: Iterable[ApplicationStream],
    *,
    accept: Callable[[ApplicationStream], bool],
    include_all: bool = False,
) -> list[tuple[str, str]]:
    """Build picker ``(label, name)`` pairs for streams that pass *accept*.

    Session surfaces pass a sticky ``heard`` check; the pre-session chooser
    passes ``playing``. The backend capability entry is independent of the
    stream predicate and is always placed before named applications.
    """
    offered = [(ALL_APPLICATIONS_LABEL, ALL_APPLICATIONS)] if include_all else []
    offered.extend(
        (stream_label(stream), stream.application)
        for stream in streams
        if accept(stream)
    )
    return offered


class ApplicationRefresher:
    """Keep the picker's list of applications current while the session runs."""

    POLL_SECONDS = 4.0

    def __init__(self, display, poll=POLL_SECONDS, dump=graph):
        self.display = display
        self.poll = poll
        self.dump = dump
        self.stopping = threading.Event()
        self.worker = None
        self.error = None
        self.offered = []
        self.heard = set()

    def start(self):
        if self.worker is not None:
            return
        self.worker = threading.Thread(
            target=self._serve, name="ApplicationRefresher", daemon=True
        )
        self.worker.start()

    def refresh(self):
        """Re-read the graph, reporting only changed applications."""
        streams = applications(application_streams(self.dump()))
        self.heard.update(stream.application for stream in streams if stream.playing)
        offered = offered_entries(
            streams,
            accept=lambda stream: stream.application in self.heard,
            include_all=supports_all(),
        )
        if offered == self.offered:
            return False
        self.offered = offered
        self.display.set_audio_streams(offered)
        return True

    def _serve(self):
        while True:
            try:
                self.refresh()
            except RuntimeError as error:
                self.error = error
                print(
                    f"Audio stream discovery stopped: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if self.stopping.wait(self.poll):
                return

    def stop(self):
        if self.worker is None:
            return
        self.stopping.set()
        self.worker.join(timeout=5)
        self.worker = None
