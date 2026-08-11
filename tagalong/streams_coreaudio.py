"""Darwin stream adapter placeholder for the phase-2 platform port.

The Core Audio process-tap implementation is phase 3 work. The port still
needs a concrete Darwin backend now so importing the app on macOS is safe and
far-end selection fails by name at the point of use.
"""

from __future__ import annotations


def graph():
    """Return an empty snapshot until Core Audio process taps are available."""
    return []


def application_streams(_objects, **_kwargs):
    """Return no applications until the Core Audio graph is implemented."""
    return []


def applications(_streams, **_kwargs):
    """Return no applications until the Core Audio graph is implemented."""
    return []


def offered_applications(_objects, **_kwargs):
    """Return no picker entries until the Core Audio graph is implemented."""
    return []


def require_stream_capture():
    """Explain why far-end selection is unavailable in this phase."""
    raise RuntimeError(
        "Core Audio process-tap capture is not implemented in this build; "
        "macOS far-end capture requires the next compatibility phase."
    )


class StreamTap:
    """Fail at construction if a caller bypasses the capability check."""

    CHANNELS = 2

    def __init__(self, application=None, **_kwargs):
        self.application = application

    def command(self, _samplerate):
        """Raise the same named capability error as the selector."""
        require_stream_capture()

    def process_options(self):
        """Reject helper startup until the Core Audio adapter exists."""
        require_stream_capture()

    def wait_ready(self, _process, _timeout):
        """Reject readiness checks until the Core Audio adapter exists."""
        require_stream_capture()

    def attach(self, _process):
        """Reject helper attachment until the Core Audio adapter exists."""
        require_stream_capture()

    def follow(self, _application):
        """Reject retargeting until the Core Audio adapter exists."""
        require_stream_capture()

    def start(self):
        """Reject tap startup until the Core Audio adapter exists."""
        require_stream_capture()

    def stop(self):
        """Reject tap shutdown until the Core Audio adapter exists."""
        require_stream_capture()
