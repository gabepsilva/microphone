"""Platform-neutral helper-session port."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Protocol


class SessionBackend(Protocol):
    """The platform-specific process identity operations."""

    def session_of(self, pid: int, **kwargs): ...

    def started_here(self, pid: int, **kwargs): ...

    def orphans(self, **kwargs): ...

    def sweep_orphans(self, **kwargs): ...


_DEFAULT_PLATFORM = sys.platform
SESSION_MARKER = "TAGALONG_SESSION"


def default_session_backend(platform: str | None = None) -> ModuleType:
    """Return the helper-session backend for ``platform`` or this host."""
    selected = _DEFAULT_PLATFORM if platform is None else platform
    if selected == "darwin":
        from . import session_darwin

        return session_darwin
    from . import session_proc

    return session_proc


def _backend() -> ModuleType:
    return default_session_backend()


def tagged_environment(base_environment=None, pid=None):
    """Copy an environment, marking it as belonging to this session."""
    environment = dict(os.environ if base_environment is None else base_environment)
    environment[SESSION_MARKER] = str(os.getpid() if pid is None else pid)
    return environment


def session_of(pid, **kwargs):
    """Name the session that started a process, or ``None``."""
    return _backend().session_of(pid, **kwargs)


def is_running(pid, kill=os.kill):
    """Whether a process still exists."""
    try:
        kill(pid, 0)
    except OSError:
        return False
    return True


def started_here(pid, **kwargs):
    """Whether a process belongs to any TagAlong session."""
    return _backend().started_here(pid, **kwargs)


def orphans(proc=None, own_pid=None, running=None):
    """Find helpers whose owning session has gone away."""
    options = {"own_pid": own_pid}
    if proc is not None:
        options["proc"] = proc
    if running is not None:
        options["running"] = running
    return _backend().orphans(**options)


def sweep_orphans(proc=None, own_pid=None, running=None, kill=os.kill):
    """End helpers left by earlier sessions and report how many."""
    options = {"own_pid": own_pid, "kill": kill}
    if proc is not None:
        options["proc"] = proc
    if running is not None:
        options["running"] = running
    return _backend().sweep_orphans(**options)
