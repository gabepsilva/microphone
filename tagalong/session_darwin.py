"""Darwin process ancestry and Core Audio helper ownership.

macOS does not expose another process's environment, so the Linux
``TAGALONG_SESSION`` marker cannot be read back here.  The picker only needs
the bounded ancestry question: a process is TagAlong's when its parent chain
reaches this session.  Core Audio taps are private to their creating helper,
so stale aggregate-device cleanup is kept as a separate, explicit operation.
"""

from __future__ import annotations

import ctypes
import os
import sys

ANCESTRY_LIMIT = 8
PROC_PIDTBSDINFO = 3
PROC_BSDINFO_PPID_OFFSET = 16


def _libproc():
    """Load libproc lazily so the Darwin adapter remains import-safe on Linux."""
    if sys.platform != "darwin":
        raise RuntimeError("Darwin process identity requires macOS libproc.")
    library = ctypes.CDLL("/usr/lib/libproc.dylib")
    library.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    return library


def parent_process(pid, library=None):
    """Return a process parent through ``proc_pidinfo`` or ``None``."""
    if pid <= 0:
        return None
    try:
        library = _libproc() if library is None else library
        data = ctypes.create_string_buffer(256)
        size = library.proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, data, len(data))
    except OSError:
        return None
    if size <= PROC_BSDINFO_PPID_OFFSET + 4:
        return None
    return int.from_bytes(
        data.raw[PROC_BSDINFO_PPID_OFFSET : PROC_BSDINFO_PPID_OFFSET + 4], "little"
    )


def started_here(
    pid,
    own_pid=None,
    parent=parent_process,
    limit=ANCESTRY_LIMIT,
):
    """Whether a process's bounded parent chain reaches this session."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    own = os.getpid() if own_pid is None else own_pid
    current = pid
    for _ in range(limit + 1):
        if current == own:
            return True
        current = parent(current)
        if current is None:
            return False
        if current == 1 and current != own:
            return False
    return False


def session_of(pid, **kwargs):
    """Return this session's pid when ancestry proves ownership."""
    own = os.getpid() if kwargs.get("own_pid") is None else kwargs["own_pid"]
    return own if started_here(pid, **kwargs) else None


def orphans(**_kwargs):
    """No process-environment orphan list exists on Darwin."""
    return []


def sweep_orphans(**_kwargs):
    """Aggregate-device cleanup is owned by the Core Audio adapter."""
    return 0
