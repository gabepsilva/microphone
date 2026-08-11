#!/usr/bin/env python3
"""Linux ``/proc`` implementation of session helper identity and cleanup.

Two helpers do the work no Python thread can: ``ffplay`` plays a synthesized
sentence, and ``pw-record`` holds the capture node the far end is tapped into.
Both are cleaned up when a session ends the way it means to. Neither survives
that promise when the session is killed outright — the kernel re-parents them
to init and they go on holding a pipe nobody will ever write to again.

That is not only untidy. An orphaned player is still a playback stream in the
audio graph, and the test for "this program's own audio" is whether the stream
belongs to a process this one started. An orphan's parent is init, so the walk
up its ancestry never reaches this session, and yesterday's speech offers
itself as something to transcribe today.

So the processes are tagged with the session that started them. The tag
answers both questions at once: it is what makes a stream recognizable as this
program's however it was re-parented, and it is what tells a leftover from a
live helper when a new session sweeps up before starting.

``PR_SET_PDEATHSIG`` would ask the kernel to do this instead, and would need
``preexec_fn`` to install it. Running arbitrary code between fork and exec in a
program with this many threads is a worse bargain than a sweep.
"""

from __future__ import annotations

import os
import signal

SESSION_MARKER = "TAGALONG_SESSION"

PROC = "/proc"


def session_of(pid, proc=PROC):
    """Name the session that started a process, or nothing if this one did not.

    Read from the process's own environment rather than tracked in memory,
    because the sweep runs before anything this session starts exists, and what
    it is looking for outlived the program that knew about it.
    """
    try:
        with open(f"{proc}/{pid}/environ", "rb") as environ:
            entries = environ.read().split(b"\0")
    except OSError:
        return None
    for entry in entries:
        name, separator, value = entry.partition(b"=")
        if separator and name.decode("utf-8", "replace") == SESSION_MARKER:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def is_running(pid, kill=os.kill):
    """Whether a process still exists."""
    try:
        kill(pid, 0)
    except OSError:
        return False
    return True


def started_here(pid, proc=PROC):
    """Whether a process was started by some session of this program.

    Any session, not only this one. A stream left behind by yesterday's run is
    still this program's own voice, and offering it as a far end to transcribe
    would be the same mistake as offering the voice speaking right now.
    """
    return session_of(pid, proc=proc) is not None


def orphans(proc=PROC, own_pid=None, running=is_running):
    """Find helpers whose session is gone.

    A helper belonging to a session that is still alive is left strictly
    alone — including this one's, which is the case that matters, since the
    sweep runs while this session is starting its own.
    """
    own = os.getpid() if own_pid is None else own_pid
    found = []
    try:
        entries = os.listdir(proc)
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        session = session_of(pid, proc=proc)
        if session is None or session == own or running(session):
            continue
        found.append(pid)
    return found


def sweep_orphans(proc=PROC, own_pid=None, running=is_running, kill=os.kill):
    """End the helpers earlier sessions left behind; report how many.

    Failures are ignored rather than reported: a process that exited between
    being listed and being signalled is the outcome this wanted, and one owned
    by another user was never this program's to begin with.
    """
    swept = 0
    for pid in orphans(proc=proc, own_pid=own_pid, running=running):
        try:
            kill(pid, signal.SIGTERM)
        except OSError:
            continue
        swept += 1
    return swept
