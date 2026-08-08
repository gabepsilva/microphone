"""Persist every finished transcript row to a session file.

The file is an artifact the operator goes and reads, so it lives under
``~/tagalong/transcripts`` rather than a cache directory. Each finished entry
is written and flushed immediately: the session ends on Ctrl-C or a closed
terminal as often as it ends on a clean quit, and a write deferred to
shutdown is a write that does not happen.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from .presentation import Entry

TRANSCRIPT_ROOT_NAME = "tagalong"
TRANSCRIPTS_DIR_NAME = "transcripts"


def default_transcript_dir(*, home: str | None = None) -> Path:
    """Return ``~/tagalong/transcripts``."""
    root = Path(home or os.path.expanduser("~"))
    return root / TRANSCRIPT_ROOT_NAME / TRANSCRIPTS_DIR_NAME


def transcript_filename(when: datetime) -> str:
    """Build ``YYYY-MM-DD_HH_MM_SS.txt`` from a timezone-aware ``when``."""
    return when.astimezone().strftime("%Y-%m-%d_%H_%M_%S") + ".txt"


def _header_stamp(when: datetime) -> str:
    """Human-readable start line for the file header."""
    local = when.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S %z")


def _inline_first_line(header: str, text: str) -> str:
    """Keep the first body line beside its stamped author header."""
    if not text:
        return header
    first, separator, remainder = text.partition("\n")
    if not separator:
        return f"{header} {first}"
    return f"{header} {first}\n{remainder}"


def format_entry(entry: Entry) -> str:
    """Render one finished transcript row as a multi-line text block."""
    stamp = entry.stamp or ""
    kind = entry.kind or ""
    text = entry.text or ""
    lines: list[str] = []

    if kind == "note":
        lines.append(_inline_first_line(f"[{stamp}] System", text))
    elif kind == "reasoning":
        seconds = entry.seconds
        label = "Taga (thinking)"
        if seconds is not None:
            label = f"Taga (thinking {seconds:.1f}s)"
        lines.append(_inline_first_line(f"[{stamp}] {label}", text))
    elif kind == "command":
        lines.append(f"[{stamp}] $ {text}")
        for line in entry.output:
            lines.append(line)
        exit_code = entry.exit_code
        if exit_code is not None:
            lines.append(f"[exit {exit_code}]")
    else:
        source = entry.source or "System"
        header = f"[{stamp}] {source}"
        if entry.interrupted:
            header += " (interrupted)"
        lines.append(_inline_first_line(header, text))

    return "\n".join(lines) + "\n\n"


def write_transcript_export(
    entries: Sequence[Entry],
    directory: Path,
    *,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Write a complete transcript snapshot and return the new file path."""
    when = (clock or (lambda: datetime.now(UTC).astimezone()))()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / transcript_filename(when)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# TagAlong transcript · {_header_stamp(when)}\n\n")
        for entry in entries:
            handle.write(format_entry(entry))
    return path.resolve()


class TranscriptRecorder:
    """Append finished transcript entries to a timestamp-named session file.

    The file opens lazily on the first entry, so a silent session and a ``/new``
    on an empty transcript leave no stray files. A directory that cannot be
    written is reported once and then stays quiet — a transcript failing must
    not take the session down.

    ``record``, ``roll``, and ``close`` share one lock so a ``session.new``
    worker rolling the file cannot race an entry-thread write after the
    transcript has cleared (handler-owned settle runs off the app thread).
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.directory = directory
        self._clock = clock or (lambda: datetime.now(UTC).astimezone())
        self._stream = stream
        self._file: TextIO | None = None
        self.path: Path | None = None
        self._reported = False
        self._closed = False
        self._lock = threading.Lock()

    def record(self, entry: Entry) -> bool:
        """Append one finished entry and flush it to disk."""
        with self._lock:
            if self._closed:
                return False
            try:
                handle = self._ensure_open()
                handle.write(format_entry(entry))
                handle.flush()
            except OSError as error:
                self._report(error)
                return False
            except ValueError:
                # Handle closed between open and write (or by a concurrent roll
                # before the lock existed). Soft-fail the way OSError does.
                return False
            return True

    def roll(self) -> None:
        """Close the current file so the next entry opens a fresh one."""
        with self._lock:
            self._close_unlocked()
            self._closed = False
            self._reported = False

    def close(self) -> None:
        """Close the open file, if any."""
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        handle = self._file
        self._file = None
        self._closed = True
        if handle is None:
            return
        try:
            handle.close()
        except OSError as error:
            self._report(error)

    def _ensure_open(self) -> TextIO:
        if self._file is not None:
            return self._file
        directory = (
            self.directory if self.directory is not None else default_transcript_dir()
        )
        when = self._clock()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / transcript_filename(when)
        handle = path.open("a", encoding="utf-8")
        handle.write("# TagAlong transcript\n")
        handle.write(f"# started {_header_stamp(when)}\n\n")
        handle.flush()
        self._file = handle
        self.path = path
        return handle

    def _report(self, error: OSError) -> None:
        if self._reported:
            return
        self._reported = True
        print(
            f"\nTranscript will not be recorded: {error}",
            file=sys.stderr if self._stream is None else self._stream,
            flush=True,
        )
