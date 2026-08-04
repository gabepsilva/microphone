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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

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


def format_entry(entry: Any) -> str:
    """Render one finished transcript row as a multi-line text block.

    Duck-typed against the fields ``Entry`` carries so this module never
    imports the TUI. The first line of speech, notes, and reasoning is kept
    beside the stamped header; later body lines retain their line breaks.
    """
    stamp = getattr(entry, "stamp", "") or ""
    kind = getattr(entry, "kind", "") or ""
    text = getattr(entry, "text", "") or ""
    lines: list[str] = []

    if kind == "note":
        lines.append(_inline_first_line(f"[{stamp}] System", text))
    elif kind == "reasoning":
        seconds = getattr(entry, "seconds", None)
        label = "Taga (thinking)"
        if seconds is not None:
            label = f"Taga (thinking {seconds:.1f}s)"
        lines.append(_inline_first_line(f"[{stamp}] {label}", text))
    elif kind == "command":
        lines.append(f"[{stamp}] $ {text}")
        for line in getattr(entry, "output", ()) or ():
            lines.append(line)
        exit_code = getattr(entry, "exit_code", None)
        if exit_code is not None:
            lines.append(f"[exit {exit_code}]")
    else:
        source = getattr(entry, "source", "") or "System"
        header = f"[{stamp}] {source}"
        if getattr(entry, "interrupted", False):
            header += " (interrupted)"
        lines.append(_inline_first_line(header, text))

    return "\n".join(lines) + "\n\n"


class TranscriptRecorder:
    """Append finished transcript entries to a timestamp-named session file.

    The file opens lazily on the first entry, so a silent session and a ``/new``
    on an empty transcript leave no stray files. A directory that cannot be
    written is reported once and then stays quiet — a transcript failing must
    not take the session down.
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

    def record(self, entry: Any) -> None:
        """Append one finished entry and flush it to disk."""
        if self._closed:
            return
        try:
            handle = self._ensure_open()
            handle.write(format_entry(entry))
            handle.flush()
        except OSError as error:
            self._report(error)

    def roll(self) -> None:
        """Close the current file so the next entry opens a fresh one."""
        self.close()
        self._closed = False
        self._reported = False

    def close(self) -> None:
        """Close the open file, if any."""
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
