"""Ordered live transcript rows owned next to the controller.

The Textual app may still paint from the same ``Entry`` objects, but this store
is the source of truth for save, the session recorder's accepted view, and the
socket wire. Provisional speech (committed, not yet ``finish_turn``-accepted)
stays in-process until accept; rejected provisionals never publish.

Streaming deltas are accepted synchronously so the Codex worker never waits on
clients. ``entry_updated`` publishes are coalesced: at most one per open row
per ``flush_updates`` call, and the payload always carries the row's full
current text.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from tagalong.presentation import Entry

Publish = Callable[[str, Mapping[str, object]], object]


@dataclass
class TranscriptRow:
    """One store-owned row: monotonic id, live ``Entry``, provisional flag."""

    id: int
    entry: Entry
    provisional: bool = False


class TranscriptStore:
    """Ordered transcript with accepted and provisional views."""

    def __init__(self, publish: Publish | None = None) -> None:
        self._lock = threading.Lock()
        self._rows: list[TranscriptRow] = []
        self._by_id: dict[int, TranscriptRow] = {}
        self._by_entry: dict[int, int] = {}
        self._next_id = 1
        self._dirty: set[int] = set()
        self._publish = publish

    def set_publisher(self, publish: Publish | None) -> None:
        """Install or clear the EventLog publish callback (controller-owned)."""
        with self._lock:
            self._publish = publish

    def append(self, entry: Entry, *, provisional: bool = False) -> int:
        """Append *entry*, assign a monotonic id, and publish when accepted."""
        with self._lock:
            row_id = self._next_id
            self._next_id += 1
            row = TranscriptRow(row_id, entry, provisional=provisional)
            self._rows.append(row)
            self._by_id[row_id] = row
            self._by_entry[id(entry)] = row_id
            if not provisional:
                self._emit_unlocked(
                    "transcript.entry_added",
                    {"id": row_id, "entry": self._payload_unlocked(row)},
                )
            return row_id

    def append_text(self, entry: Entry, delta: str) -> None:
        """Accept a streaming delta synchronously; coalesce the wire update."""
        if not delta:
            return
        with self._lock:
            row = self._row_for_entry_unlocked(entry)
            row.entry.text += delta
            if not row.provisional:
                self._dirty.add(row.id)

    def finalize(
        self,
        entry: Entry,
        *,
        interrupted: bool | None = None,
        seconds: float | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Mark an open row finished and flush any coalesced text update."""
        with self._lock:
            row = self._row_for_entry_unlocked(entry)
            row.entry.streaming = False
            if interrupted is not None:
                row.entry.interrupted = interrupted
            if seconds is not None:
                row.entry.seconds = seconds
            if exit_code is not None:
                row.entry.exit_code = exit_code
            if not row.provisional:
                self._dirty.add(row.id)
            self._flush_updates_unlocked()

    def append_command_output(self, entry: Entry, delta: str) -> None:
        """Append command output text and mark the row dirty for publish."""
        if not delta:
            return
        with self._lock:
            row = self._row_for_entry_unlocked(entry)
            row.entry.output.append(delta)
            if not row.provisional:
                self._dirty.add(row.id)

    def accept(self, entries: Iterable[Entry]) -> None:
        """Promote provisional rows and publish each as ``entry_added``."""
        with self._lock:
            for entry in entries:
                row_id = self._by_entry.get(id(entry))
                if row_id is None:
                    continue
                row = self._by_id[row_id]
                if not row.provisional:
                    continue
                row.provisional = False
                self._emit_unlocked(
                    "transcript.entry_added",
                    {"id": row.id, "entry": self._payload_unlocked(row)},
                )

    def reject(self, entries: Iterable[Entry]) -> None:
        """Drop provisional rows. They were never on the wire."""
        with self._lock:
            rejected = {id(entry) for entry in entries}
            kept: list[TranscriptRow] = []
            for row in self._rows:
                if id(row.entry) in rejected:
                    self._by_id.pop(row.id, None)
                    self._by_entry.pop(id(row.entry), None)
                    self._dirty.discard(row.id)
                    continue
                kept.append(row)
            self._rows = kept

    def clear(self) -> None:
        """Forget every row and reset ids for the next session."""
        with self._lock:
            self._rows.clear()
            self._by_id.clear()
            self._by_entry.clear()
            self._dirty.clear()
            self._next_id = 1
            self._emit_unlocked("transcript.cleared", {})

    def flush_updates(self) -> None:
        """Publish at most one ``entry_updated`` per dirty row (full text)."""
        with self._lock:
            self._flush_updates_unlocked()

    def entries(self, *, include_provisional: bool = False) -> tuple[Entry, ...]:
        """Return live ``Entry`` objects in store order."""
        with self._lock:
            if include_provisional:
                return tuple(row.entry for row in self._rows)
            return tuple(row.entry for row in self._rows if not row.provisional)

    def transcript_entries(self) -> Sequence[Entry]:
        """Accepted-only view for ``transcript.save`` (F5 / recorded view)."""
        return self.entries()

    def rows(self, *, include_provisional: bool = False) -> tuple[TranscriptRow, ...]:
        """Return store rows (ids + flags) for snapshots and tests."""
        with self._lock:
            if include_provisional:
                return tuple(self._rows)
            return tuple(row for row in self._rows if not row.provisional)

    def id_for(self, entry: Entry) -> int | None:
        """The monotonic id for *entry*, or None if it is not in the store."""
        with self._lock:
            return self._by_entry.get(id(entry))

    def row_payload(self, row_id: int) -> Mapping[str, object]:
        """JSON-ready fields for one row (full current text)."""
        with self._lock:
            return self._payload_unlocked(self._by_id[row_id])

    def _row_for_entry_unlocked(self, entry: Entry) -> TranscriptRow:
        row_id = self._by_entry.get(id(entry))
        if row_id is None:
            raise KeyError("entry is not in the transcript store")
        return self._by_id[row_id]

    def _flush_updates_unlocked(self) -> None:
        dirty = sorted(self._dirty)
        self._dirty.clear()
        for row_id in dirty:
            row = self._by_id.get(row_id)
            if row is None or row.provisional:
                continue
            self._emit_unlocked(
                "transcript.entry_updated",
                {"id": row_id, "entry": self._payload_unlocked(row)},
            )

    def _emit_unlocked(self, name: str, payload: Mapping[str, object]) -> None:
        if self._publish is not None:
            self._publish(name, payload)

    @staticmethod
    def _payload_unlocked(row: TranscriptRow) -> dict[str, object]:
        entry = row.entry
        return {
            "kind": entry.kind,
            "source": entry.source,
            "text": entry.text,
            "stamp": entry.stamp,
            "reply_to": entry.reply_to,
            "interrupted": entry.interrupted,
            "output": list(entry.output),
            "exit_code": entry.exit_code,
            "streaming": entry.streaming,
            "seconds": entry.seconds,
        }
