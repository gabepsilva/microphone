"""Ordered live transcript rows owned next to the controller.

The Textual app may still paint from the same ``Entry`` objects, but this store
is the source of truth for save, the session recorder's accepted view, and the
socket wire. Provisional speech (committed, not yet ``finish_turn``-accepted)
stays in-process until accept; rejected provisionals never publish.

Streaming deltas are accepted synchronously so the Codex worker never waits on
clients. ``entry_updated`` publishes are coalesced: at most one per open row
per flush interval, and the payload always carries the row's full current text.
A store-owned pump drains the dirty set so headless sessions still publish
updates without a Textual repaint timer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from tagalong.presentation import Entry

Publish = Callable[[str, Mapping[str, object]], object]

# Same cadence as ``VoiceCodexApp.STREAM_FLUSH_INTERVAL_SECONDS`` — paint and
# wire share one coalesce rate so clients see the same growth steps.
FLUSH_INTERVAL_SECONDS = 0.05


@dataclass
class TranscriptRow:
    """One store-owned row: optional wire id, live ``Entry``, provisional flag.

    Monotonic ids are assigned when a row becomes accepted (non-provisional
    ``append``, or ``accept``), so id order matches ``entry_added`` publish
    order. Provisionals are identity-keyed only until then.
    """

    entry: Entry
    id: int | None = None
    provisional: bool = False


class TranscriptStore:
    """Ordered transcript with accepted and provisional views.

    When constructed by :class:`~tagalong.control.controller.Controller`,
    ``lock`` is the controller's writer lock so transcript publishes and
    ``EventLog.publish`` stay in one order without nested-lock deadlocks.
    """

    def __init__(
        self,
        publish: Publish | None = None,
        *,
        lock: threading.RLock | None = None,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self._lock = lock if lock is not None else threading.RLock()
        self._rows: list[TranscriptRow] = []
        self._by_id: dict[int, TranscriptRow] = {}
        self._by_entry: dict[int, TranscriptRow] = {}
        self._next_id = 1
        self._dirty: set[int] = set()
        self._publish = publish
        self._flush_interval = flush_interval
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None

    @property
    def lock(self) -> threading.RLock:
        """Writer lock shared with the controller when one owns this store."""
        return self._lock

    def set_publisher(self, publish: Publish | None) -> None:
        """Install or clear the EventLog publish callback (controller-owned)."""
        with self._lock:
            self._publish = publish

    def start_coalesce_pump(self) -> None:
        """Drain dirty rows on ``flush_interval`` without a Textual timer."""
        with self._lock:
            if self._pump_thread is not None:
                return
            self._pump_stop.clear()
            thread = threading.Thread(
                target=self._coalesce_loop,
                name="transcript-coalesce",
                daemon=True,
            )
            self._pump_thread = thread
        thread.start()

    def stop_coalesce_pump(self) -> None:
        """Stop the coalesce pump (tests / shutdown)."""
        self._pump_stop.set()
        with self._lock:
            thread = self._pump_thread
            self._pump_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def append(self, entry: Entry, *, provisional: bool = False) -> int | None:
        """Append *entry*; assign a wire id and publish only when accepted."""
        with self._lock:
            if provisional:
                row = TranscriptRow(entry=entry, provisional=True)
                self._rows.append(row)
                self._by_entry[id(entry)] = row
                return None
            row_id = self._allocate_id_unlocked()
            row = TranscriptRow(entry=entry, id=row_id, provisional=False)
            self._rows.append(row)
            self._by_id[row_id] = row
            self._by_entry[id(entry)] = row
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
            if not row.provisional and row.id is not None:
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
            if not row.provisional and row.id is not None:
                self._dirty.add(row.id)
            self._flush_updates_unlocked()

    def append_command_output(self, entry: Entry, delta: str) -> None:
        """Append command output text and mark the row dirty for publish."""
        if not delta:
            return
        with self._lock:
            row = self._row_for_entry_unlocked(entry)
            row.entry.output.append(delta)
            if not row.provisional and row.id is not None:
                self._dirty.add(row.id)

    def accept(self, entries: Iterable[Entry]) -> None:
        """Assign wire ids to provisional rows and publish ``entry_added``.

        Accepted rows move to the end of the store so store order matches
        publish / append-blind client order (a note may have landed while the
        speech was still provisional).
        """
        with self._lock:
            for entry in entries:
                row = self._by_entry.get(id(entry))
                if row is None or not row.provisional:
                    continue
                self._rows.remove(row)
                row_id = self._allocate_id_unlocked()
                row.id = row_id
                row.provisional = False
                self._rows.append(row)
                self._by_id[row_id] = row
                self._emit_unlocked(
                    "transcript.entry_added",
                    {"id": row_id, "entry": self._payload_unlocked(row)},
                )

    def reject(self, entries: Iterable[Entry]) -> None:
        """Drop provisional rows. They were never on the wire.

        Non-provisional matches are kept: removing a published row without an
        ``entry_removed`` event would desync every socket client.
        """
        with self._lock:
            rejected = {id(entry) for entry in entries}
            kept: list[TranscriptRow] = []
            for row in self._rows:
                if id(row.entry) in rejected:
                    if not row.provisional:
                        kept.append(row)
                        continue
                    self._by_entry.pop(id(row.entry), None)
                    if row.id is not None:
                        self._by_id.pop(row.id, None)
                        self._dirty.discard(row.id)
                    continue
                kept.append(row)
            self._rows = kept

    def clear(self) -> None:
        """Forget every row. Ids keep climbing so ``(instance, id)`` stays unique."""
        with self._lock:
            self._rows.clear()
            self._by_id.clear()
            self._by_entry.clear()
            self._dirty.clear()
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
        """The monotonic id for *entry*, or None if unassigned / absent."""
        with self._lock:
            row = self._by_entry.get(id(entry))
            return None if row is None else row.id

    def row_payload(self, row_id: int) -> Mapping[str, object]:
        """JSON-ready fields for one row (full current text)."""
        with self._lock:
            return self._payload_unlocked(self._by_id[row_id])

    def _allocate_id_unlocked(self) -> int:
        row_id = self._next_id
        self._next_id += 1
        return row_id

    def _row_for_entry_unlocked(self, entry: Entry) -> TranscriptRow:
        row = self._by_entry.get(id(entry))
        if row is None:
            raise KeyError("entry is not in the transcript store")
        return row

    def _coalesce_loop(self) -> None:
        while not self._pump_stop.wait(self._flush_interval):
            self.flush_updates()

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
