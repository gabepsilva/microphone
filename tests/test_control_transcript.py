"""TranscriptStore: ordered rows, provisional gate, coalesce-before-publish."""

from __future__ import annotations

import pytest

from tagalong.control.transcript import TranscriptStore
from tagalong.presentation import Entry


def test_append_assigns_monotonic_ids() -> None:
    store = TranscriptStore()
    first = store.append(Entry(kind="note", text="a"))
    second = store.append(Entry(kind="note", text="b"))

    assert first == 1
    assert second == 2
    assert [row.id for row in store.rows()] == [1, 2]


def test_accepted_view_excludes_provisional_rows() -> None:
    store = TranscriptStore()
    store.append(Entry(kind="speech", source="Voice", text="kept"))
    store.append(Entry(kind="speech", source="Voice", text="pending"), provisional=True)

    assert [entry.text for entry in store.entries()] == ["kept"]
    assert [entry.text for entry in store.entries(include_provisional=True)] == [
        "kept",
        "pending",
    ]
    assert [entry.text for entry in store.transcript_entries()] == ["kept"]


def test_accept_promotes_provisional_and_publish_adds_once() -> None:
    published: list[tuple[str, dict[str, object]]] = []
    store = TranscriptStore(
        publish=lambda name, payload: published.append((name, dict(payload)))
    )
    entry = Entry(kind="speech", source="Voice", text="hi")
    row_id = store.append(entry, provisional=True)

    assert published == []
    store.accept([entry])

    assert [entry.text for entry in store.entries()] == ["hi"]
    assert published == [
        (
            "transcript.entry_added",
            {"id": row_id, "entry": store.row_payload(row_id)},
        )
    ]


def test_reject_removes_provisional_without_publishing() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    kept = Entry(kind="note", text="kept")
    store.append(kept)
    published.clear()
    entry = Entry(kind="speech", source="Voice", text="echo")
    store.append(entry, provisional=True)

    store.reject([entry])

    assert [row.text for row in store.entries(include_provisional=True)] == ["kept"]
    assert published == []


def test_non_provisional_append_publishes_entry_added() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    store.append(Entry(kind="note", text="system"))

    assert published == ["transcript.entry_added"]


def test_text_deltas_coalesce_to_one_entry_updated_per_flush() -> None:
    published: list[tuple[str, dict[str, object]]] = []
    store = TranscriptStore(
        publish=lambda name, payload: published.append((name, dict(payload)))
    )
    entry = Entry(kind="speech", source="Taga", text="", streaming=True)
    row_id = store.append(entry)
    published.clear()

    store.append_text(entry, "Hel")
    store.append_text(entry, "lo")
    store.append_text(entry, "!")
    assert published == []
    assert entry.text == "Hello!"

    store.flush_updates()

    assert len(published) == 1
    name, payload = published[0]
    assert name == "transcript.entry_updated"
    assert payload["id"] == row_id
    assert payload["entry"] == store.row_payload(row_id)
    assert store.row_payload(row_id)["text"] == "Hello!"


def test_finalize_marks_streaming_done_and_flushes() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    entry = Entry(kind="speech", source="Taga", text="hi", streaming=True)
    store.append(entry)
    published.clear()

    store.append_text(entry, "!")
    store.finalize(entry)

    assert entry.streaming is False
    assert entry.text == "hi!"
    assert published == ["transcript.entry_updated"]


def test_finalize_records_interrupt_seconds_and_exit_code() -> None:
    store = TranscriptStore()
    entry = Entry(kind="command", source="Taga", text="ls", streaming=True)
    store.append(entry)

    store.finalize(entry, interrupted=True, seconds=1.5, exit_code=2)

    assert entry.streaming is False
    assert entry.interrupted is True
    assert entry.seconds == 1.5
    assert entry.exit_code == 2


def test_append_command_output_coalesces_like_text_deltas() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    entry = Entry(kind="command", source="Taga", text="ls")
    store.append(entry)
    published.clear()

    store.append_command_output(entry, "a")
    store.append_command_output(entry, "b")
    assert published == []
    store.flush_updates()

    assert published == ["transcript.entry_updated"]
    assert entry.output == ["a", "b"]


def test_empty_deltas_are_ignored() -> None:
    store = TranscriptStore()
    entry = Entry(kind="speech", source="Taga", text="x", streaming=True)
    store.append(entry)
    store.append_text(entry, "")
    store.append_command_output(entry, "")
    assert entry.text == "x"
    assert entry.output == []


def test_set_publisher_replaces_callback() -> None:
    first: list[str] = []
    second: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: first.append(name))
    store.set_publisher(lambda name, _payload: second.append(name))
    store.append(Entry(kind="note", text="n"))
    assert first == []
    assert second == ["transcript.entry_added"]


def test_accept_skips_unknown_and_already_accepted_rows() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    kept = Entry(kind="speech", source="Voice", text="kept")
    store.append(kept)
    published.clear()

    store.accept([Entry(kind="speech", source="Voice", text="ghost"), kept])

    assert published == []
    assert [entry.text for entry in store.entries()] == ["kept"]


def test_unknown_entry_update_raises() -> None:
    store = TranscriptStore()
    with pytest.raises(KeyError, match="not in the transcript store"):
        store.append_text(Entry(kind="note", text="missing"), "x")


def test_rows_view_matches_provisional_filter() -> None:
    store = TranscriptStore()
    store.append(Entry(kind="note", text="a"))
    store.append(Entry(kind="note", text="b"), provisional=True)
    assert [row.entry.text for row in store.rows()] == ["a"]
    assert [row.entry.text for row in store.rows(include_provisional=True)] == [
        "a",
        "b",
    ]


def test_clear_empties_store_and_publishes_cleared() -> None:
    published: list[str] = []
    store = TranscriptStore(publish=lambda name, _payload: published.append(name))
    store.append(Entry(kind="note", text="a"))
    store.append(Entry(kind="note", text="b"), provisional=True)
    published.clear()

    store.clear()

    assert store.entries(include_provisional=True) == ()
    assert published == ["transcript.cleared"]


def test_ids_reset_on_clear() -> None:
    store = TranscriptStore()
    store.append(Entry(kind="note", text="a"))
    store.clear()
    assert store.append(Entry(kind="note", text="b")) == 1


def test_lookup_by_entry_identity() -> None:
    store = TranscriptStore()
    entry = Entry(kind="note", text="x")
    row_id = store.append(entry)
    assert store.id_for(entry) == row_id
