"""Session transcript file recording."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from tagalong.presentation import Entry
from tagalong.recording import (
    TranscriptRecorder,
    default_transcript_dir,
    format_entry,
    transcript_filename,
)


def test_default_transcript_dir_is_under_home_tagalong(tmp_path) -> None:
    assert default_transcript_dir(home=str(tmp_path)) == (
        tmp_path / "tagalong" / "transcripts"
    )


def test_transcript_filename_uses_local_timestamp() -> None:
    when = datetime(2026, 7, 31, 22, 45, 3, tzinfo=UTC)
    name = transcript_filename(when)
    assert name.endswith(".txt")
    assert name.startswith("2026-07-3")
    assert "_" in name


def test_format_entry_renders_speech_note_reasoning_and_command() -> None:
    speech = format_entry(
        Entry(kind="speech", source="Voice", text="hello Taga", stamp="22:45:11")
    )
    note = format_entry(Entry(kind="note", text="tts off", stamp="22:45:20"))
    reasoning = format_entry(
        Entry(
            kind="reasoning",
            source="Taga",
            text="Checking the window.",
            stamp="22:45:12",
            seconds=1.4,
        )
    )
    command = format_entry(
        Entry(
            kind="command",
            text="ls",
            stamp="22:45:30",
            output=["file.txt"],
            exit_code=1,
        )
    )
    interrupted = format_entry(
        Entry(
            kind="speech",
            source="Taga",
            text="half an answ",
            stamp="22:45:40",
            interrupted=True,
        )
    )
    multiline = format_entry(
        Entry(
            kind="speech",
            source="Taga",
            text="line one\nline two",
            stamp="22:45:50",
        )
    )

    assert speech == "[22:45:11] Voice hello Taga\n\n"
    assert note == "[22:45:20] System tts off\n\n"
    assert reasoning == ("[22:45:12] Taga (thinking 1.4s) Checking the window.\n\n")
    assert command == "[22:45:30] $ ls\nfile.txt\n[exit 1]\n\n"
    assert interrupted == "[22:45:40] Taga (interrupted) half an answ\n\n"
    assert multiline == "[22:45:50] Taga line one\nline two\n\n"


def test_format_entry_omits_thinking_duration_when_unknown() -> None:
    text = format_entry(
        Entry(kind="reasoning", source="Taga", text="", stamp="01:00:00")
    )
    assert text == "[01:00:00] Taga (thinking)\n\n"


def test_format_entry_handles_empty_speech_and_command_without_exit() -> None:
    speech = format_entry(
        Entry(kind="speech", source="Voice", text="", stamp="01:00:01")
    )
    command = format_entry(
        Entry(kind="command", text="pwd", stamp="01:00:02", output=[], exit_code=None)
    )
    assert speech == "[01:00:01] Voice\n\n"
    assert command == "[01:00:02] $ pwd\n\n"


def test_recorder_reports_a_close_failure_once(tmp_path, monkeypatch) -> None:
    when = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC).astimezone()
    stream = StringIO()
    real_open = Path.open

    class BrokenHandle(StringIO):
        def close(self) -> None:
            raise OSError("disk gone")

    def open_broken(self, *args, **kwargs):
        if self.parent == tmp_path:
            return BrokenHandle()
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_broken)
    recorder = TranscriptRecorder(directory=tmp_path, clock=lambda: when, stream=stream)
    recorder.record(Entry(kind="note", text="hi", stamp="00:00:01"))
    recorder.close()
    recorder._closed = False
    recorder._file = BrokenHandle()
    recorder.close()

    assert stream.getvalue().count("Transcript will not be recorded:") == 1


def test_recorder_creates_directory_and_flushes_each_entry(tmp_path) -> None:
    directory = tmp_path / "transcripts"
    when = datetime(2026, 7, 31, 22, 45, 3, tzinfo=UTC).astimezone()
    recorder = TranscriptRecorder(directory=directory, clock=lambda: when)
    try:
        assert (
            recorder.record(
                Entry(
                    kind="speech", source="Voice", text="first turn", stamp="22:45:11"
                )
            )
            is True
        )

        assert directory.is_dir()
        assert recorder.path is not None
        assert recorder.path.name == transcript_filename(when)
        # Behavioural flush check: a second handle sees the entry while the
        # recorder still holds the file open.
        on_disk = recorder.path.read_text(encoding="utf-8")
        assert "# TagAlong transcript" in on_disk
        assert "first turn" in on_disk

        assert (
            recorder.record(Entry(kind="note", text="tts off", stamp="22:45:20"))
            is True
        )
        assert "tts off" in recorder.path.read_text(encoding="utf-8")
    finally:
        recorder.close()


def test_recorder_roll_keeps_old_file_and_opens_a_fresh_one(tmp_path) -> None:
    stamps = iter(
        [
            datetime(2026, 7, 31, 22, 45, 3, tzinfo=UTC).astimezone(),
            datetime(2026, 7, 31, 23, 0, 0, tzinfo=UTC).astimezone(),
        ]
    )
    recorder = TranscriptRecorder(directory=tmp_path, clock=lambda: next(stamps))
    try:
        recorder.record(
            Entry(kind="speech", source="Voice", text="before /new", stamp="22:45:11")
        )
        first = recorder.path
        assert first is not None
        recorder.roll()
        recorder.record(
            Entry(kind="speech", source="Voice", text="after /new", stamp="23:00:01")
        )
        second = recorder.path
        assert second is not None
        assert first != second
        assert "before /new" in first.read_text(encoding="utf-8")
        assert "before /new" not in second.read_text(encoding="utf-8")
        assert "after /new" in second.read_text(encoding="utf-8")
    finally:
        recorder.close()


def test_recorder_stays_quiet_after_one_unwritable_directory_report(
    tmp_path,
) -> None:
    # A regular file where the directory should be makes mkdir fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    stream = StringIO()
    recorder = TranscriptRecorder(directory=blocked, stream=stream)

    assert recorder.record(Entry(kind="note", text="one", stamp="00:00:01")) is False
    assert recorder.record(Entry(kind="note", text="two", stamp="00:00:02")) is False

    report = stream.getvalue()
    assert report.count("Transcript will not be recorded:") == 1
    assert recorder.path is None


def test_closed_recorder_ignores_further_entries(tmp_path) -> None:
    recorder = TranscriptRecorder(
        directory=tmp_path,
        clock=lambda: datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    )
    recorder.close()
    recorder.record(Entry(kind="note", text="too late", stamp="00:00:01"))
    assert list(tmp_path.iterdir()) == []


def test_lazy_open_leaves_no_file_when_nothing_is_recorded(tmp_path) -> None:
    recorder = TranscriptRecorder(directory=tmp_path)
    recorder.roll()
    recorder.close()
    assert list(tmp_path.iterdir()) == []
