"""The transcription listener's turn boundaries, muting, and teardown.

The listener is the only thing standing between a stream of partial
transcription events and a Codex request, so the behavior that matters is
when it decides a turn is over and when it decides to stay quiet. The silence
timer is driven directly rather than waited on, so these finish immediately
and never depend on wall-clock timing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voice_codex.listener import ConversationListener


class RecordingDisplay:
    """Capture what the listener asks the transcript to show."""

    def __init__(self):
        self.partials: list[tuple[str, str]] = []
        self.commits: list[tuple[str, str]] = []
        self.finished: list[str] = []
        self.closed: list[str] = []

    def update(self, speaker, text):
        self.partials.append((speaker, text))

    def commit(self, speaker, text):
        self.commits.append((speaker, text))

    def finish_turn(self, speaker):
        self.finished.append(speaker)

    def close_speaker(self, speaker):
        self.closed.append(speaker)


class RecordedListener(ConversationListener):
    """A listener that also keeps what it submitted and what it displayed."""

    def __init__(self, display, submitted, **kwargs):
        super().__init__(**kwargs)
        self.recorded = display
        self.submitted = submitted


def line(text="", words=()):
    return SimpleNamespace(
        words=[
            SimpleNamespace(word=word, confidence=confidence)
            for word, confidence in words
        ],
        text=text,
    )


@pytest.fixture
def listener():
    """A listener whose submissions and display calls are both recorded."""
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    return RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=3.0,
        speaker="User Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        on_speech=lambda partial: True,
    )


def test_a_completed_line_is_committed_and_starts_the_silence_timer(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("hello there")))

    assert listener.recorded.commits == [("User Voice", "hello there")]
    assert listener.timer is not None
    listener.timer.cancel()


def test_the_silence_timer_flushes_every_buffered_line_as_one_turn(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("first sentence")))
    listener.on_line_completed(SimpleNamespace(line=line("second sentence")))
    listener.timer.cancel()

    listener._flush(listener.timer_generation)

    assert listener.submitted == [("User Voice", "first sentence second sentence")]
    assert listener.recorded.finished == ["User Voice"]
    assert listener.pending == []


def test_a_flush_from_a_superseded_timer_is_ignored(listener) -> None:
    """A cancelled timer that already fired must not submit a stale turn."""
    listener.on_line_completed(SimpleNamespace(line=line("buffered")))
    listener.timer.cancel()
    stale_generation = listener.timer_generation

    listener._cancel_timer()
    listener._flush(stale_generation)

    assert listener.submitted == []
    assert listener.pending == ["buffered"]


def test_resumed_speech_cancels_the_pending_flush(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("half a thought")))
    listener.timer.cancel()

    listener.on_line_started(SimpleNamespace(line=line()))

    assert listener.timer is None
    assert listener.pending == ["half a thought"]
    assert listener.speech_callback_triggered is False


def test_a_partial_updates_the_display_and_reports_speech_once(listener) -> None:
    calls: list[str] = []
    listener.on_speech = lambda partial: calls.append(partial) or True

    listener.on_line_text_changed(SimpleNamespace(line=line("still talk")))
    listener.on_line_text_changed(SimpleNamespace(line=line("still talking")))

    assert listener.recorded.partials == [
        ("User Voice", "still talk"),
        ("User Voice", "still talking"),
    ]
    assert calls == ["still talk"]


def test_speech_is_reported_again_when_the_callback_declines(listener) -> None:
    """A partial the engine judged an echo must not suppress the next one."""
    calls: list[str] = []
    listener.on_speech = lambda partial: bool(calls.append(partial))

    listener.on_line_text_changed(SimpleNamespace(line=line("echo one")))
    listener.on_line_text_changed(SimpleNamespace(line=line("echo two")))

    assert calls == ["echo one", "echo two"]


def test_an_empty_partial_never_reports_speech(listener) -> None:
    calls: list[str] = []
    listener.on_speech = lambda partial: calls.append(partial) or True

    listener.on_line_text_changed(SimpleNamespace(line=line("   ")))

    assert calls == []


def test_low_confidence_words_are_dropped_from_a_turn(listener) -> None:
    spoken = line(
        text="fallback",
        words=((" keep ", 0.8), ("discard", 0.2), ("this", 0.9)),
    )

    assert listener._text(spoken) == "keep this"


def test_a_line_without_word_confidences_falls_back_to_its_text(listener) -> None:
    assert listener._text(line("  plain text  ")) == "plain text"


def test_an_empty_completed_line_does_not_start_a_turn(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("   ")))

    assert listener.recorded.commits == []
    assert listener.timer is None


def test_muting_discards_the_buffer_and_every_later_event(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("before the mute")))
    listener.timer.cancel()

    listener.set_muted(True)
    listener.on_line_started(SimpleNamespace(line=line()))
    listener.on_line_text_changed(SimpleNamespace(line=line("ignored partial")))
    listener.on_line_completed(SimpleNamespace(line=line("ignored turn")))

    assert listener.pending == []
    assert listener.submitted == []
    assert listener.recorded.partials == []
    assert listener.recorded.commits == [("User Voice", "before the mute")]


def test_unmuting_resumes_submitting_turns(listener) -> None:
    listener.set_muted(True)
    listener.set_muted(False)

    listener.on_line_completed(SimpleNamespace(line=line("back on")))
    listener.timer.cancel()
    listener._flush(listener.timer_generation)

    assert listener.submitted == [("User Voice", "back on")]


def test_closing_cancels_the_timer_and_closes_the_speaker(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("never flushed")))
    pending_timer = listener.timer

    listener.close()

    assert listener.timer is None
    assert pending_timer.finished.is_set() or not pending_timer.is_alive()
    assert listener.recorded.closed == ["User Voice"]
    assert listener.submitted == []


def test_closing_twice_is_harmless(listener) -> None:
    listener.close()
    listener.close()

    assert listener.recorded.closed == ["User Voice", "User Voice"]


class RecordingCountdown:
    """Record which speakers the listener reports as waiting."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def started(self, speaker):
        self.events.append(("started", speaker))

    def cleared(self, speaker):
        self.events.append(("cleared", speaker))

    @property
    def waiting(self):
        """The speakers still counting down, in the order they started."""
        pending: list[str] = []
        for action, speaker in self.events:
            if action == "started" and speaker not in pending:
                pending.append(speaker)
            elif action == "cleared" and speaker in pending:
                pending.remove(speaker)
        return pending


def counting_listener(countdown=None):
    """A listener reporting its silence timer to a recording countdown."""
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    countdown = RecordingCountdown() if countdown is None else countdown
    listener = RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=3.0,
        speaker="User Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        countdown=countdown,
    )
    return listener, countdown


def test_a_completed_line_starts_the_countdown() -> None:
    listener, countdown = counting_listener()

    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    assert countdown.waiting == ["User Voice"]
    listener.close()


def test_resuming_speech_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener.on_line_started(SimpleNamespace())

    assert countdown.waiting == []


def test_a_partial_transcript_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener.on_line_text_changed(SimpleNamespace(line=line("still talking")))

    assert countdown.waiting == []


def test_a_submitted_turn_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener._flush(listener.timer_generation)

    assert countdown.waiting == []
    assert listener.submitted == [("User Voice", "all done")]


def test_muting_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener.set_muted(True)

    assert countdown.waiting == []


def test_unmuting_does_not_resurrect_a_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))
    listener.set_muted(True)

    listener.set_muted(False)

    assert countdown.waiting == []


def test_closing_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener.close()

    assert countdown.waiting == []


def test_a_listener_without_a_countdown_still_completes_a_turn(listener) -> None:
    """The countdown is a display concern; a listener must not need one."""
    assert listener.countdown is None

    listener.on_line_completed(SimpleNamespace(line=line("all done")))
    listener._flush(listener.timer_generation)

    assert listener.submitted == [("User Voice", "all done")]
    listener.close()
