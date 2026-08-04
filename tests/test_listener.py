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

from tagalong.domain import (
    CLEAR_TURN_SILENCE,
    RESPONSE_POLICIES,
    SpeakerGate,
    TurnSilence,
)
from tagalong.listener import ConversationListener, TranscriptSubmitter


class RecordingDisplay:
    """Capture what the listener asks the transcript to show."""

    def __init__(self):
        self.partials: list[tuple[str, str]] = []
        self.commits: list[tuple[str, str]] = []
        self.finished: list[str] = []
        self.rejected: list[str] = []
        self.closed: list[str] = []

    def update(self, speaker, text):
        self.partials.append((speaker, text))

    def commit(self, speaker, text):
        self.commits.append((speaker, text))

    def finish_turn(self, speaker, accepted=True):
        self.finished.append(speaker)
        if not accepted:
            self.rejected.append(speaker)

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
        turn_silence=TurnSilence(3.0),
        speaker="Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        on_speech=lambda partial: True,
    )


def test_a_completed_line_is_committed_and_starts_the_silence_timer(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("hello there")))

    assert listener.recorded.commits == [("Voice", "hello there")]
    assert listener.timer is not None
    listener.timer.cancel()


def test_the_silence_timer_flushes_every_buffered_line_as_one_turn(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("first sentence")))
    listener.on_line_completed(SimpleNamespace(line=line("second sentence")))
    listener.timer.cancel()

    listener._flush(listener.timer_generation)

    assert listener.submitted == [("Voice", "first sentence second sentence")]
    assert listener.recorded.finished == ["Voice"]
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


def test_flushing_now_submits_the_buffer_without_waiting_for_silence(
    listener,
) -> None:
    """The buffered turn goes out at once, rather than at the timer's deadline."""
    listener.on_line_completed(SimpleNamespace(line=line("worth knowing")))
    pending_timer = listener.timer

    listener.flush_now()

    assert listener.submitted == [("Voice", "worth knowing")]
    assert listener.recorded.finished == ["Voice"]
    assert listener.pending == []
    assert listener.timer is None
    assert pending_timer.finished.is_set()


def test_an_immediate_flush_leaves_nothing_for_the_timer_to_resubmit(
    listener,
) -> None:
    """The superseded timer must not send the same words a second time."""
    listener.on_line_completed(SimpleNamespace(line=line("said once")))
    stale_generation = listener.timer_generation

    listener.flush_now()
    listener._flush(stale_generation)

    assert listener.submitted == [("Voice", "said once")]


def test_flushing_an_empty_buffer_now_submits_nothing(listener) -> None:
    listener.flush_now()

    assert listener.submitted == []
    assert listener.recorded.finished == []


def test_an_immediate_flush_stops_the_countdown() -> None:
    listener, countdown = counting_listener()
    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    listener.flush_now()

    assert countdown.waiting == []


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
        ("Voice", "still talk"),
        ("Voice", "still talking"),
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
    assert listener.recorded.commits == [("Voice", "before the mute")]


def test_unmuting_resumes_submitting_turns(listener) -> None:
    listener.set_muted(True)
    listener.set_muted(False)

    listener.on_line_completed(SimpleNamespace(line=line("back on")))
    listener.timer.cancel()
    listener._flush(listener.timer_generation)

    assert listener.submitted == [("Voice", "back on")]


def test_closing_cancels_the_timer_and_closes_the_speaker(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("never flushed")))
    pending_timer = listener.timer

    listener.close()

    assert listener.timer is None
    assert pending_timer.finished.is_set() or not pending_timer.is_alive()
    assert listener.recorded.closed == ["Voice"]
    assert listener.submitted == []


def test_closing_twice_is_harmless(listener) -> None:
    listener.close()
    listener.close()

    assert listener.recorded.closed == ["Voice", "Voice"]


class RecordingCountdown:
    """Record which speakers the listener reports as waiting."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.windows: list[float | None] = []

    def started(self, speaker, seconds=None):
        self.events.append(("started", speaker))
        self.windows.append(seconds)

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
        turn_silence=TurnSilence(3.0),
        speaker="Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        countdown=countdown,
    )
    return listener, countdown


def test_a_completed_line_starts_the_countdown() -> None:
    listener, countdown = counting_listener()

    listener.on_line_completed(SimpleNamespace(line=line("all done")))

    assert countdown.waiting == ["Voice"]
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
    assert listener.submitted == [("Voice", "all done")]


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

    assert listener.submitted == [("Voice", "all done")]
    listener.close()


class RecordingConversation:
    """Record what reached Codex, in the order the request would carry it."""

    def __init__(self):
        self.ingested: list[tuple[str, str, bool]] = []

    def ingest(self, speaker, text, respond):
        self.ingested.append((speaker, text, respond))


def two_channels(policy):
    """Two live listeners sharing one submitter, as a session wires them."""
    conversation = RecordingConversation()
    gate = SpeakerGate(RESPONSE_POLICIES[policy].speakers, {"Voice", "Audio"})
    submitter = TranscriptSubmitter(conversation, gate, None)
    display = RecordingDisplay()
    listeners = {
        speaker: ConversationListener(
            confidence_threshold=0.6,
            turn_silence=TurnSilence(3.0),
            speaker=speaker,
            submit=submitter.submit,
            presentation=display,
        )
        for speaker in ("Voice", "Audio")
    }
    for listener in listeners.values():
        submitter.add_listener(listener)
    return conversation, listeners


def test_answering_audio_takes_the_voice_context_transcribed_so_far() -> None:
    """The reply goes out on Audio's silence, carrying words Voice has
    already said but not yet finished waiting out."""
    conversation, listeners = two_channels("audio")
    listeners["Voice"].on_line_completed(
        SimpleNamespace(line=line("check the latency"))
    )
    listeners["Audio"].on_line_completed(
        SimpleNamespace(line=line("what do you think"))
    )
    listeners["Audio"].timer.cancel()

    listeners["Audio"]._flush(listeners["Audio"].timer_generation)

    assert conversation.ingested == [
        ("Voice", "check the latency", False),
        ("Audio", "what do you think", True),
    ]
    assert listeners["Voice"].timer is None
    assert listeners["Voice"].pending == []


def test_answering_voice_takes_the_audio_context_transcribed_so_far() -> None:
    conversation, listeners = two_channels("voice")
    listeners["Audio"].on_line_completed(SimpleNamespace(line=line("the build is red")))
    listeners["Voice"].on_line_completed(SimpleNamespace(line=line("why is that")))
    listeners["Voice"].timer.cancel()

    listeners["Voice"]._flush(listeners["Voice"].timer_generation)

    assert conversation.ingested == [
        ("Audio", "the build is red", False),
        ("Voice", "why is that", True),
    ]


def test_a_silent_other_channel_adds_nothing_to_the_reply() -> None:
    conversation, listeners = two_channels("audio")
    listeners["Audio"].on_line_completed(
        SimpleNamespace(line=line("what do you think"))
    )
    listeners["Audio"].timer.cancel()

    listeners["Audio"]._flush(listeners["Audio"].timer_generation)

    assert conversation.ingested == [("Audio", "what do you think", True)]


# --------------------------------------------------------------------------
# Answering before the window closes
# --------------------------------------------------------------------------


class RecordingPrefire:
    """Stand in for the conversation, recording each moment as it is asked for."""

    def __init__(self, delay=1.0, accepted=True):
        self.configured_delay = delay
        self.accepted = accepted
        self.started: list[tuple[str, str]] = []
        self.committed: list[str] = []
        self.cancelled: list[str] = []
        self.commit_result = True

    def delay(self, window):
        return min(self.configured_delay, window)

    def start(self, speaker, text):
        self.started.append((speaker, text))
        return self.accepted

    def commit(self, speaker):
        self.committed.append(speaker)
        return self.commit_result

    def cancel(self, speaker):
        self.cancelled.append(speaker)
        return True


@pytest.fixture
def prefire():
    return RecordingPrefire()


@pytest.fixture
def guessing(prefire):
    """A listener that guesses a turn is over one second into a three-second wait."""
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    return RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=TurnSilence(3.0),
        speaker="Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        on_speech=lambda partial: True,
        prefire=prefire,
    )


def test_a_completed_line_arms_both_the_guess_and_the_deadline(guessing) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("hello there")))

    assert guessing.prefire_timer is not None
    assert guessing.timer is not None
    assert guessing.prefire_timer.interval < guessing.timer.interval
    guessing.prefire_timer.cancel()
    guessing.timer.cancel()


def test_the_guess_starts_a_turn_from_what_is_buffered_so_far(
    guessing, prefire
) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("a question")))
    guessing.prefire_timer.cancel()
    guessing.timer.cancel()

    guessing._speculate(guessing.timer_generation)

    assert prefire.started == [("Voice", "a question")]
    assert guessing.prefired is True
    # The buffer is untouched: the deadline still owns the turn until it fires.
    assert guessing.pending == ["a question"]


def test_a_guess_that_was_right_becomes_the_reply(guessing, prefire) -> None:
    """The window closed without the speaker resuming, so the turn stands."""
    guessing.on_line_completed(SimpleNamespace(line=line("a question")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)

    guessing._flush(guessing.timer_generation)

    assert prefire.committed == ["Voice"]
    assert guessing.submitted == []
    assert guessing.recorded.finished == ["Voice"]


def test_a_refused_commit_still_submits_the_turn(guessing, prefire) -> None:
    """The conversation is the authority; a listener told no must not go quiet."""
    guessing.on_line_completed(SimpleNamespace(line=line("a question")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)
    prefire.commit_result = False

    guessing._flush(guessing.timer_generation)

    assert prefire.committed == ["Voice"]
    assert guessing.submitted == [("Voice", "a question")]


def test_resumed_speech_abandons_the_guess(guessing, prefire) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("half a thought")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)

    guessing.on_line_text_changed(SimpleNamespace(line=line("and the rest")))

    assert prefire.cancelled == ["Voice"]
    assert guessing.prefired is False


def test_the_whole_turn_is_submitted_after_a_wrong_guess(guessing, prefire) -> None:
    """A cancelled guess must cost the words it guessed from, not lose them."""
    guessing.on_line_completed(SimpleNamespace(line=line("half a thought")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)
    guessing.on_line_text_changed(SimpleNamespace(line=line("and the rest")))
    guessing.on_line_completed(SimpleNamespace(line=line("and the rest")))
    guessing.prefire_timer.cancel()

    guessing._flush(guessing.timer_generation)

    assert guessing.submitted == [("Voice", "half a thought and the rest")]
    assert prefire.committed == []


def test_muting_abandons_the_guess(guessing, prefire) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("half a thought")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)

    guessing.set_muted(True)

    assert prefire.cancelled == ["Voice"]
    assert guessing.timer is None
    assert guessing.prefire_timer is None


def test_closing_abandons_the_guess(guessing, prefire) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("half a thought")))
    guessing.prefire_timer.cancel()
    guessing._speculate(guessing.timer_generation)

    guessing.close()

    assert prefire.cancelled == ["Voice"]


def test_a_guess_from_a_superseded_timer_is_ignored(guessing, prefire) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("buffered")))
    guessing.prefire_timer.cancel()
    stale_generation = guessing.timer_generation
    guessing._cancel_timer()

    guessing._speculate(stale_generation)

    assert prefire.started == []


def test_a_refused_guess_leaves_nothing_to_commit(guessing, prefire) -> None:
    """A turn the conversation declined to start is not one it can adopt."""
    prefire.accepted = False
    guessing.on_line_completed(SimpleNamespace(line=line("an echo")))
    guessing.prefire_timer.cancel()

    guessing._speculate(guessing.timer_generation)
    guessing._flush(guessing.timer_generation)

    assert guessing.prefired is False
    assert prefire.committed == []
    assert guessing.submitted == [("Voice", "an echo")]


def test_the_same_turn_is_only_guessed_at_once(guessing, prefire) -> None:
    guessing.on_line_completed(SimpleNamespace(line=line("a question")))
    guessing.prefire_timer.cancel()

    guessing._speculate(guessing.timer_generation)
    guessing._speculate(guessing.timer_generation)

    assert prefire.started == [("Voice", "a question")]


def test_an_empty_buffer_is_not_guessed_at(guessing, prefire) -> None:
    guessing._speculate(guessing.timer_generation)

    assert prefire.started == []


def test_a_listener_without_prefire_arms_only_the_deadline(listener) -> None:
    listener.on_line_completed(SimpleNamespace(line=line("hello there")))

    assert listener.prefire_timer is None
    assert listener.timer is not None
    listener.timer.cancel()


def test_a_guess_landing_no_sooner_than_the_deadline_is_not_made(guessing) -> None:
    """Guessing at the deadline is a slower way of doing what the deadline does."""
    guessing.prefire.configured_delay = 3.0

    guessing.on_line_completed(SimpleNamespace(line=line("hello there")))

    assert guessing.prefire_timer is None
    assert guessing.timer is not None
    guessing.timer.cancel()


# --------------------------------------------------------------------------
# What a guess has to pass before it reaches Codex
# --------------------------------------------------------------------------


class PrefiringConversation(RecordingConversation):
    """Record guesses alongside settled turns, as the conversation sees them."""

    def __init__(self):
        super().__init__()
        self.prefired: list[tuple[str, str]] = []

    def prefire(self, speaker, text):
        self.prefired.append((speaker, text))
        return True

    def commit_prefire(self, _speaker):
        return True

    def cancel_prefire(self, _speaker):
        return True


class EchoingTTS:
    """Speech that claims one particular transcript is its own echo."""

    def __init__(self, echo):
        self.echo = echo

    def is_likely_echo(self, text):
        return text == self.echo

    def interrupt(self):
        return None


def prefiring_submitter(policy="audio", tts=None):
    conversation = PrefiringConversation()
    gate = SpeakerGate(RESPONSE_POLICIES[policy].speakers, {"Voice", "Audio"})
    submitter = TranscriptSubmitter(
        conversation, gate, tts, prefire_plan=RecordingPrefire()
    )
    return conversation, submitter


def test_a_guess_reaches_codex_when_the_policy_answers_that_speaker() -> None:
    conversation, submitter = prefiring_submitter()

    assert submitter.prefire("Audio", "what do you think")
    assert conversation.prefired == [("Audio", "what do you think")]


def test_a_speaker_the_policy_stays_silent_for_is_never_guessed_at() -> None:
    """A late reply nobody wanted is caught downstream; an early one is not."""
    conversation, submitter = prefiring_submitter(policy="audio")

    assert not submitter.prefire("Voice", "thinking out loud")
    assert conversation.prefired == []


def test_codex_hearing_itself_is_not_guessed_at() -> None:
    conversation, submitter = prefiring_submitter(tts=EchoingTTS("my own words"))

    assert not submitter.prefire("Audio", "my own words")
    assert conversation.prefired == []


def test_a_guess_sweeps_the_context_channels_first() -> None:
    """The reply being guessed at must carry what the other channel has said."""
    conversation = PrefiringConversation()
    gate = SpeakerGate(RESPONSE_POLICIES["audio"].speakers, {"Voice", "Audio"})
    submitter = TranscriptSubmitter(
        conversation, gate, None, prefire_plan=RecordingPrefire()
    )
    display = RecordingDisplay()
    user = ConversationListener(
        confidence_threshold=0.6,
        turn_silence=TurnSilence(3.0),
        speaker="Voice",
        submit=submitter.submit,
        presentation=display,
    )
    submitter.add_listener(user)
    user.on_line_completed(SimpleNamespace(line=line("check the latency")))
    assert user.timer is not None
    user.timer.cancel()

    submitter.prefire("Audio", "what do you think")

    assert conversation.ingested == [("Voice", "check the latency", False)]
    assert conversation.prefired == [("Audio", "what do you think")]


def test_a_session_without_a_plan_builds_channels_that_never_guess() -> None:
    conversation = PrefiringConversation()
    gate = SpeakerGate(RESPONSE_POLICIES["audio"].speakers, {"Audio"})
    submitter = TranscriptSubmitter(conversation, gate, None)

    listener = submitter.channel(0.6, TurnSilence(3.0), "Audio", RecordingDisplay())

    assert listener.prefire is None


def test_a_session_with_a_plan_builds_channels_that_guess() -> None:
    conversation = PrefiringConversation()
    gate = SpeakerGate(RESPONSE_POLICIES["audio"].speakers, {"Audio"})
    submitter = TranscriptSubmitter(
        conversation, gate, None, prefire_plan=RecordingPrefire()
    )

    listener = submitter.channel(0.6, TurnSilence(3.0), "Audio", RecordingDisplay())

    assert listener.prefire is not None
    assert listener.prefire.delay(3.0) == pytest.approx(1.0)


def test_the_prefire_channel_passes_each_moment_to_the_submitter() -> None:
    """The seam is a pass-through; a dropped call would silently stop guessing."""
    conversation, submitter = prefiring_submitter()
    channel = submitter.channel(
        0.6, TurnSilence(3.0), "Audio", RecordingDisplay()
    ).prefire

    assert channel.start("Audio", "what do you think")
    assert channel.commit("Audio")
    assert channel.cancel("Audio")
    assert conversation.prefired == [("Audio", "what do you think")]


# --------------------------------------------------------------------------
# Holding a turn open for someone who is still talking
#
# Transcription arrives about half a second behind the speech, so a speaker
# who resumes just before the deadline has nothing on the transcript to save
# them. The level tap answers from the audio instead. These hold both halves:
# that a turn is kept open while its speaker is audible, and that it is still
# always sent, however long the channel refuses to go quiet.
# --------------------------------------------------------------------------


class StubPresence:
    """Report whatever a test says the level tap is hearing."""

    def __init__(self, speaking=True):
        self.answer = speaking
        self.asked = 0

    def speaking(self):
        self.asked += 1
        return self.answer


def listening_for_speech(speaking=True, window=3.0, **kwargs):
    """A listener that consults a level tap when its deadline arrives."""
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    presence = StubPresence(speaking)
    listener = RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=TurnSilence(window),
        speaker="Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
        presence=presence,
        **kwargs,
    )
    return listener, presence


def reach_deadline(listener):
    """Fire the pending deadline the way the timer thread would."""
    listener.timer.cancel()
    listener._flush(listener.timer_generation)


def test_a_deadline_reached_while_the_speaker_is_audible_sends_nothing() -> None:
    listener, _ = listening_for_speech(speaking=True)
    listener.on_line_completed(SimpleNamespace(line=line("I was going to say")))

    reach_deadline(listener)

    assert listener.submitted == []
    assert listener.pending == ["I was going to say"]
    listener.close()


def test_a_held_turn_is_re_armed_on_the_grace_rather_than_the_window() -> None:
    """A whole window each time would answer a finished speaker far too late."""
    listener, _ = listening_for_speech(speaking=True, window=3.0)
    listener.on_line_completed(SimpleNamespace(line=line("one moment")))

    reach_deadline(listener)

    assert listener.timer is not None
    assert listener.timer.interval == ConversationListener.EXTENSION_GRACE
    listener.close()


def test_a_deadline_reached_in_silence_sends_the_turn_as_before() -> None:
    listener, presence = listening_for_speech(speaking=False)
    listener.on_line_completed(SimpleNamespace(line=line("that is all")))

    reach_deadline(listener)

    assert listener.submitted == [("Voice", "that is all")]
    assert presence.asked == 1
    listener.close()


def test_a_turn_held_open_puts_the_grace_back_on_the_countdown() -> None:
    """The wait visibly returns, which is what says the speaker was heard."""
    countdown = RecordingCountdown()
    listener, _ = listening_for_speech(speaking=True, countdown=countdown)
    listener.on_line_completed(SimpleNamespace(line=line("hold on")))

    reach_deadline(listener)

    assert countdown.waiting == ["Voice"]
    assert countdown.windows == [3.0, ConversationListener.EXTENSION_GRACE]
    listener.close()


def test_a_channel_that_never_goes_quiet_still_sends_its_turn() -> None:
    """A fan is sound too, and a room loud enough would never submit at all."""
    listener, _ = listening_for_speech(speaking=True, window=3.0)
    listener.on_line_completed(SimpleNamespace(line=line("said once")))

    for _ in range(listener._extension_budget()):
        reach_deadline(listener)
        assert listener.submitted == []
    reach_deadline(listener)

    assert listener.submitted == [("Voice", "said once")]
    listener.close()


def test_a_turn_is_never_held_open_for_longer_than_its_own_window() -> None:
    """The budget scales with the wait, so a short window stays short."""
    listener, _ = listening_for_speech(window=3.0)
    brief, _ = listening_for_speech(window=0.5)

    assert listener._extension_budget() == 6
    assert brief._extension_budget() == 1
    listener.close()
    brief.close()


def test_even_a_window_shorter_than_the_grace_may_be_held_open_once() -> None:
    """Rounding the budget to zero would switch the whole thing off."""
    listener, _ = listening_for_speech(window=TurnSilence.MINIMUM)

    assert listener._extension_budget() == 1
    listener.close()


def test_transcription_catching_up_gives_the_budget_back() -> None:
    """The extensions bought the words that have now arrived."""
    listener, _ = listening_for_speech(speaking=True)
    listener.on_line_completed(SimpleNamespace(line=line("first")))
    reach_deadline(listener)
    assert listener.extensions == 1

    listener.on_line_text_changed(SimpleNamespace(line=line("and then")))

    assert listener.extensions == 0
    listener.close()


def test_an_empty_buffer_is_never_held_open() -> None:
    """There is no turn to protect, and the countdown would never clear."""
    listener, presence = listening_for_speech(speaking=True)
    listener.on_line_completed(SimpleNamespace(line=line("")))

    listener._flush(listener.timer_generation)

    assert listener.timer is None
    assert presence.asked == 0
    listener.close()


def test_an_immediate_flush_is_never_held_open() -> None:
    """The caller has already decided not to wait for this speaker."""
    listener, presence = listening_for_speech(speaking=True)
    listener.on_line_completed(SimpleNamespace(line=line("context for the reply")))

    listener.flush_now()

    assert listener.submitted == [("Voice", "context for the reply")]
    assert presence.asked == 0
    listener.close()


def test_holding_a_turn_open_abandons_the_guess_it_had_already_made() -> None:
    """A guess made mid-sentence is answering half of one."""
    guess = RecordingPrefire()
    listener, _ = listening_for_speech(speaking=True, prefire=guess)
    listener.on_line_completed(SimpleNamespace(line=line("what do you think")))
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    assert guess.started == [("Voice", "what do you think")]

    reach_deadline(listener)

    assert guess.cancelled == ["Voice"]
    assert listener.prefired is False
    assert guess.committed == []
    listener.close()


def test_a_held_turn_does_not_guess_again_during_its_grace() -> None:
    """The reason it was held is that the speaker is probably mid-sentence."""
    listener, _ = listening_for_speech(speaking=True, prefire=RecordingPrefire())
    listener.on_line_completed(SimpleNamespace(line=line("still going")))
    listener.prefire_timer.cancel()

    reach_deadline(listener)

    assert listener.prefire_timer is None
    listener.close()


def test_muting_forgets_what_a_turn_was_held_open_on() -> None:
    listener, _ = listening_for_speech(speaking=True)
    listener.on_line_completed(SimpleNamespace(line=line("mid sentence")))
    reach_deadline(listener)
    assert listener.extensions == 1

    listener.set_muted(True)

    assert listener.extensions == 0
    listener.close()


def test_retiring_a_channel_that_was_never_registered_is_quiet() -> None:
    """A far end can be dropped before its listener ever reached the submitter."""
    submitter = TranscriptSubmitter(None, SpeakerGate({"Audio"}, {"Audio"}), None)
    submitter.add_listener("kept")

    submitter.remove_listener("never added")

    assert submitter.listeners == ["kept"]


# --------------------------------------------------------------------------
# Energy-aware turn closure, adaptive silence, and partial prefires
# --------------------------------------------------------------------------


def test_energy_quiet_arms_silence_from_a_live_partial() -> None:
    """The wait starts when the tap falls, not when Moonshine closes the line."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    presence.answer = False
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga what's next?")))

    listener.on_energy_quiet()

    assert listener.timer is not None
    assert listener.timer.interval == CLEAR_TURN_SILENCE
    assert listener.pending == []
    assert listener._partial == "Taga what's next?"
    listener.close()


def test_late_partials_do_not_cancel_an_energy_armed_wait() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga status")))
    listener.on_energy_quiet()
    generation = listener.timer_generation

    # Tap still quiet: STT catching up must not throw the wait away.
    presence.answer = False
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga status now?")))

    assert listener.timer_generation == generation
    assert listener.timer is not None
    assert listener._partial == "Taga status now?"
    listener.close()


def test_a_quiet_line_start_keeps_an_energy_armed_wait() -> None:
    """Moonshine opening a new line after the tap fell is catch-up, not resume."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("first bit")))
    generation = listener.timer_generation
    presence.answer = False

    listener.on_line_started(SimpleNamespace(line=line()))

    assert listener.timer_generation == generation
    assert listener.timer is not None
    assert listener.pending == ["first bit"]
    listener.close()


def test_a_quiet_line_start_does_not_drop_a_partial_only_turn() -> None:
    """Clearing the live partial on quiet catch-up would flush an empty turn."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga status?")))
    listener.on_energy_quiet()
    presence.answer = False

    listener.on_line_started(SimpleNamespace(line=line()))
    listener.flush_now()

    assert listener.submitted == [("Voice", "Taga status?")]
    listener.close()


def test_energy_loud_cancels_a_wait_when_the_speaker_resumes() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("hold that thought")))
    presence.answer = True

    listener.on_energy_loud()

    assert listener.timer is None
    assert listener.pending == ["hold that thought"]
    listener.close()


def test_suppressed_loudness_does_not_cancel_a_wait() -> None:
    """TTS / far-end on the mic must not look like the user resuming."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("ready when you are")))
    assert listener.timer is not None
    generation = listener.timer_generation
    # Tap rose, but presence (suppressors) still says this is not the speaker.
    presence.answer = False

    listener.on_energy_loud()

    assert listener.timer_generation == generation
    assert listener.timer is not None
    listener.close()


def test_a_clear_question_uses_the_short_silence_window() -> None:
    listener, _ = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("What is the status?")))

    assert listener.timer.interval == CLEAR_TURN_SILENCE
    listener.close()


def test_an_incomplete_tail_keeps_the_full_silence_window() -> None:
    listener, _ = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("I was going to say and")))

    assert listener.timer.interval == 1.25
    listener.close()


def test_energy_quiet_prefires_the_live_partial() -> None:
    guess = RecordingPrefire(delay=0.2)
    listener, _ = listening_for_speech(speaking=False, window=1.25, prefire=guess)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga ship it?")))
    listener.on_energy_quiet()

    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)

    assert guess.started == [("Voice", "Taga ship it?")]
    assert listener.prefired is True
    listener.close()


def test_a_material_stt_revision_restarts_a_stale_prefire() -> None:
    guess = RecordingPrefire(delay=0.2)
    listener, presence = listening_for_speech(
        speaking=False, window=1.25, prefire=guess
    )
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga open one")))
    listener.on_energy_quiet()
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    assert guess.started == [("Voice", "Taga open one")]

    presence.answer = False
    listener.on_line_completed(SimpleNamespace(line=line("Taga open two files?")))

    assert guess.cancelled == ["Voice"]
    assert listener.prefired is False
    # Fresh speculate is armed on the revised text.
    assert listener.prefire_timer is not None
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    assert guess.started[-1] == ("Voice", "Taga open two files?")
    listener.close()


def test_flushing_a_partial_only_turn_commits_it_to_the_transcript() -> None:
    listener, _ = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("send this now")))
    listener.on_energy_quiet()

    listener.flush_now()

    assert listener.submitted == [("Voice", "send this now")]
    assert listener.recorded.commits == [("Voice", "send this now")]
    assert listener.recorded.finished == ["Voice"]
    listener.close()


def test_end_turn_flushes_the_named_speaker() -> None:
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    listener = RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=TurnSilence(1.25),
        speaker="Voice",
        submit=lambda speaker, text: submitted.append((speaker, text)),
        presentation=display,
    )
    submitter = TranscriptSubmitter(None, SpeakerGate({"Voice"}, {"Voice"}), None)
    submitter.add_listener(listener)
    listener.on_line_completed(SimpleNamespace(line=line("done talking")))

    submitter.end_turn("Audio")
    assert submitted == []

    submitter.end_turn("Voice")

    assert submitted == [("Voice", "done talking")]
    listener.close()


def test_energy_hooks_ignore_mute_and_empty_buffers() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.set_muted(True)
    listener.on_energy_quiet()
    listener.on_energy_loud()
    assert listener.timer is None

    listener.set_muted(False)
    presence.answer = True
    listener.on_energy_quiet()  # still audible — do not arm
    assert listener.timer is None

    presence.answer = False
    listener.on_energy_quiet()  # quiet but nothing buffered
    assert listener.timer is None
    listener.close()


def test_energy_quiet_keeps_a_matching_prefire() -> None:
    """A quiet re-arm with the same text must not cancel a good guess."""
    guess = RecordingPrefire(delay=0.2)
    listener, _ = listening_for_speech(speaking=False, window=1.25, prefire=guess)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga ship it?")))
    listener.on_energy_quiet()
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    assert listener.prefired is True
    generation = listener.timer_generation

    listener.on_energy_quiet()

    assert guess.cancelled == []
    assert listener.prefired is True
    assert listener.timer_generation == generation
    listener.close()


def test_a_late_line_close_after_partial_flush_is_not_a_second_turn() -> None:
    """Energy flush of a partial must not let Moonshine's close re-submit it."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("send this now")))
    listener.on_energy_quiet()
    listener.flush_now()
    assert listener.submitted == [("Voice", "send this now")]

    presence.answer = False
    listener.on_line_completed(SimpleNamespace(line=line("send this now")))

    assert listener.submitted == [("Voice", "send this now")]
    assert listener.timer is None
    assert listener.pending == []
    listener.close()


def test_a_late_close_is_absorbed_even_after_the_speaker_resumes() -> None:
    """Matching STT catch-up must not become a new turn if the user spoke again."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("first question")))
    listener.flush_now()
    presence.answer = True
    listener.on_energy_loud()
    listener.on_line_text_changed(SimpleNamespace(line=line("and a follow up")))

    presence.answer = False
    listener.on_line_completed(SimpleNamespace(line=line("first question")))

    assert listener.submitted == [("Voice", "first question")]
    assert listener.pending == []
    assert listener._partial == "and a follow up"
    assert listener.timer is None
    listener.close()


def test_flushed_catch_up_does_not_block_new_speech_under_tts_suppressors() -> None:
    """Presence quiet from TTS must not swallow a new utterance after a flush."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("already sent")))
    listener.flush_now()
    # TTS / far-end suppressors keep presence quiet even while the user talks.
    presence.answer = False

    listener.on_line_text_changed(SimpleNamespace(line=line("brand new question?")))
    listener.on_energy_quiet()

    assert listener.timer is not None
    assert listener._partial == "brand new question?"
    listener.flush_now()
    assert listener.submitted == [
        ("Voice", "already sent"),
        ("Voice", "brand new question?"),
    ]
    listener.close()


def test_late_partials_matching_a_flush_do_not_rearm_silence() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("already sent")))
    listener.flush_now()
    presence.answer = False
    generation_before = listener.timer_generation

    listener.on_energy_quiet()
    listener.on_line_text_changed(SimpleNamespace(line=line("already sent")))

    assert listener.timer is None
    assert listener.timer_generation == generation_before
    assert listener.submitted == [("Voice", "already sent")]
    listener.close()


def test_energy_quiet_ignores_a_buffer_that_only_echoes_the_flush() -> None:
    """If STT puts the flushed words back, quiet must not start another wait."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("already sent")))
    listener.flush_now()
    presence.answer = False
    with listener.lock:
        listener._partial = "already sent"

    listener.on_energy_quiet()

    assert listener.timer is None
    assert listener.submitted == [("Voice", "already sent")]
    listener.close()


def test_late_close_with_added_punctuation_is_still_absorbed() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("what's the status")))
    listener.flush_now()
    presence.answer = False

    listener.on_line_completed(SimpleNamespace(line=line("what's the status?")))

    assert listener.submitted == [("Voice", "what's the status")]
    assert listener.pending == []
    assert listener.timer is None
    listener.close()


def test_punctuation_only_stt_is_not_treated_as_flush_catch_up() -> None:
    listener, _ = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("hello")))
    listener.flush_now()
    with listener.lock:
        assert listener._late_stt_for_flush("???") is False
    listener.close()


def test_flushing_pending_plus_partial_absorbs_the_trailing_segment() -> None:
    """A late close of only the partial must not become a second turn."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_completed(SimpleNamespace(line=line("first sentence")))
    listener.on_line_text_changed(SimpleNamespace(line=line("trailing bit")))
    listener.flush_now()
    assert listener.submitted == [("Voice", "first sentence trailing bit")]
    presence.answer = False

    listener.on_line_completed(SimpleNamespace(line=line("trailing bit")))

    assert listener.submitted == [("Voice", "first sentence trailing bit")]
    assert listener.pending == []
    assert listener.timer is None
    listener.close()


def test_a_quiet_partial_that_becomes_a_clear_ending_shortens_the_wait() -> None:
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("I was going to say and")))
    listener.on_energy_quiet()
    assert listener.timer.interval == 1.25
    presence.answer = False

    listener.on_line_text_changed(SimpleNamespace(line=line("Taga status?")))

    assert listener.timer is not None
    assert listener.timer.interval == CLEAR_TURN_SILENCE
    listener.close()


def test_completing_the_same_words_after_a_partial_prefire_keeps_the_guess() -> None:
    """Matching line close must not cancel a prefire that already had those words."""
    guess = RecordingPrefire(delay=0.2)
    listener, presence = listening_for_speech(
        speaking=False, window=1.25, prefire=guess
    )
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga ship it?")))
    listener.on_energy_quiet()
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    presence.answer = False

    listener.on_line_completed(SimpleNamespace(line=line("Taga ship it?")))

    assert guess.cancelled == []
    assert listener.prefired is True
    listener.close()


def test_a_quiet_partial_revision_cancels_a_stale_prefire() -> None:
    guess = RecordingPrefire(delay=0.2)
    listener, presence = listening_for_speech(
        speaking=False, window=1.25, prefire=guess
    )
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga open one")))
    listener.on_energy_quiet()
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)

    presence.answer = False
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga open two?")))

    assert guess.cancelled == ["Voice"]
    assert listener.prefired is False
    assert listener.prefire_timer is not None
    listener.close()


def test_energy_quiet_cancels_when_the_buffer_diverged_offline() -> None:
    """A quiet re-arm after an out-of-band buffer edit drops the stale guess."""
    guess = RecordingPrefire(delay=0.2)
    listener, _ = listening_for_speech(speaking=False, window=1.25, prefire=guess)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga one")))
    listener.on_energy_quiet()
    listener.prefire_timer.cancel()
    listener._speculate(listener.timer_generation)
    with listener.lock:
        listener._partial = "Taga two files?"

    listener.on_energy_quiet()

    assert guess.cancelled == ["Voice"]
    listener.close()


def test_line_close_keeps_an_energy_armed_wait() -> None:
    """Completing the same clear turn must not restart the silence clock."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("Taga status?")))
    listener.on_energy_quiet()
    generation = listener.timer_generation
    interval = listener.timer.interval
    presence.answer = False

    listener.on_line_completed(SimpleNamespace(line=line("Taga status?")))

    assert listener.timer is not None
    assert listener.timer_generation == generation
    assert listener.timer.interval == interval
    assert listener.pending == ["Taga status?"]
    listener.close()


def test_a_rejected_echo_flush_does_not_arm_catch_up() -> None:
    """Echo drops must not leave a flush marker that blocks the real turn."""
    display = RecordingDisplay()
    submitted: list[tuple[str, str]] = []
    drop_once = {"echo": True}

    def submit(_speaker, text):
        if drop_once["echo"] and text == "please ignore this echo":
            drop_once["echo"] = False
            return False
        submitted.append((_speaker, text))
        return True

    presence = StubPresence(False)
    listener = RecordedListener(
        display,
        submitted,
        confidence_threshold=0.6,
        turn_silence=TurnSilence(1.25),
        speaker="Voice",
        submit=submit,
        presentation=display,
        presence=presence,
    )
    listener.on_line_text_changed(SimpleNamespace(line=line("please ignore this echo")))
    listener.flush_now()

    assert submitted == []
    assert display.rejected == ["Voice"]
    with listener.lock:
        assert listener._flushed_text == ""
        assert listener._flushed_partial == ""

    listener.on_line_text_changed(SimpleNamespace(line=line("please ignore this echo")))
    listener.on_energy_quiet()
    assert listener.timer is not None
    listener.flush_now()
    assert submitted == [("Voice", "please ignore this echo")]
    listener.close()


def test_a_new_phrase_that_is_a_flushed_suffix_is_not_absorbed() -> None:
    """Only exact flush / flushed-partial matches are catch-up, not suffixes."""
    listener, presence = listening_for_speech(speaking=False, window=1.25)
    listener.on_line_text_changed(SimpleNamespace(line=line("please open the file")))
    listener.flush_now()
    presence.answer = False

    listener.on_line_text_changed(SimpleNamespace(line=line("the file")))
    listener.on_energy_quiet()

    assert listener._partial == "the file"
    assert listener.timer is not None
    listener.flush_now()
    assert listener.submitted == [
        ("Voice", "please open the file"),
        ("Voice", "the file"),
    ]
    listener.close()
