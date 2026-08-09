from __future__ import annotations

import threading

import pytest

from tagalong.domain import (
    CLEAR_TURN_SILENCE,
    CodexRequest,
    EchoMatcher,
    EchoMemory,
    PrefirePlan,
    SentenceChunker,
    SpeakerGate,
    SpeakerPresence,
    SpeechActivity,
    TranscriptRouter,
    TurnGate,
    TurnLatencyEstimator,
    TurnSilence,
    TurnSilenceClock,
    markdown_to_speech,
    parse_turn_silence,
    resolve_response_policy,
    same_turn_text,
    silence_for_turn,
    speech_sink,
    strip_chrome,
)


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.mark.parametrize(
    ("requested", "label", "speakers"),
    [
        ("audio", "Audio", frozenset({"Audio"})),
        ("both", "Voice and Audio", frozenset({"Voice", "Audio"})),
        ("voice", "Voice", frozenset({"Voice"})),
        ("quiet", "Taga will be quiet for voice", frozenset()),
    ],
)
def test_response_policy_mapping(requested, label, speakers) -> None:
    policy = resolve_response_policy(requested)

    assert (policy.label, policy.speakers) == (label, speakers)


def test_sentence_chunker_preserves_sentence_order_and_size() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append, max_chars=12)

    chunker.feed("First line. A longer second sentence")
    chunker.flush()

    assert emitted == ["First line.", "A longer", "second", "sentence"]
    assert all(0 < len(chunk) <= 12 for chunk in emitted)


def chunks(text, max_chars, *, as_sentence):
    """Run text through one of the two chunking paths.

    A sentence-terminated feed routes through ``_emit_bounded``; a feed with
    no sentence end routes through ``_emit_long_chunks`` and leaves the tail
    in the buffer. Both split identically, so both are asserted.
    """
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append, max_chars=max_chars)
    chunker.feed(f"{text} " if as_sentence else text)
    return emitted, chunker.buffer


@pytest.mark.parametrize("as_sentence", [True, False])
@pytest.mark.parametrize(
    ("text", "max_chars", "expected", "tail"),
    [
        # An over-long run splits at the last space that fits.
        (
            "Alpha bravo charlie delta echo.",
            12,
            ["Alpha bravo", "charlie"],
            "delta echo.",
        ),
        # A newline beats an earlier space, and the last one wins over the first.
        ("ab\ncd\nefghijklmnop qr.", 8, ["ab\ncd", "efghijkl"], "mnop qr."),
        # A space exactly at the limit is usable.
        ("abcde fghi jklmn.", 10, ["abcde fghi"], "jklmn."),
        # A space one past the limit is not.
        ("abcde fghij klmn.", 10, ["abcde", "fghij"], "klmn."),
        # A space in the first half is worse than a hard split at the limit.
        ("abcde fghijklmnopqrst.", 12, ["abcde fghijk"], "lmnopqrst."),
        # A space exactly at half is still preferred to a hard split.
        ("abcdef ghijklmnop.", 12, ["abcdef"], "ghijklmnop."),
        # The half comparison floors, so an odd limit behaves like the even one.
        ("abcdef ghijklmnop.", 13, ["abcdef"], "ghijklmnop."),
        # A newline exactly at the limit is usable, like a space.
        ("abcdefghij\nklmno pqr.", 10, ["abcdefghij"], "klmno pqr."),
        # A newline past the limit is not, so the earlier space wins.
        ("abcde fghij\nklmn.", 10, ["abcde", "fghij"], "klmn."),
        # A newline outside the window is not considered at all, however far
        # away it is; only the window is searched.
        ("abcde fghij\nklmno pqr.", 10, ["abcde", "fghij"], "klmno pqr."),
        # A newline inside the window beats an earlier space, even when the
        # space alone would have been an acceptable break.
        ("abcdefg h\njklmno pq.", 10, ["abcdefg h"], "jklmno pq."),
    ],
)
def test_over_long_text_splits_on_the_last_break_that_fits(
    text, max_chars, expected, tail, as_sentence
) -> None:
    emitted, buffer = chunks(text, max_chars, as_sentence=as_sentence)

    if as_sentence:
        assert emitted == [*expected, tail]
        assert buffer == ""
    else:
        assert emitted == expected
        assert buffer == tail
    assert all(0 < len(chunk) <= max_chars for chunk in emitted)


@pytest.mark.parametrize("as_sentence", [True, False])
def test_text_at_exactly_the_limit_is_never_split(as_sentence) -> None:
    emitted, buffer = chunks("abcdefgh ijk", 12, as_sentence=as_sentence)

    assert (emitted, buffer) == (
        (["abcdefgh ijk"], "") if as_sentence else ([], "abcdefgh ijk")
    )


def test_the_default_chunk_size_is_four_hundred_characters() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("a" * 401)

    assert [len(chunk) for chunk in emitted] == [400]
    assert chunker.buffer == "a"


def test_text_arriving_in_pieces_is_joined_before_splitting() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append, max_chars=40)

    chunker.feed("Hello wo")
    chunker.feed("rld. Goodbye.")
    chunker.flush()

    assert emitted == ["Hello world.", "Goodbye."]


def test_flushing_empties_the_buffer_so_nothing_is_said_twice() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append, max_chars=40)

    chunker.feed("Only once")
    chunker.flush()
    chunker.flush()
    chunker.feed(" and then more")
    chunker.flush()

    assert emitted == ["Only once", "and then more"]
    assert chunker.buffer == ""


def test_echo_matching_is_case_and_punctuation_insensitive() -> None:
    assert EchoMatcher.normalize("Hello, WORLD!") == "hello world"
    assert EchoMatcher.matches(
        "please open the settings panel",
        "open the settings panel",
    )
    assert not EchoMatcher.matches("first unrelated", "second text")


# --------------------------------------------------------------------------
# Is this speaker still talking?
#
# The level tap hears sound, not speakers. These hold the line between the
# two: an open microphone picks up the assistant and the far end as readily
# as the person in front of it, and answering "yes" for either would hold a
# turn open for as long as they went on.
# --------------------------------------------------------------------------


class StubTap:
    """A level tap reporting whatever a test says is on the channel."""

    def __init__(self, hearing_sound: bool):
        self.hearing_sound = hearing_sound


def test_a_quiet_channel_means_its_speaker_is_not_talking() -> None:
    presence = SpeakerPresence(StubTap(False))

    assert presence.speaking() is False


def test_sound_on_an_unsuppressed_channel_is_its_speaker() -> None:
    """A monitored sink carries the meeting and nothing else."""
    presence = SpeakerPresence(StubTap(True))

    assert presence.speaking() is True


def test_sound_that_something_else_is_making_is_not_this_speaker() -> None:
    presence = SpeakerPresence(StubTap(True), [lambda: True])

    assert presence.speaking() is False


def test_sound_is_this_speaker_once_every_suppressor_has_let_go() -> None:
    presence = SpeakerPresence(StubTap(True), [lambda: False, lambda: False])

    assert presence.speaking() is True


def test_any_one_suppressor_is_enough_to_disown_the_sound() -> None:
    """Either the assistant or the far end playing is enough to explain it."""
    presence = SpeakerPresence(StubTap(True), [lambda: False, lambda: True])

    assert presence.speaking() is False


def test_a_quiet_channel_is_not_talking_whatever_the_suppressors_say() -> None:
    """Silence settles it, so nothing else needs asking."""
    asked = []

    def suppressor():
        asked.append(True)
        return False

    presence = SpeakerPresence(StubTap(False), [suppressor])

    assert presence.speaking() is False
    assert asked == []


def test_speaker_gate_answers_only_the_selected_speakers() -> None:
    gate = SpeakerGate({"Audio"}, {"Voice", "Audio"})

    assert gate.should_respond("Audio")
    assert not gate.should_respond("Voice")

    gate.set_policy("both")

    assert gate.should_respond("Audio")
    assert gate.should_respond("Voice")


def test_speaker_gate_never_answers_a_speaker_this_session_lacks() -> None:
    gate = SpeakerGate({"Voice", "Audio"}, {"Voice"})

    assert gate.active == frozenset({"Voice"})
    assert not gate.should_respond("Audio")

    gate.set_policy("audio")

    assert gate.active == frozenset()
    assert not gate.should_respond("Audio")
    assert not gate.should_respond("Voice")


def test_speaker_gate_quiet_policy_answers_nobody() -> None:
    gate = SpeakerGate({"Voice", "Audio"}, {"Voice", "Audio"})

    gate.set_policy("quiet")

    assert not gate.should_respond("Voice")
    assert not gate.should_respond("Audio")


COMMON_SEVEN = "alpha bravo charlie delta echo foxtrot golf"


@pytest.mark.parametrize(
    ("transcript", "spoken", "expected", "because"),
    [
        # A containment match needs enough text to be more than coincidence.
        ("abcdef", "abcdef gh", True, "six characters is long enough"),
        ("abcde", "abcde gh", False, "five characters is not"),
        ("a b", "a b c", False, "a short substring beats a high word ratio"),
        ("settings", "settings panel here", True, "one long word can still match"),
        ("", "hello there", False, "empty text matches nothing"),
        ("hello there", "", False, "and neither does empty speech"),
        # Word similarity carries the match when neither contains the other.
        (
            "please open the settings panel",
            "open the settings panel now",
            True,
            "a 0.80 word ratio is similar enough",
        ),
        (
            "alpha bravo charlie delta echo foxtrot golf hotel india one two three",
            "alpha bravo charlie delta echo foxtrot golf hotel india four five six seven",
            True,
            "a 0.72 word ratio is exactly the threshold",
        ),
        # Otherwise a long enough shared run decides it.
        (
            "open the settings panel right now",
            "open the settings today",
            True,
            "three shared words out of four is enough",
        ),
        (
            COMMON_SEVEN + " one two three",
            COMMON_SEVEN + " four five six seven eight nine ten eleven twelve thirteen",
            True,
            "seven shared words out of ten is exactly the threshold",
        ),
        (
            "open the settings panel right now here",
            "open the settings today please",
            False,
            "three shared words out of five is not enough",
        ),
        (
            "open the panel",
            "close the window",
            False,
            "one shared word is never enough",
        ),
        ("first unrelated", "second text", False, "nothing in common"),
        # Exactly at the ratio threshold, where the shared-run rule alone
        # would reject it: the ratio has to be what admits it.
        (
            "alpha bravo xx1 charlie delta xx2 echo foxtrot xx3 golf hotel india",
            "alpha bravo yy1 charlie delta yy2 echo foxtrot yy3 golf hotel india yy4",
            True,
            "a 0.72 word ratio admits what a 3-of-12 shared run would not",
        ),
    ],
)
def test_speech_matching_thresholds(transcript, spoken, expected, because) -> None:
    assert EchoMatcher.matches(transcript, spoken) is expected, because


def test_a_new_turn_supersedes_the_one_before_it() -> None:
    gate = TurnGate()
    gate.begin_turn()
    first = gate.current_turn

    assert gate.is_active(first)

    gate.begin_turn()

    assert not gate.is_active(first)
    assert gate.is_active(gate.current_turn)


def test_turns_are_numbered_upward_from_zero() -> None:
    gate = TurnGate()

    assert (gate.current_turn, gate.cancelled) == (0, False)

    gate.begin_turn()

    assert (gate.current_turn, gate.cancelled) == (1, False)

    gate.cancel()
    gate.begin_turn()

    assert (gate.current_turn, gate.cancelled) == (2, False)


def test_a_deadline_already_in_the_past_does_not_survive() -> None:
    clock = FakeClock()
    clock.now = 0.0
    memory = EchoMemory(clock=clock)

    memory.remember("a very short deadline", retention=0.5)
    clock.now = 0.6

    assert not memory.matches("a very short deadline")


def test_cancelling_a_turn_stops_speech_already_in_flight() -> None:
    gate = TurnGate()
    gate.begin_turn()
    turn, accepting = gate.accepting_turn()

    assert accepting

    gate.cancel()

    assert not gate.is_active(turn)
    assert gate.accepting_turn() == (turn, False)


def test_a_cancelled_turn_recovers_when_the_next_one_begins() -> None:
    gate = TurnGate()
    gate.begin_turn()
    gate.cancel()
    gate.begin_turn()

    assert gate.accepting_turn()[1]
    assert gate.is_active(gate.current_turn)


def test_disabling_speech_asks_for_an_interrupt_and_enabling_does_not() -> None:
    gate = TurnGate()

    assert gate.set_enabled(False) is True
    assert gate.accepting_turn()[1] is False
    assert gate.set_enabled(True) is False
    assert gate.accepting_turn()[1] is True


def test_recently_spoken_text_is_recognised_until_it_expires() -> None:
    clock = FakeClock()
    memory = EchoMemory(clock=clock)

    memory.remember("Opening the settings panel now", retention=30)

    assert memory.matches("opening the settings panel now")

    clock.advance(30)

    assert not memory.matches("opening the settings panel now")


def test_speech_is_matched_through_transcription_noise() -> None:
    memory = EchoMemory(clock=FakeClock())

    memory.remember("Please open the settings panel", retention=30)

    assert memory.matches("please open the settings panel!")
    assert not memory.matches("what time is the meeting")


def test_re_queuing_a_sentence_never_shortens_its_deadline() -> None:
    clock = FakeClock()
    memory = EchoMemory(clock=clock)

    memory.remember("the same sentence twice", retention=100)
    memory.remember("the same sentence twice", retention=5)
    clock.advance(50)

    assert memory.matches("the same sentence twice")


def test_replacing_a_deadline_shortens_it() -> None:
    clock = FakeClock()
    memory = EchoMemory(clock=clock)

    memory.remember("the same sentence twice", retention=100)
    memory.remember("the same sentence twice", retention=5, replace=True)
    clock.advance(50)

    assert not memory.matches("the same sentence twice")


def test_text_with_nothing_to_match_is_never_an_echo() -> None:
    memory = EchoMemory(clock=FakeClock())

    memory.remember("...", retention=30)
    memory.remember("", retention=30)

    assert not memory.matches("!!!")
    assert not memory.matches("anything at all")


def test_transcript_router_keeps_context_until_a_reply_is_requested() -> None:
    router = TranscriptRouter()

    assert (
        router.ingest("Audio", "Could you help?", "2026-07-26T12:00:00-04:00", False)
        is None
    )
    request = router.ingest("Voice", "Yes", "2026-07-26T12:00:03-04:00", True)

    assert request is not None
    assert request.reply_to == "Voice"
    assert [
        (entry.speaker, entry.text, entry.timestamp) for entry in request.entries
    ] == [
        ("Audio", "Could you help?", "2026-07-26T12:00:00-04:00"),
        ("Voice", "Yes", "2026-07-26T12:00:03-04:00"),
    ]
    assert router.pending_context == []


def silence_clock(window=3.0):
    clock = FakeClock()
    return TurnSilenceClock(TurnSilence(window), clock=clock), clock


def test_nothing_is_counting_down_before_a_turn_starts() -> None:
    countdown, _ = silence_clock()

    assert countdown.remaining() is None


def test_a_started_turn_counts_down_the_full_window() -> None:
    countdown, _ = silence_clock(3.0)

    countdown.started("Voice")

    assert countdown.remaining() == 3.0


def test_a_turn_held_open_counts_down_the_grace_it_was_given() -> None:
    """An extension runs on a grace, and the countdown has to show that one."""
    countdown, _ = silence_clock(3.0)

    countdown.started("Voice", 0.5)

    assert countdown.remaining() == 0.5


def test_the_countdown_shrinks_as_the_silence_runs() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Voice")

    clock.advance(1.2)

    assert countdown.remaining() == pytest.approx(1.8)


def test_a_cleared_turn_stops_counting_down() -> None:
    countdown, _ = silence_clock()
    countdown.started("Voice")

    countdown.cleared("Voice")

    assert countdown.remaining() is None


def test_clearing_a_speaker_that_never_started_is_harmless() -> None:
    countdown, _ = silence_clock()

    countdown.cleared("Audio")

    assert countdown.remaining() is None


def test_restarting_a_turn_resets_its_window() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Voice")
    clock.advance(2.5)

    countdown.started("Voice")

    assert countdown.remaining() == 3.0


def test_the_soonest_speaker_is_the_one_being_waited_on() -> None:
    """A later timer is not what the session is about to act on."""
    countdown, clock = silence_clock(3.0)
    countdown.started("Audio")
    clock.advance(2.0)
    countdown.started("Voice")

    assert countdown.remaining() == pytest.approx(1.0)


def test_clearing_the_soonest_speaker_falls_back_to_the_other() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Audio")
    clock.advance(2.0)
    countdown.started("Voice")

    countdown.cleared("Audio")

    assert countdown.remaining() == pytest.approx(3.0)


def test_a_turn_that_already_fired_stops_being_shown() -> None:
    """A countdown wedged at zero would be worse than one that disappears."""
    countdown, clock = silence_clock(3.0)
    countdown.started("Voice")

    clock.advance(3.0)

    assert countdown.remaining() is None


def test_an_overdue_turn_is_dropped_rather_than_going_negative() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Voice")

    clock.advance(30.0)

    assert countdown.remaining() is None
    assert countdown.remaining() is None


def test_a_speaker_still_waiting_survives_another_one_expiring() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Audio")
    clock.advance(2.0)
    countdown.started("Voice")

    clock.advance(1.5)

    assert countdown.remaining() == pytest.approx(1.5)


def test_a_window_starts_at_the_value_it_was_given() -> None:
    assert TurnSilence(2.5).seconds == 2.5


def test_a_window_can_be_changed_while_the_session_runs() -> None:
    silence = TurnSilence(3.0)

    assert silence.set(1.5) == 1.5
    assert silence.seconds == 1.5


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0.0, TurnSilence.MINIMUM),
        (-5.0, TurnSilence.MINIMUM),
        (1000.0, TurnSilence.MAXIMUM),
        (TurnSilence.MINIMUM, TurnSilence.MINIMUM),
        (TurnSilence.MAXIMUM, TurnSilence.MAXIMUM),
    ],
)
def test_a_window_is_held_inside_its_bounds(requested, expected) -> None:
    assert TurnSilence(requested).seconds == expected
    assert TurnSilence(3.0).set(requested) == expected


def test_the_countdown_adopts_a_window_changed_between_turns() -> None:
    """A new window applies to the next turn, not the one already waiting."""
    silence = TurnSilence(3.0)
    clock = FakeClock()
    countdown = TurnSilenceClock(silence, clock=clock)
    countdown.started("Voice")

    silence.set(1.0)

    assert countdown.remaining() == 3.0
    countdown.started("Voice")
    assert countdown.remaining() == 1.0


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("2.5", 2.5),
        ("  2.5  ", 2.5),
        ("2.5s", 2.5),
        ("2.5 s", 2.5),
        ("3", 3.0),
        (str(TurnSilence.MINIMUM), TurnSilence.MINIMUM),
        (str(TurnSilence.MAXIMUM), TurnSilence.MAXIMUM),
    ],
)
def test_a_typed_window_is_read_back(typed, expected) -> None:
    assert parse_turn_silence(typed) == expected


@pytest.mark.parametrize(
    "typed",
    [
        "",
        "   ",
        "abc",
        "s",
        "2.5.1",
        "--2",
        "1e",
        "nan",
        "inf",
        "-inf",
        "0",
        "-1",
        "0.1",
        "31",
        "1000",
    ],
)
def test_a_value_the_field_cannot_use_is_refused(typed) -> None:
    """Refused rather than clamped: a silent correction looks like a dropped key."""
    assert parse_turn_silence(typed) is None


@pytest.mark.parametrize(
    ("text", "configured", "expected"),
    [
        ("Taga, what's the status?", 1.25, CLEAR_TURN_SILENCE),
        ("Stop the build!", 3.0, CLEAR_TURN_SILENCE),
        ("Taga open the file", 1.25, CLEAR_TURN_SILENCE),
        ("I was going to say and", 1.25, 1.25),
        ("the report covers several open items from last week", 1.25, 1.25),
        ("", 1.25, 1.25),
        ("Ready?", 0.3, 0.3),  # already below CLEAR; stay at configured
    ],
)
def test_clear_turn_endings_shorten_silence(text, configured, expected) -> None:
    assert silence_for_turn(text, configured) == expected


def test_same_turn_text_ignores_case_and_spacing() -> None:
    assert same_turn_text("Taga  status?", "taga status?")
    assert not same_turn_text("Taga one", "Taga two")


def test_an_engine_with_nothing_queued_is_not_speaking() -> None:
    assert SpeechActivity().speaking is False


def test_a_queued_sentence_counts_as_speaking() -> None:
    activity = SpeechActivity()

    activity.queued()

    assert activity.speaking is True


def test_speech_continues_across_the_gap_between_sentences() -> None:
    """The gap between two sentences must not read as silence."""
    activity = SpeechActivity()
    activity.queued()
    activity.queued()

    activity.finished()

    assert activity.speaking is True


def test_an_observer_is_told_about_every_state_you_make() -> None:
    activity = SpeechActivity()
    transitions = []

    def observed(state):
        transitions.append(state.speaking)

    activity.observe(observed)
    activity.queued()
    activity.finished()
    activity.queued()
    activity.silenced()

    assert transitions == [True, False, True, False]


def test_an_observer_can_read_the_state_it_was_triggered_by() -> None:
    """Re-reading ``speaking`` inside the observer must not deadlock.

    The count is the observable the media controls mirror, so the observer
    reads it back while the transition that fired it is still unwinding. The
    mutation runs on a helper thread so a regression hangs that thread rather
    than the test; the bounded wait turns the hang into a failure within two
    seconds instead of wedging an xdist worker.
    """
    activity = SpeechActivity()
    done = threading.Event()
    broken = []

    def observer(state):
        if not state.speaking:
            broken.append("observer did not see the transition")
        done.set()

    activity.observe(observer)
    mutator = threading.Thread(target=activity.queued, daemon=True)
    mutator.start()

    assert done.wait(timeout=2) is True
    assert broken == []


def test_speech_ends_once_every_sentence_has_been_delivered() -> None:
    activity = SpeechActivity()
    activity.queued()
    activity.queued()

    activity.finished()
    activity.finished()

    assert activity.speaking is False


def test_an_extra_completion_cannot_mute_the_next_sentence() -> None:
    activity = SpeechActivity()
    activity.queued()
    activity.finished()
    activity.finished()

    activity.queued()

    assert activity.speaking is True


def test_being_silenced_drops_everything_outstanding() -> None:
    activity = SpeechActivity()
    activity.queued()
    activity.queued()

    activity.silenced()

    assert activity.speaking is False


def test_speech_can_start_again_after_being_silenced() -> None:
    activity = SpeechActivity()
    activity.queued()
    activity.silenced()

    activity.queued()

    assert activity.speaking is True


def test_unanswered_context_stops_growing_at_its_bound() -> None:
    """Nothing replies under the "stay silent" policy, so nothing clears it."""
    router = TranscriptRouter(max_pending_context=3)

    for index in range(20):
        assert (
            router.ingest("Audio", f"line {index}", "2026-07-26T12:00:00-04:00", False)
            is None
        )

    assert len(router.pending_context) == 3


def test_the_bound_keeps_the_most_recent_turns() -> None:
    """A reply is about what was just said, not about the start of the day."""
    router = TranscriptRouter(max_pending_context=2)

    for index in range(5):
        router.ingest("Audio", f"line {index}", "2026-07-26T12:00:00-04:00", False)
    request = router.ingest("Voice", "Yes", "2026-07-26T12:00:03-04:00", True)

    assert request is not None
    assert [entry.text for entry in request.entries] == ["line 3", "line 4", "Yes"]


def test_context_within_the_bound_is_carried_whole() -> None:
    router = TranscriptRouter(max_pending_context=10)

    router.ingest("Audio", "first", "2026-07-26T12:00:00-04:00", False)
    router.ingest("Audio", "second", "2026-07-26T12:00:01-04:00", False)
    request = router.ingest("Voice", "third", "2026-07-26T12:00:02-04:00", True)

    assert request is not None
    assert [entry.text for entry in request.entries] == ["first", "second", "third"]


# --------------------------------------------------------------------------
# Markdown cleaned for speech
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("", ""),
        ("Plain prose with no markup.", "Plain prose with no markup."),
        (
            "Hello **bold** and *italic* with `code`.",
            "Hello bold and italic with code.",
        ),
        (
            "See the [docs](https://example.com/path) for detail.",
            "See the docs for detail.",
        ),
        (
            "Here is a diagram: ![network](net.png).",
            "Here is a diagram: network.",
        ),
        ("# Title\n\nA short answer.", "Title A short answer."),
        (
            "Steps:\n\n- first\n- second\n\n1. third",
            "Steps: first second third",
        ),
        ("> quoted advice\n\nand more.", "quoted advice and more."),
        (
            "Before.\n```python\nprint(1)\n```\nAfter.",
            "Before. After.",
        ),
        (
            "Use `path/to/file` carefully.",
            "Use path/to/file carefully.",
        ),
        (
            "keep snake_case and file_name intact.",
            "keep snake_case and file_name intact.",
        ),
        ("~~old~~ plan.", "old plan."),
        ("```\nonly code\n```", ""),
        # Unclosed fence: CommonMark treats the rest as a fence; drop it.
        ("Start\n```python\nprint(1)", "Start"),
        (
            "Visit https://example.com/now please.",
            "Visit please.",
        ),
        (
            "Also see <https://example.com/docs>.",
            "Also see.",
        ),
    ],
)
def test_markdown_to_speech_strips_markup_not_words(source, spoken) -> None:
    """TTS hears the words; asterisks, fences, and URLs must not be spoken."""
    assert markdown_to_speech(source) == spoken


def test_markdown_to_speech_does_not_leave_emphasis_markers() -> None:
    """Closed or orphan bold markers are both audible garbage if left behind."""
    cleaned = markdown_to_speech("Really **important** point.")
    assert cleaned == "Really important point."
    assert "*" not in cleaned
    assert "`" not in cleaned

    unclosed = markdown_to_speech("This is **not closed")
    assert unclosed == "This is not closed"
    assert "*" not in unclosed

    # Asterisk italic must drop both open and close markers, not only strong.
    italic = markdown_to_speech("A *soft* word.")
    assert italic == "A soft word."
    assert "*" not in italic


def test_markdown_to_speech_prefers_link_labels_over_urls() -> None:
    cleaned = markdown_to_speech("Open [settings](https://example.com/settings) now.")
    assert cleaned == "Open settings now."
    assert "http" not in cleaned
    assert "example.com" not in cleaned
    assert "settings" in cleaned


def test_markdown_to_speech_preserves_code_and_math_like_prose() -> None:
    """A coding assistant must not mangle identifiers the model wrote plainly.

    CommonMark leaves ``3 * 4``, ``*ptr``, and ``**kwargs`` as literal text
    (they are not emphasis). Orphan ``**`` stripping turns ``**kwargs`` into
    ``kwargs``, which is still speakable and better than "asterisk asterisk".
    """
    assert markdown_to_speech("Compute 3 * 4 please.") == "Compute 3 * 4 please."
    assert markdown_to_speech("Dereference *ptr carefully.") == (
        "Dereference *ptr carefully."
    )
    assert markdown_to_speech("Pass **kwargs through.") == "Pass kwargs through."


def test_markdown_to_speech_keeps_dunder_names() -> None:
    """``__init__`` is CommonMark strong emphasis; speech still needs the dunders.

    When the model remembers backticks, inline code already preserves them.
    When it forgets, underscore markup is re-emitted around the span.
    """
    assert markdown_to_speech("Call `__init__` once.") == "Call __init__ once."
    assert markdown_to_speech("Call __init__ once.") == "Call __init__ once."
    # Single-underscore emphasis is the em form; markers must still return.
    assert markdown_to_speech("See _name_ here.") == "See _name_ here."


def test_strip_chrome_removes_slack_noise_without_paraphrasing() -> None:
    """Selection chrome only — wording that must survive stays substring-intact."""
    assert strip_chrome("") == ""
    slack = (
        "Gabriel Silva 10:42 AM Hey team - the new deploy is live :tada:\n"
        "rollback is `make rollback`\n"
        ":eyes: 4 :tada: 2\n"
        "3 replies\n"
        'quoted: "we should watch p99 for an hour" (edited)\n'
        "Sounds right.\n"
        "I'll keep an eye on it and post at noon if anything drifts."
    )
    cleaned = strip_chrome(slack)
    assert cleaned == (
        "Hey team - the new deploy is live rollback is `make rollback` "
        'quoted: "we should watch p99 for an hour" Sounds right. '
        "I'll keep an eye on it and post at noon if anything drifts."
    )
    spoken = markdown_to_speech(cleaned)
    assert "the new deploy is live" in spoken
    assert "make rollback" in spoken


def test_strip_chrome_header_is_first_line_only() -> None:
    """Unanchored time matching would eat mid-paragraph 'at 10:42 AM'."""
    body = (
        "Gabriel Silva 10:42 AM Opening line.\n"
        "we shipped it at 10:42 AM and watched p99."
    )
    cleaned = strip_chrome(body)
    assert cleaned == "Opening line. we shipped it at 10:42 AM and watched p99."


def test_strip_chrome_same_line_reaction_counts_not_cross_line_replies() -> None:
    """`:tada: 2` on one line must not swallow the next line's prose replies."""
    text = "Live:tada: 2\n3 replies left in the thread."
    assert strip_chrome(text) == "Live 3 replies left in the thread."


def test_strip_chrome_preserves_ordinary_prose() -> None:
    """Anchored chrome rules must leave clocks, reply phrases, and Edited intact."""
    prose = (
        "The build finished at 10:30:45 and we shipped. "
        "Aspect ratio is 16:9:1 in the export. "
        "Use the key path a:b:c in the config. "
        "She got 3 replies within a minute. "
        "The contract was Edited by counsel before signing."
    )
    assert strip_chrome(prose) == prose


def test_markdown_to_speech_url_punctuation_stays_with_the_sentence() -> None:
    """URL bodies drop; a trailing period is sentence structure, not the URL."""
    assert markdown_to_speech("See https://example.com/path.") == "See."
    assert markdown_to_speech("See https://example.com/path please.") == "See please."
    assert "http" not in markdown_to_speech("See https://example.com/path.")
    assert "example" not in markdown_to_speech("See https://example.com/path.")


def test_markdown_to_speech_skips_empty_inlines_without_stopping() -> None:
    """An empty heading inline must not ``break`` the walk before later prose.

    ``continue`` vs ``break`` on empty children is a real behavior fork: break
    would drop everything after the first vacant inline.
    """
    assert markdown_to_speech("# \n\nSpoken body.") == "Spoken body."
    assert markdown_to_speech("# \n\nFirst.\n\nSecond.") == "First. Second."


def test_markdown_to_speech_reads_image_alt_from_children() -> None:
    """Image alts live in nested inline children; that path must actually run."""
    assert (
        markdown_to_speech("Diagram: ![left hub](a.png) and done.")
        == "Diagram: left hub and done."
    )


def test_speech_sink_drops_empty_chunks_and_forwards_prose() -> None:
    spoken: list[str] = []
    emit = speech_sink(spoken.append)

    emit("```\nprint(1)\n```")
    emit("Hello **world**.")

    assert spoken == ["Hello world."]


def test_markdown_to_speech_joins_breaks_and_skips_rules() -> None:
    """Soft and hard breaks become spaces; rules and empty alts contribute nothing."""
    assert markdown_to_speech("line one\\\nline two") == "line one line two"
    # Soft break: a single newline inside a paragraph.
    assert markdown_to_speech("soft\nbreak") == "soft break"
    assert markdown_to_speech("---\n\nAfter the rule.") == "After the rule."
    assert markdown_to_speech("![](ignored.png)") == ""
    assert markdown_to_speech("a<br>b") == "ab"
    assert "<" not in markdown_to_speech("a<br>b")


def test_markdown_to_speech_unknown_inline_shapes_stay_silent_or_recurse() -> None:
    """Defensive branches: the walker must not crash on unexpected tokens."""
    from types import SimpleNamespace

    from tagalong.domain import _render_speech_inlines

    nested = SimpleNamespace(
        type="custom",
        content="",
        markup="",
        children=[
            SimpleNamespace(type="text", content="nested", markup="", children=None)
        ],
    )
    bare = SimpleNamespace(type="custom", content="x", markup="", children=None)
    image_fallback = SimpleNamespace(
        type="image", content="fallback-alt", markup="", children=None
    )

    assert _render_speech_inlines([nested]) == "nested"
    assert _render_speech_inlines([bare]) == ""
    assert _render_speech_inlines([image_fallback]) == "fallback-alt"


# --------------------------------------------------------------------------
# Speaking before the sentence ends
# --------------------------------------------------------------------------


def test_the_opening_clause_is_spoken_without_waiting_for_the_sentence() -> None:
    """The first chunk is the one a listener is waiting through silence for."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("That depends on whether you want it fast, but I can check.")

    assert emitted == ["That depends on whether you want it fast,"]


def test_only_the_opening_chunk_breaks_at_a_clause() -> None:
    """Once speech is playing a fragment costs prosody and buys nothing."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("Yes it is working correctly. ")
    chunker.feed("The second one is much longer, and it keeps going.")

    assert emitted == ["Yes it is working correctly."]
    assert chunker.buffer == "The second one is much longer, and it keeps going."


def test_a_clause_shorter_than_the_minimum_is_not_spoken_alone() -> None:
    """ "Sure," on its own is a worse start than the wait it saves."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("Sure, ")

    assert emitted == []


def test_an_early_comma_is_passed_over_rather_than_ending_the_chunk() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("Yes, that is the one you asked about, and here is why.")

    assert emitted == ["Yes, that is the one you asked about,"]


def test_clause_breaking_can_be_turned_off_entirely() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(
        emitted.append,
        first_clause_min_chars=None,
        first_chunk_max_words=None,
    )

    chunker.feed("That depends on whether you want it fast, but I can check.")

    assert emitted == []


def test_a_sentence_break_beats_a_clause_break_in_the_same_feed() -> None:
    """Both rules can match at once; the sentence is the better chunk."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("It works, and it is fast. And there is more to say here.")

    assert emitted == ["It works, and it is fast."]


def test_the_opening_words_are_spoken_without_waiting_for_punctuation() -> None:
    """A long first sentence with no comma must not hold speech for its period."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed(
        "The deployment finished without errors and the service looks healthy right now too"
    )

    assert emitted == [
        "The deployment finished without errors and the service looks healthy right now"
    ]
    assert chunker.buffer == "too"


def test_opening_words_accumulate_across_streamed_feeds() -> None:
    """Real Codex deltas arrive in pieces; the cap must count across them."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("The deployment finished without ")
    assert emitted == []
    chunker.feed("errors and the service looks healthy right ")
    assert emitted == []
    chunker.feed("now too")

    assert emitted == [
        "The deployment finished without errors and the service looks healthy right now"
    ]
    assert chunker.buffer == "too"


def test_leading_whitespace_does_not_block_the_opening_word_break() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed(
        "  The deployment finished without errors and the service looks healthy right now too"
    )

    assert emitted == [
        "The deployment finished without errors and the service looks healthy right now"
    ]
    assert chunker.buffer == "too"


def test_a_short_opening_stays_buffered_until_it_ends() -> None:
    """Twelve words or fewer may still be the whole answer; wait for the end."""
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("Yes the web service is healthy and ready for traffic right now")

    assert emitted == []
    assert chunker.buffer == (
        "Yes the web service is healthy and ready for traffic right now"
    )


def test_word_breaking_only_applies_to_the_opening_chunk() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed("Yes it is working correctly. ")
    chunker.feed(
        "The deployment finished without errors and the service looks healthy right now too"
    )

    assert emitted == ["Yes it is working correctly."]
    assert chunker.buffer == (
        "The deployment finished without errors and the service looks healthy right now too"
    )


def test_a_clause_break_beats_a_word_break_in_the_same_feed() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append)

    chunker.feed(
        "That depends on whether you want it fast, but I can also check the logs."
    )

    assert emitted == ["That depends on whether you want it fast,"]


def test_word_breaking_can_be_turned_off_entirely() -> None:
    emitted: list[str] = []
    chunker = SentenceChunker(emitted.append, first_chunk_max_words=None)

    chunker.feed(
        "The deployment finished without errors and the service looks healthy right now too"
    )

    assert emitted == []


# --------------------------------------------------------------------------
# Learning how long Codex takes to speak
# --------------------------------------------------------------------------


def test_the_estimator_starts_at_its_seed() -> None:
    assert TurnLatencyEstimator(seed=0.9).estimate == pytest.approx(0.9)


def test_one_sample_moves_the_estimate_part_of_the_way() -> None:
    """A moving average, not a replacement: one slow turn must not relocate it."""
    estimator = TurnLatencyEstimator(seed=1.0, smoothing=0.25)

    estimator.record(0.0)

    assert estimator.estimate == pytest.approx(0.75)


def test_repeated_samples_converge_on_what_codex_is_actually_doing() -> None:
    estimator = TurnLatencyEstimator(seed=2.0, smoothing=0.5)

    for _ in range(20):
        estimator.record(0.6)

    assert estimator.estimate == pytest.approx(0.6, abs=1e-3)


def test_a_slower_sample_raises_the_estimate() -> None:
    estimator = TurnLatencyEstimator(seed=0.5, smoothing=0.5)

    estimator.record(1.5)

    assert estimator.estimate == pytest.approx(1.0)


# --------------------------------------------------------------------------
# When to guess that a turn is over
# --------------------------------------------------------------------------


def test_the_guess_is_timed_so_the_reply_lands_as_the_window_closes() -> None:
    plan = PrefirePlan(TurnLatencyEstimator(seed=0.6))

    assert plan.delay(1.0) == pytest.approx(0.4)


def test_a_slow_estimate_cannot_fire_the_moment_a_speaker_pauses() -> None:
    """Capped by fraction, so the guess still waits out most of a long pause."""
    plan = PrefirePlan(TurnLatencyEstimator(seed=10.0), max_lead_fraction=0.65)

    assert plan.delay(1.0) == pytest.approx(0.35)


def test_a_short_window_still_leaves_a_moment_to_interrupt() -> None:
    plan = PrefirePlan(TurnLatencyEstimator(seed=10.0), minimum_delay=0.15)

    assert plan.delay(0.25) == pytest.approx(0.15)


def test_a_long_window_waits_out_all_of_it_but_the_estimate() -> None:
    plan = PrefirePlan(TurnLatencyEstimator(seed=0.7))

    assert plan.delay(30.0) == pytest.approx(29.3)


def test_the_delay_tracks_the_estimator_as_it_learns() -> None:
    estimator = TurnLatencyEstimator(seed=1.0, smoothing=1.0)
    plan = PrefirePlan(estimator)

    before = plan.delay(2.0)
    estimator.record(0.4)

    assert before == pytest.approx(1.0)
    assert plan.delay(2.0) == pytest.approx(1.6)


# --------------------------------------------------------------------------
# Speculating on a turn that is not yet over
# --------------------------------------------------------------------------


def test_speculating_leaves_the_context_it_carried_pending() -> None:
    """Abandoning a guess must cost nothing, so nothing is consumed to make it."""
    router = TranscriptRouter()
    router.ingest("Audio", "context", "T1", False)

    request = router.speculate("Voice", "a question", "T2")

    assert [entry.text for entry in request.entries] == ["context", "a question"]
    assert [entry.text for entry in router.pending_context] == [
        "context",
        "a question",
    ]


def test_an_abandoned_speculation_is_carried_by_the_next_request() -> None:
    router = TranscriptRouter()
    router.speculate("Voice", "half a sentence", "T1")

    request = router.ingest("Voice", "the whole sentence", "T2", True)

    assert request is not None
    assert [entry.text for entry in request.entries] == [
        "half a sentence",
        "the whole sentence",
    ]


def test_committing_consumes_only_what_the_request_carried() -> None:
    """Context arriving mid-flight belongs to the next reply, not this one."""
    router = TranscriptRouter()
    request = router.speculate("Voice", "a question", "T1")
    router.ingest("Audio", "arrived while answering", "T2", False)

    router.commit(request)

    assert [entry.text for entry in router.pending_context] == [
        "arrived while answering"
    ]


def test_committing_survives_the_context_bound_trimming_the_front() -> None:
    """The boundary is found by identity, so a moved prefix cannot mislead it."""
    router = TranscriptRouter(max_pending_context=2)
    request = router.speculate("Voice", "a question", "T1")
    router.ingest("Audio", "later", "T2", False)
    router.ingest("Audio", "latest", "T3", False)

    router.commit(request)

    assert [entry.text for entry in router.pending_context] == ["later", "latest"]


def test_committing_twice_does_not_consume_a_later_turn() -> None:
    router = TranscriptRouter()
    request = router.speculate("Voice", "a question", "T1")
    router.commit(request)
    router.ingest("Audio", "afterwards", "T2", False)

    router.commit(request)

    assert [entry.text for entry in router.pending_context] == ["afterwards"]


def test_identical_text_does_not_confuse_the_commit_boundary() -> None:
    """Entries compare equal by value; only the one actually sent may be dropped."""
    router = TranscriptRouter()
    request = router.speculate("Audio", "yes", "T1")
    router.ingest("Audio", "yes", "T1", False)

    router.commit(request)

    assert len(router.pending_context) == 1


def test_committing_a_request_that_carried_nothing_consumes_nothing() -> None:
    """A reply built from no entries has no prefix to drop."""
    router = TranscriptRouter()
    router.ingest("Audio", "context", "T1", False)

    router.commit(CodexRequest("Audio", ()))

    assert [entry.text for entry in router.pending_context] == ["context"]
