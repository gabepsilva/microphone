from __future__ import annotations

import pytest

from voice_codex.domain import (
    EchoMatcher,
    EchoMemory,
    SentenceChunker,
    SpeakerGate,
    TranscriptRouter,
    TurnGate,
    TurnSilenceClock,
    resolve_response_policy,
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
        ("them", "Them", frozenset({"Them"})),
        ("both", "User Voice and Them", frozenset({"User Voice", "Them"})),
        ("user", "User Voice", frozenset({"User Voice"})),
        ("quiet", "Codex will be quiet for voice", frozenset()),
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


def test_speaker_gate_answers_only_the_selected_speakers() -> None:
    gate = SpeakerGate({"Them"}, {"User Voice", "Them"})

    assert gate.should_respond("Them")
    assert not gate.should_respond("User Voice")

    gate.set_policy("both")

    assert gate.should_respond("Them")
    assert gate.should_respond("User Voice")


def test_speaker_gate_never_answers_a_speaker_this_session_lacks() -> None:
    gate = SpeakerGate({"User Voice", "Them"}, {"User Voice"})

    assert gate.active == frozenset({"User Voice"})
    assert not gate.should_respond("Them")

    gate.set_policy("them")

    assert gate.active == frozenset()
    assert not gate.should_respond("Them")
    assert not gate.should_respond("User Voice")


def test_speaker_gate_quiet_policy_answers_nobody() -> None:
    gate = SpeakerGate({"User Voice", "Them"}, {"User Voice", "Them"})

    gate.set_policy("quiet")

    assert not gate.should_respond("User Voice")
    assert not gate.should_respond("Them")


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
        router.ingest("Them", "Could you help?", "2026-07-26T12:00:00-04:00", False)
        is None
    )
    request = router.ingest("User Voice", "Yes", "2026-07-26T12:00:03-04:00", True)

    assert request is not None
    assert request.reply_to == "User Voice"
    assert [
        (entry.speaker, entry.text, entry.timestamp) for entry in request.entries
    ] == [
        ("Them", "Could you help?", "2026-07-26T12:00:00-04:00"),
        ("User Voice", "Yes", "2026-07-26T12:00:03-04:00"),
    ]
    assert router.pending_context == []


def silence_clock(window=3.0):
    clock = FakeClock()
    return TurnSilenceClock(window, clock=clock), clock


def test_nothing_is_counting_down_before_a_turn_starts() -> None:
    countdown, _ = silence_clock()

    assert countdown.remaining() is None


def test_a_started_turn_counts_down_the_full_window() -> None:
    countdown, _ = silence_clock(3.0)

    countdown.started("User Voice")

    assert countdown.remaining() == 3.0


def test_the_countdown_shrinks_as_the_silence_runs() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("User Voice")

    clock.advance(1.2)

    assert countdown.remaining() == pytest.approx(1.8)


def test_a_cleared_turn_stops_counting_down() -> None:
    countdown, _ = silence_clock()
    countdown.started("User Voice")

    countdown.cleared("User Voice")

    assert countdown.remaining() is None


def test_clearing_a_speaker_that_never_started_is_harmless() -> None:
    countdown, _ = silence_clock()

    countdown.cleared("Them")

    assert countdown.remaining() is None


def test_restarting_a_turn_resets_its_window() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("User Voice")
    clock.advance(2.5)

    countdown.started("User Voice")

    assert countdown.remaining() == 3.0


def test_the_soonest_speaker_is_the_one_being_waited_on() -> None:
    """A later timer is not what the session is about to act on."""
    countdown, clock = silence_clock(3.0)
    countdown.started("Them")
    clock.advance(2.0)
    countdown.started("User Voice")

    assert countdown.remaining() == pytest.approx(1.0)


def test_clearing_the_soonest_speaker_falls_back_to_the_other() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Them")
    clock.advance(2.0)
    countdown.started("User Voice")

    countdown.cleared("Them")

    assert countdown.remaining() == pytest.approx(3.0)


def test_a_turn_that_already_fired_stops_being_shown() -> None:
    """A countdown wedged at zero would be worse than one that disappears."""
    countdown, clock = silence_clock(3.0)
    countdown.started("User Voice")

    clock.advance(3.0)

    assert countdown.remaining() is None


def test_an_overdue_turn_is_dropped_rather_than_going_negative() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("User Voice")

    clock.advance(30.0)

    assert countdown.remaining() is None
    assert countdown.remaining() is None


def test_a_speaker_still_waiting_survives_another_one_expiring() -> None:
    countdown, clock = silence_clock(3.0)
    countdown.started("Them")
    clock.advance(2.0)
    countdown.started("User Voice")

    clock.advance(1.5)

    assert countdown.remaining() == pytest.approx(1.5)
