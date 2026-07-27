from __future__ import annotations

import pytest

from voice_codex.domain import (
    EchoMatcher,
    EchoMemory,
    SentenceChunker,
    SpeakerGate,
    TranscriptRouter,
    TurnGate,
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


def test_a_new_turn_supersedes_the_one_before_it() -> None:
    gate = TurnGate()
    gate.begin_turn()
    first = gate.current_turn

    assert gate.is_active(first)

    gate.begin_turn()

    assert not gate.is_active(first)
    assert gate.is_active(gate.current_turn)


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
