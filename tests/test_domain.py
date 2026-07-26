from __future__ import annotations

import pytest

from voice_codex.domain import (
    EchoMatcher,
    SentenceChunker,
    TranscriptRouter,
    resolve_response_policy,
)


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
