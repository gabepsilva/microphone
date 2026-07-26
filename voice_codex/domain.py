"""Pure transcript, response-policy, and speech-matching logic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

USER_VOICE = "User Voice"
USER_TEXT = "User Text"
THEM = "Them"
CODEX = "Codex"


@dataclass(frozen=True)
class ResponsePolicy:
    """The completed voice turns that are allowed to trigger a reply."""

    name: str
    label: str
    speakers: frozenset[str]


RESPONSE_POLICIES = {
    "them": ResponsePolicy("them", "Them", frozenset({THEM})),
    "both": ResponsePolicy(
        "both", "User Voice and Them", frozenset({USER_VOICE, THEM})
    ),
    "user": ResponsePolicy("user", "User Voice", frozenset({USER_VOICE})),
    "quiet": ResponsePolicy("quiet", "Codex will be quiet for voice", frozenset()),
}
_POLICY_ALIASES = {
    "1": "them",
    "2": "both",
    "3": "user",
    "4": "quiet",
    **{name: name for name in RESPONSE_POLICIES},
}


def resolve_response_policy(value: str) -> ResponsePolicy:
    """Resolve a CLI answer or persisted policy name into a policy."""
    return RESPONSE_POLICIES[_POLICY_ALIASES[value]]


class SentenceChunker:
    """Turn streamed text into sentence-sized chunks with a hard size cap."""

    SENTENCE_END = re.compile(
        r'(?<=[.!?])(?:["”\N{RIGHT SINGLE QUOTATION MARK}\')\]]*)\s+'
    )

    def __init__(self, emit, max_chars: int = 400) -> None:
        self.emit = emit
        self.max_chars = max_chars
        self.buffer = ""

    def _emit_bounded(self, text: str) -> None:
        while len(text) > self.max_chars:
            split_at = max(
                text.rfind("\n", 0, self.max_chars + 1),
                text.rfind(" ", 0, self.max_chars + 1),
            )
            if split_at < self.max_chars // 2:
                split_at = self.max_chars
            chunk = text[:split_at].strip()
            text = text[split_at:].lstrip()
            if chunk:
                self.emit(chunk)
        if text:
            self.emit(text)

    def _emit_long_chunks(self) -> None:
        while len(self.buffer) > self.max_chars:
            split_at = max(
                self.buffer.rfind("\n", 0, self.max_chars + 1),
                self.buffer.rfind(" ", 0, self.max_chars + 1),
            )
            if split_at < self.max_chars // 2:
                split_at = self.max_chars
            text = self.buffer[:split_at].strip()
            self.buffer = self.buffer[split_at:].lstrip()
            if text:
                self.emit(text)

    def feed(self, text: str) -> None:
        self.buffer += text
        while match := self.SENTENCE_END.search(self.buffer):
            sentence = self.buffer[: match.end()].strip()
            self.buffer = self.buffer[match.end() :]
            if sentence:
                self._emit_bounded(sentence)
        self._emit_long_chunks()

    def flush(self) -> None:
        self._emit_long_chunks()
        text = self.buffer.strip()
        self.buffer = ""
        if text:
            self._emit_bounded(text)


class EchoMatcher:
    """Compare normalized speech without depending on an audio provider."""

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def matches(transcript: str, spoken: str) -> bool:
        transcript_words = transcript.split()
        spoken_words = spoken.split()
        shorter_length = min(len(transcript_words), len(spoken_words))
        if shorter_length == 0:
            return False
        if transcript in spoken or spoken in transcript:
            return min(len(transcript), len(spoken)) >= 6

        matcher = SequenceMatcher(None, transcript_words, spoken_words)
        if matcher.ratio() >= 0.72:
            return True
        longest_match = max(matcher.get_matching_blocks(), key=lambda block: block.size)
        return longest_match.size >= 3 and longest_match.size / shorter_length >= 0.70


@dataclass(frozen=True)
class TranscriptEntry:
    """One transcript item sent to Codex with its capture timestamp."""

    speaker: str
    text: str
    timestamp: str


@dataclass(frozen=True)
class CodexRequest:
    """A serialized reply request and the context accumulated before it."""

    reply_to: str
    entries: tuple[TranscriptEntry, ...]


@dataclass
class TranscriptRouter:
    """Accumulate chronological context and create requests at reply boundaries."""

    pending_context: list[TranscriptEntry] = field(default_factory=list)

    def ingest(
        self, speaker: str, text: str, timestamp: str, respond: bool
    ) -> CodexRequest | None:
        self.pending_context.append(TranscriptEntry(speaker, text, timestamp))
        if not respond:
            return None
        request = CodexRequest(speaker, tuple(self.pending_context))
        self.pending_context.clear()
        return request
