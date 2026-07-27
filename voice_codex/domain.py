"""Transcript, response-policy, speech-matching, and turn-state logic.

Nothing here opens a device, spawns a process, or makes a network call. The
locks and the injected clock exist so that this logic can be shared with the
threads in the runtime, not because it talks to anything.
"""

from __future__ import annotations

import re
import threading
import time
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


class TurnGate:
    """Track which response turn is current and whether it may still speak.

    Speech is produced across several threads and an event loop, so a sentence
    can finish synthesizing after the turn that requested it was interrupted.
    Every stage therefore re-checks its own turn number rather than trusting
    that it is still wanted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current_turn = 0
        self.cancelled = False
        self.enabled = True

    def begin_turn(self) -> None:
        with self._lock:
            self.current_turn += 1
            self.cancelled = False

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True

    def set_enabled(self, enabled: bool) -> bool:
        """Set the enabled flag; report whether speech must now be stopped."""
        with self._lock:
            self.enabled = enabled
        return not enabled

    def is_active(self, turn: int) -> bool:
        with self._lock:
            return turn == self.current_turn and not self.cancelled

    def accepting_turn(self) -> tuple[int, bool]:
        """Return the current turn and whether new speech may join it."""
        with self._lock:
            return self.current_turn, self.enabled and not self.cancelled


class EchoMemory:
    """Remember recently spoken text so the microphones can ignore it.

    Retention is a deadline rather than a queue length: a sentence stays
    recognizable for as long as it could still be heard, and entries are
    expired lazily on the next lookup rather than by a timer thread.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._expiry: dict[str, float] = {}

    def remember(self, text: str, retention: float, replace: bool = False) -> None:
        """Record text as spoken.

        ``replace`` shortens an existing deadline; without it a deadline only
        ever moves later, so re-queuing a sentence cannot make it expire sooner.
        """
        normalized = EchoMatcher.normalize(text)
        if not normalized:
            return
        expires_at = self._clock() + retention
        with self._lock:
            if replace:
                self._expiry[normalized] = expires_at
            else:
                self._expiry[normalized] = max(
                    expires_at, self._expiry.get(normalized, 0)
                )

    def matches(self, text: str) -> bool:
        """Return True when a transcript resembles something recently spoken."""
        transcript = EchoMatcher.normalize(text)
        if not transcript:
            return False
        now = self._clock()
        with self._lock:
            for spoken in [
                spoken
                for spoken, expires_at in self._expiry.items()
                if expires_at <= now
            ]:
                del self._expiry[spoken]
            recent = tuple(self._expiry)
        return any(EchoMatcher.matches(transcript, spoken) for spoken in recent)


class SpeakerGate:
    """Decide which completed turns trigger a reply, as the policy changes.

    A policy names speakers that may not exist in this session: selecting
    "both" with no Them output must not make Them replies possible. The gate
    therefore intersects every policy with the speakers actually available.
    """

    def __init__(self, speakers, available):
        self.available = frozenset(available)
        self.active = frozenset(speakers) & self.available

    def set_policy(self, policy_name: str) -> None:
        self.active = resolve_response_policy(policy_name).speakers & self.available

    def should_respond(self, speaker: str) -> bool:
        return speaker in self.active


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
