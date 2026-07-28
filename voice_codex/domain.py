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
    """The completed voice turns that are allowed to trigger a reply.

    Both wordings live here because both name the same choice. ``label`` is
    the startup menu's, which has a full terminal line; ``sidebar_label`` is
    the interface picker's, which has 34 columns.
    """

    name: str
    label: str
    sidebar_label: str
    speakers: frozenset[str]


# The one place a response policy is defined. Everything that offers, saves,
# validates, or displays one derives from this mapping rather than restating
# it: the CLI choices, the startup menu and its numbering, the saved config
# value, and the interface picker. Insertion order is menu order.
RESPONSE_POLICIES = {
    "them": ResponsePolicy("them", "Them", "Them", frozenset({THEM})),
    "both": ResponsePolicy(
        "both",
        "User Voice and Them",
        "User Voice + Them",
        frozenset({USER_VOICE, THEM}),
    ),
    "user": ResponsePolicy("user", "User Voice", "User Voice", frozenset({USER_VOICE})),
    "quiet": ResponsePolicy(
        "quiet", "Codex will be quiet for voice", "stay silent", frozenset()
    ),
}
POLICY_NAMES = tuple(RESPONSE_POLICIES)
# Menu numbers are positions, so they follow the mapping rather than a second
# hand-written list that a reordering would silently invalidate.
_POLICY_ALIASES = {
    **{str(number): name for number, name in enumerate(POLICY_NAMES, start=1)},
    **{name: name for name in POLICY_NAMES},
}


def resolve_response_policy(value: str) -> ResponsePolicy:
    """Resolve a CLI answer or persisted policy name into a policy."""
    return RESPONSE_POLICIES[_POLICY_ALIASES[value]]


class SentenceChunker:
    """Turn streamed text into sentence-sized chunks with a hard size cap.

    The opening chunk of a turn may also break at a clause. Everything after
    it waits for a sentence, because by then speech is already playing and a
    fragment only costs prosody; the first chunk is the one the listener is
    waiting through silence for.
    """

    SENTENCE_END = re.compile(
        r'(?<=[.!?])(?:["”\N{RIGHT SINGLE QUOTATION MARK}\')\]]*)\s+'
    )
    CLAUSE_END = re.compile(r"[,;:\N{EM DASH}\N{EN DASH}]\s")
    # Below this, an opening clause is not worth hearing on its own: "Sure,"
    # spoken alone is a worse start than the fraction of a second it saves.
    FIRST_CLAUSE_MIN_CHARS = 20

    def __init__(
        self,
        emit,
        max_chars: int = 400,
        first_clause_min_chars: int | None = FIRST_CLAUSE_MIN_CHARS,
    ) -> None:
        self.emit = emit
        self.max_chars = max_chars
        self.first_clause_min_chars = first_clause_min_chars
        self.buffer = ""
        self.has_emitted = False

    def _split_once(self, text: str) -> tuple[str, str]:
        """Take the longest chunk that fits, and return it with the rest.

        The split prefers the last line or word break inside the cap. A break
        in the first half is worse than none at all — it would emit a sliver
        and leave the bulk of the text still over the cap — so the split falls
        back to the cap itself.
        """
        split_at = max(
            text.rfind("\n", 0, self.max_chars + 1),
            text.rfind(" ", 0, self.max_chars + 1),
        )
        if split_at < self.max_chars // 2:
            split_at = self.max_chars
        return text[:split_at].strip(), text[split_at:].lstrip()

    def _emit_chunk(self, text: str) -> None:
        """Hand one chunk to the engine, recording that the turn has spoken."""
        self.has_emitted = True
        self.emit(text)

    def _emit_bounded(self, text: str) -> None:
        """Emit text as capped chunks, the short remainder included."""
        while len(text) > self.max_chars:
            chunk, text = self._split_once(text)
            if chunk:
                self._emit_chunk(chunk)
        if text:
            self._emit_chunk(text)

    def _emit_long_chunks(self) -> None:
        """Emit whole capped chunks, leaving a short remainder buffered."""
        while len(self.buffer) > self.max_chars:
            chunk, self.buffer = self._split_once(self.buffer)
            if chunk:
                self._emit_chunk(chunk)

    def _emit_first_clause(self) -> None:
        """Release the opening clause so speech can start mid-sentence.

        The search starts at the minimum rather than filtering afterwards, so
        an early comma is passed over instead of ending the chunk: the clause
        that gets spoken is always at least as long as the minimum.
        """
        if self.first_clause_min_chars is None:
            return
        match = self.CLAUSE_END.search(self.buffer, self.first_clause_min_chars)
        if match is None:
            return
        clause = self.buffer[: match.end()].strip()
        self.buffer = self.buffer[match.end() :]
        if clause:
            self._emit_bounded(clause)

    def feed(self, text: str) -> None:
        self.buffer += text
        while match := self.SENTENCE_END.search(self.buffer):
            sentence = self.buffer[: match.end()].strip()
            self.buffer = self.buffer[match.end() :]
            if sentence:
                self._emit_bounded(sentence)
        if not self.has_emitted:
            self._emit_first_clause()
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


class SpeechActivity:
    """Count the sentences a speech engine still owes the listener.

    Asking the player whether a process is alive is not enough to answer "is
    Codex still speaking": between two sentences the player has exited and the
    next one is still being synthesized, so a poll lands in the gap and reads
    silence. At ten frames a second that gap is visible as a flicker.

    A count spans the gaps. It rises when a sentence is accepted and falls
    only once that sentence has been played or abandoned.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = 0

    def queued(self) -> None:
        with self._lock:
            self._pending += 1

    def finished(self) -> None:
        with self._lock:
            # Floored rather than allowed negative: an engine that reports one
            # extra completion would otherwise go quiet for the next sentence.
            self._pending = max(0, self._pending - 1)

    def silenced(self) -> None:
        """Drop everything outstanding, as an interrupt or a shutdown does."""
        with self._lock:
            self._pending = 0

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._pending > 0


class TurnSilence:
    """The quiet a finished turn waits out before it is sent.

    One object rather than a number copied into each listener, the countdown,
    and the sidebar: the value is editable while the session runs, and four
    copies of it would be four chances to change three of them.

    A change lands on the next turn. Re-arming a timer that is already running
    would either cut short a wait the speaker is relying on or extend one they
    have already stopped talking through, and neither is what typing a new
    number asks for.
    """

    MINIMUM = 0.25
    MAXIMUM = 30.0

    def __init__(self, seconds: float) -> None:
        self._lock = threading.Lock()
        self._seconds = self.clamp(seconds)

    @classmethod
    def clamp(cls, seconds: float) -> float:
        return max(cls.MINIMUM, min(cls.MAXIMUM, seconds))

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._seconds

    def set(self, seconds: float) -> float:
        """Adopt a new window, and report the value actually in force."""
        applied = self.clamp(seconds)
        with self._lock:
            self._seconds = applied
        return applied


class TurnLatencyEstimator:
    """Track how long Codex takes to reach the first word of a reply.

    A moving average rather than the last sample: one slow turn should nudge
    the moment a speculative turn fires, not move it across half the silence
    window. The seed is deliberately high, because the two directions are not
    symmetric — overestimating fires the turn earlier and only risks wasting
    it, while underestimating leaves the wait it exists to hide exposed.
    """

    DEFAULT_SEED = 0.75
    SMOOTHING = 0.3

    def __init__(
        self, seed: float = DEFAULT_SEED, smoothing: float = SMOOTHING
    ) -> None:
        self._lock = threading.Lock()
        self._estimate = seed
        self._smoothing = smoothing

    def record(self, seconds: float) -> None:
        """Fold one observed time-to-first-word into the estimate."""
        with self._lock:
            self._estimate += self._smoothing * (seconds - self._estimate)

    @property
    def estimate(self) -> float:
        with self._lock:
            return self._estimate


class PrefirePlan:
    """Decide how far into a silence window a speculative turn should wait.

    Firing at ``window - estimate`` puts Codex's first word at the moment the
    window closes, which is the whole point: the model thinks during the wait
    that was already happening rather than after it. Two bounds keep that from
    becoming reckless. The lead never takes more than ``MAX_LEAD_FRACTION`` of
    the window, so a slow estimate cannot fire the turn the instant a speaker
    draws breath, and the delay never falls below ``MINIMUM_DELAY``, so even a
    short window leaves a moment in which a speaker who has not finished can
    still cancel.
    """

    MAX_LEAD_FRACTION = 0.65
    MINIMUM_DELAY = 0.15

    def __init__(
        self,
        estimator: TurnLatencyEstimator,
        max_lead_fraction: float = MAX_LEAD_FRACTION,
        minimum_delay: float = MINIMUM_DELAY,
    ) -> None:
        self.estimator = estimator
        self.max_lead_fraction = max_lead_fraction
        self.minimum_delay = minimum_delay

    def delay(self, window: float) -> float:
        """Seconds into ``window`` at which the speculative turn should fire."""
        lead = min(self.estimator.estimate, window * self.max_lead_fraction)
        return max(self.minimum_delay, window - lead)


def parse_turn_silence(text: str) -> float | None:
    """Read a typed turn-silence value, or None when it is not one.

    The trailing unit is accepted because the field displays one, so the
    obvious thing to type back is what was already shown. Out-of-range values
    are refused rather than clamped: silently turning a typed 60 into 30 would
    look like the field ignored the keystrokes.
    """
    try:
        seconds = float(text.strip().removesuffix("s").strip())
    except ValueError:
        return None
    # Rejects NaN too, which compares false against everything.
    if not TurnSilence.MINIMUM <= seconds <= TurnSilence.MAXIMUM:
        return None
    return seconds


class TurnSilenceClock:
    """Track how long each speaker's turn has left before it is submitted.

    The listeners own the silence timers; this only records when each one is
    due, so the interface can show the wait without a transcription thread
    driving the display. Reads are a poll rather than a callback on purpose: a
    countdown repaints ten times a second, and pushing that would cross
    threads ten times a second mostly to report that nothing has changed.

    A due deadline is dropped on the next read rather than by a timer of its
    own. The listener clears its own entry when a turn is submitted, so this
    only matters if that ever fails to happen — and a countdown wedged at zero
    would be a worse failure than one that simply disappears.
    """

    def __init__(self, window: TurnSilence, clock=time.monotonic) -> None:
        self.window = window
        self._clock = clock
        self._lock = threading.Lock()
        self._deadlines: dict[str, float] = {}

    def started(self, speaker: str) -> None:
        """Record that a speaker's silence timer has just begun."""
        due_at = self._clock() + self.window.seconds
        with self._lock:
            self._deadlines[speaker] = due_at

    def cleared(self, speaker: str) -> None:
        """Record that a speaker is no longer waiting to be submitted."""
        with self._lock:
            self._deadlines.pop(speaker, None)

    def remaining(self) -> float | None:
        """Seconds until the soonest pending turn fires, or None if none is.

        The soonest wins because it is the one about to interrupt the silence;
        a later speaker's timer is not what the session is waiting on.
        """
        now = self._clock()
        with self._lock:
            for speaker in [
                speaker for speaker, due_at in self._deadlines.items() if due_at <= now
            ]:
                del self._deadlines[speaker]
            due = min(self._deadlines.values(), default=None)
        return None if due is None else due - now


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


# How many unanswered turns are carried as context. Reaching this at all
# means nothing is being replied to — the "stay silent" policy, or a long
# stretch of one-sided talk — and under it the context would otherwise grow
# for the whole session. That costs memory, and it costs far more on the
# reply that finally comes: the entire session would be sent as one request.
# The most recent turns are the ones kept, being the ones a reply is about.
MAX_PENDING_CONTEXT = 200


@dataclass
class TranscriptRouter:
    """Accumulate chronological context and create requests at reply boundaries."""

    pending_context: list[TranscriptEntry] = field(default_factory=list)
    max_pending_context: int = MAX_PENDING_CONTEXT

    def ingest(
        self, speaker: str, text: str, timestamp: str, respond: bool
    ) -> CodexRequest | None:
        self.pending_context.append(TranscriptEntry(speaker, text, timestamp))
        if not respond:
            del self.pending_context[: -self.max_pending_context]
            return None
        request = CodexRequest(speaker, tuple(self.pending_context))
        self.pending_context.clear()
        return request

    def speculate(self, speaker: str, text: str, timestamp: str) -> CodexRequest:
        """Build a reply request without consuming the context behind it.

        A speculative turn is abandoned whenever the speaker turns out not to
        have finished. Leaving the entries pending is what makes abandoning it
        lossless: giving up is simply never calling ``commit``, and the next
        request carries everything this one would have.
        """
        self.pending_context.append(TranscriptEntry(speaker, text, timestamp))
        return CodexRequest(speaker, tuple(self.pending_context))

    def commit(self, request: CodexRequest) -> None:
        """Drop the entries a speculative request turned out to carry.

        The boundary is found by identity rather than by counting, because the
        two ends of the list move independently: entries arriving while the
        speculative turn is in flight extend it, and ``ingest`` can trim the
        oldest off the front. A count taken before either would delete the
        wrong entries; the last entry the request actually carried cannot.

        A boundary that is already gone leaves the list alone. Trimming
        dropped those entries because they had aged out, and the reply they
        belong to is arriving anyway.
        """
        if not request.entries:
            return
        boundary = request.entries[-1]
        for index, entry in enumerate(self.pending_context):
            if entry is boundary:
                del self.pending_context[: index + 1]
                return
