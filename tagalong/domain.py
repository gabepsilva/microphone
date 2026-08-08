"""Transcript, response-policy, speech-matching, and turn-state logic.

Nothing here opens a device, spawns a process, or makes a network call. The
locks and the injected clock exist so that this logic can be shared with the
threads in the runtime, not because it talks to anything.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from markdown_it import MarkdownIt

VOICE = "Voice"
TEXT = "Text"
AUDIO = "Audio"
TAGA = "Taga"
AGENT = "Agent"


@dataclass(frozen=True, slots=True)
class UserTextMessage:
    """A typed Text turn, optionally carrying pasted image attachments.

    ``images`` holds opaque attachment ids returned by ``attachment.upload``.
    Tokens like ``[Image #1]`` in ``text`` are the human-facing handles; the
    controller resolves ids to on-disk paths before Codex sees the turn.
    """

    text: str
    images: tuple[str, ...] = ()


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
    "audio": ResponsePolicy("audio", "Audio", "Audio", frozenset({AUDIO})),
    "both": ResponsePolicy(
        "both",
        "Voice and Audio",
        "Voice + Audio",
        frozenset({VOICE, AUDIO}),
    ),
    "voice": ResponsePolicy("voice", "Voice", "Voice", frozenset({VOICE})),
    "quiet": ResponsePolicy(
        "quiet", "Taga will be quiet for voice", "Stay silent", frozenset()
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


# --------------------------------------------------------------------------
# Markdown → speakable prose
#
# Taga writes markdown for the transcript; the same stream is sentence-chunked
# for TTS. Markup spoken aloud is worse than dropping it. This is the cheap
# half of dual-channel replies — display keeps the source, speech gets a
# CommonMark walk with a speech-specific render policy.
#
# A stack of regex substitutions is the wrong tool: it false-positives on
# code (``__init__``, ``*ptr``, ``3 * 4``) and still leaks unclosed markers.
# Applied to finished chunks from the sentence chunker, not raw deltas.
# --------------------------------------------------------------------------

# One parser for the process. CommonMark only — no linkify (would invent
# URLs from bare hostnames and then we'd have to strip them again).
_SPEECH_MD = MarkdownIt("commonmark")
# Block types with nothing worth hearing, or whose text is source code.
_SPEECH_SKIP_BLOCKS = frozenset({"fence", "code_block", "hr", "html_block"})
# CommonMark leaves unclosed ``**`` as literal text; residual markers are
# still audible garbage. Single ``*`` is left alone so ``3 * 4`` survives.
_ORPHAN_BOLD_MARKERS = re.compile(r"\*\*")
# Not in CommonMark; models still emit it. One cheap pass after the parse.
_STRIKE_MARKERS = re.compile(r"~~(.+?)~~")
# Bare URLs (and autolink labels that equal the href) are not words. Spoken
# letter-by-letter they destroy a sentence; drop them after the walk. Trailing
# sentence punctuation is not part of the URL and must stay.
_BARE_URL = re.compile(r"https?://[^\s<>\]]+")
_URL_TRAIL_PUNCT = ".,;:!?"
_SPEECH_WHITESPACE = re.compile(r"\s+")
# Dropping a URL can leave "word ." — tidy for the ear and for tests.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
# Emphasis markers that must survive speech (dunder-style names). Asterisk
# markers are pure stress and are dropped.
_KEEP_EMPHASIS_MARKUP = frozenset({"_", "__"})
_BREAK_KINDS = frozenset({"softbreak", "hardbreak"})
_EMPHASIS_KINDS = frozenset({"strong_open", "em_open", "strong_close", "em_close"})


def _drop_bare_urls(text: str) -> str:
    """Remove URL spellings while keeping any sentence punctuation after them."""

    def replace(match: re.Match[str]) -> str:
        url = match.group(0)
        body = url.rstrip(_URL_TRAIL_PUNCT)
        # Only the trailing punctuation (if any) remains for the sentence.
        return url[len(body) :]

    return _BARE_URL.sub(replace, text)


def _speech_inline_piece(token) -> str:
    """One inline token as speech text (empty means skip)."""
    kind = token.type
    if kind in {"text", "code_inline"}:
        # Code keeps identifiers and paths literal, including underscores.
        return token.content
    if kind in _BREAK_KINDS:
        return " "
    if kind == "image":
        # Alt text only; CommonMark puts it in children when present.
        if token.children:
            return _render_speech_inlines(token.children)
        return token.content or ""
    if kind in _EMPHASIS_KINDS:
        # Asterisk emphasis is pure stress — drop the markers. Underscore
        # emphasis is how dunder names parse when the model forgets backticks
        # (``__init__`` → strong "init"); keep the underscores so the ear
        # still hears a dunder, not a bare word.
        if token.markup in _KEEP_EMPHASIS_MARKUP:
            return token.markup
        return ""
    # link_open / link_close / html_inline and anything unknown: labels are
    # sibling text tokens; destinations and tags are not spoken. Recurse only
    # when the token actually nests inlines.
    if token.children:
        return _render_speech_inlines(token.children)
    return ""


def _render_speech_inlines(tokens) -> str:
    """Render inline tokens for the ear, not the screen."""
    return "".join(_speech_inline_piece(token) for token in tokens)


def markdown_to_speech(text: str) -> str:
    """Reduce markdown source to plain prose a TTS engine can speak.

    Policy (deliberate, not accidental):
    - Fenced / indented code blocks are dropped (source aloud is noise).
    - Inline code keeps its content (paths, identifiers).
    - Links keep the label; destinations and bare autolinks are dropped.
    - Images keep the alt text only.
    - Asterisk bold/italic markers drop; underscore markers around emphasis
      are kept so dunder names survive a missing code span.
    - Headings, lists, and quotes contribute their text only.
    - Residual unclosed ``**`` markers are stripped after the parse.

    Returns an empty string when nothing speakable remains.
    """
    if not text:
        return ""

    pieces: list[str] = []
    for token in _SPEECH_MD.parse(text):
        if token.type in _SPEECH_SKIP_BLOCKS:
            continue
        if token.type != "inline":
            continue
        if not token.children:
            continue
        piece = _render_speech_inlines(token.children)
        if piece:
            pieces.append(piece)

    spoken = " ".join(pieces)
    spoken = _STRIKE_MARKERS.sub(r"\1", spoken)
    spoken = _ORPHAN_BOLD_MARKERS.sub("", spoken)
    spoken = _drop_bare_urls(spoken)
    spoken = _SPEECH_WHITESPACE.sub(" ", spoken)
    spoken = _SPACE_BEFORE_PUNCT.sub(r"\1", spoken)
    return spoken.strip()


# Selection chrome (#128b). Kept out of markdown_to_speech — that path is on
# every Codex reply (codex.py → speech_sink), and chrome rules must not strip
# Taga quoting a Slack header back to the user.
#
# Chrome occupies structural positions (own line / trailing metadata). Whole-
# body substitutions would delete ordinary English — clocks (`10:30:45`),
# "3 replies", capitalized "Edited" — so each rule is anchored like
# ``_LEADING_NAME_TIME`` below. Shortcodes require a letter so ``:30:`` inside
# a timestamp never matches.
_EMOJI_SHORTCODE = re.compile(r":[a-zA-Z][a-zA-Z0-9_+-]*:")
# Reaction clusters like ``:eyes: 4`` / ``:tada: 2`` (same line only — do not
# eat a following line's ``3 replies`` count across a newline).
_REACTION_CLUSTER = re.compile(r"(?::[a-zA-Z][a-zA-Z0-9_+-]*:[ \t]*)+\d+")
# Slack reply chrome is its own line ("3 replies"), never mid-sentence.
_REPLY_COUNT_LINE = re.compile(r"(?m)^[ \t]*\d+[ \t]+replies?[ \t]*$")
# Slack edit marker is parenthesised and trailing, not the word "Edited".
_EDITED_TRAILING = re.compile(r"[ \t]*\(edited\)[ \t]*(?=\n|$)", re.IGNORECASE)
# First-line-only Slack-ish header: Capitalized Name(s) then HH:MM AM/PM.
# Unanchored would eat mid-paragraph "we shipped it at 10:42 AM".
_LEADING_NAME_TIME = re.compile(
    r"^(?:[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,5})\s+"
    r"\d{1,2}:\d{2}\s*(?:AM|PM)\b\s*",
)


def strip_chrome(text: str) -> str:
    """Drop source-UI chrome from a primary selection before TTS prep.

    Compose only on the selection path: ``strip_chrome`` →
    :func:`markdown_to_speech` → :class:`SentenceChunker`. Never call this
    from the Codex render path.
    """
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    first = _LEADING_NAME_TIME.sub("", lines[0], count=1)
    body = first + "".join(lines[1:])
    body = _REACTION_CLUSTER.sub(" ", body)
    body = _EMOJI_SHORTCODE.sub(" ", body)
    body = _REPLY_COUNT_LINE.sub(" ", body)
    body = _EDITED_TRAILING.sub(" ", body)
    body = _SPEECH_WHITESPACE.sub(" ", body)
    body = _SPACE_BEFORE_PUNCT.sub(r"\1", body)
    return body.strip()


def speech_sink(speak: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap a ``speak`` callback so it only receives cleaned markdown chunks.

    Empty results (code-only fences, pure markup) are not forwarded, so the
    engine never queues silence as a sentence.
    """

    def emit(text: str) -> None:
        cleaned = markdown_to_speech(text)
        if cleaned:
            speak(cleaned)

    return emit


class SentenceChunker:
    """Turn streamed text into sentence-sized chunks with a hard size cap.

    The opening chunk of a turn may also break at a clause, or at a word
    bound when the model starts a long sentence without punctuation.
    Everything after it waits for a sentence, because by then speech is
    already playing and a fragment only costs prosody; the first chunk is the
    one the listener is waiting through silence for.
    """

    SENTENCE_END = re.compile(
        r'(?<=[.!?])(?:["”\N{RIGHT SINGLE QUOTATION MARK}\')\]]*)\s+'
    )
    CLAUSE_END = re.compile(r"[,;:\N{EM DASH}\N{EN DASH}]\s")
    WORD = re.compile(r"\S+")
    # Below this, an opening clause is not worth hearing on its own: "Sure,"
    # spoken alone is a worse start than the fraction of a second it saves.
    FIRST_CLAUSE_MIN_CHARS = 20
    # Cap the first spoken fragment when no sentence or clause end arrives.
    # A twelve-word start is long enough to be a real answer and short enough
    # that Piper can begin while the rest of the sentence is still streaming.
    # Keep in lockstep with CODEX_DEVELOPER_INSTRUCTIONS, which quotes this
    # value so the model and the chunker agree on the opening length.
    FIRST_CHUNK_MAX_WORDS = 12

    def __init__(
        self,
        emit,
        max_chars: int = 400,
        first_clause_min_chars: int | None = FIRST_CLAUSE_MIN_CHARS,
        first_chunk_max_words: int | None = FIRST_CHUNK_MAX_WORDS,
    ) -> None:
        self.emit = emit
        self.max_chars = max_chars
        self.first_clause_min_chars = first_clause_min_chars
        self.first_chunk_max_words = first_chunk_max_words
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

    def _emit_first_words(self) -> None:
        """Release the opening words when no clause or sentence end has arrived.

        Clause breaking covers the common case. This covers the rest: a long
        first sentence with no comma would otherwise hold speech until a
        period or the character cap. Requiring one word past the limit keeps
        a complete short answer buffered until it finishes or punctuates.
        """
        if self.has_emitted or self.first_chunk_max_words is None:
            return
        limit = self.first_chunk_max_words
        if limit < 1:
            return
        # Leading whitespace must not block the match: a feed can arrive with
        # padding, and re.match on the raw buffer would wait forever.
        text = self.buffer.lstrip()
        words = list(self.WORD.finditer(text))
        if len(words) <= limit:
            return
        cut = words[limit - 1].end()
        chunk = text[:cut]
        self.buffer = text[cut:].lstrip()
        self._emit_bounded(chunk)

    def feed(self, text: str) -> None:
        self.buffer += text
        while match := self.SENTENCE_END.search(self.buffer):
            sentence = self.buffer[: match.end()].strip()
            self.buffer = self.buffer[match.end() :]
            if sentence:
                self._emit_bounded(sentence)
        if not self.has_emitted:
            self._emit_first_clause()
        if not self.has_emitted:
            self._emit_first_words()
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
    Taga still speaking": between two sentences the player has exited and the
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
    """Track how long Taga takes to reach the first word of a reply.

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

    Firing at ``window - estimate`` puts Taga's first word at the moment the
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


# A finished-looking turn does not need the full silence window. Short enough
# that a clear Voice command feels snappy; long enough that a trailing breath
# or a late STT revision can still cancel before Codex commits.
CLEAR_TURN_SILENCE = 0.55

# Trailing glue that usually means the speaker is mid-phrase, not done.
_INCOMPLETE_TAIL = re.compile(
    r"(?:\b(?:and|or|but|so|because|if|when|while|the|a|an|to|for|with|of)\b"
    r"|[,:;…]|\.\.\.)$",
    re.IGNORECASE,
)
_TAGA_ADDRESS = re.compile(r"\btaga\b", re.IGNORECASE)


def same_turn_text(left: str, right: str) -> bool:
    """True when two speculative transcripts target the same answer."""
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def silence_for_turn(text: str, configured: float) -> float:
    """Pick a silence wait for ``text``, never longer than ``configured``.

    The configured window stays the ceiling for ambiguous speech. Clear
    endings — a question mark, an exclamation, or a short address to Taga —
    take the shorter ``CLEAR_TURN_SILENCE`` instead, so command-style Voice
    turns stop waiting out a meeting-length pause. Incomplete tails keep the
    full window.
    """
    configured = TurnSilence.clamp(configured)
    stripped = text.strip()
    if not stripped:
        return configured
    if _INCOMPLETE_TAIL.search(stripped):
        return configured
    short = min(configured, CLEAR_TURN_SILENCE)
    short = max(TurnSilence.MINIMUM, short)
    if stripped.endswith(("?", "!")):
        return short
    words = stripped.split()
    if _TAGA_ADDRESS.search(stripped) and len(words) <= 12:
        return short
    return configured


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

    def started(self, speaker: str, seconds: float | None = None) -> None:
        """Record that a speaker's silence timer has just begun.

        The duration comes from the caller rather than from the window,
        because a turn held open for a speaker who is still talking runs on a
        short grace instead — and a countdown showing the full window there
        would promise a wait the listener is not going to take. It falls back
        to the window for callers arming the ordinary one.
        """
        window = self.window.seconds if seconds is None else seconds
        due_at = self._clock() + window
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


class SpeakerPresence:
    """Whether a channel is hearing its own speaker, at this moment.

    The silence timer keys off transcription events, and those arrive about
    half a second after someone starts talking: the model needs that much
    audio before it will commit to a word. A speaker who resumes inside that
    half second has their turn sent out from under them, because nothing the
    timer can see has happened yet.

    The level tap answers the same question from the audio itself, without
    waiting for a word. What it cannot do is say *whose* sound it is — an open
    microphone hears the assistant's own speech and the far end coming out of
    the speakers as readily as the person sitting in front of it. The
    suppressors are the things known to be making sound that is not this
    speaker; while any of them holds, the tap is evidence of nothing and the
    answer is no.

    A monitored sink needs none of them: it carries what the meeting app wrote
    and nothing else. An open microphone needs both.
    """

    def __init__(self, source, suppressors=()) -> None:
        self.source = source
        self.suppressors = tuple(suppressors)

    def speaking(self) -> bool:
        """Report whether this speaker is audibly talking right now."""
        if not self.source.hearing_sound:
            return False
        return not any(suppressed() for suppressed in self.suppressors)


class SpeakerGate:
    """Decide which completed turns trigger a reply, as the policy changes.

    A policy names speakers that may not exist in this session: selecting
    "both" with no far end must not make Audio replies possible. The gate
    therefore intersects every policy with the speakers actually available.

    Both halves move while the session runs — the policy from its picker, the
    available speakers as a far end is chosen or dropped — so the policy is
    kept rather than folded into the result. Folding it would mean a far end
    arriving after the policy was set could never be answered, because nothing
    would remember that the policy had asked for it.
    """

    def __init__(self, speakers, available):
        self.available = frozenset(available)
        self.requested = frozenset(speakers)
        self.active = self.requested & self.available

    def set_policy(self, policy_name: str) -> None:
        self.requested = resolve_response_policy(policy_name).speakers
        self.active = self.requested & self.available

    def set_available(self, available) -> None:
        """Say which speakers this session now has, keeping the policy asked for."""
        self.available = frozenset(available)
        self.active = self.requested & self.available

    def should_respond(self, speaker: str) -> bool:
        return speaker in self.active


@dataclass(frozen=True)
class TranscriptEntry:
    """One transcript item sent to Codex with its capture timestamp.

    ``images`` holds absolute filesystem paths for files attached to this
    entry (typed Text only, today). Tokens like ``[Image #1]`` in ``text``
    are human-facing handles. Paths are attached to the Codex turn as local
    image inputs and are not serialised into the text prompt.
    """

    speaker: str
    text: str
    timestamp: str
    images: tuple[str, ...] = ()


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
        self,
        speaker: str,
        text: str,
        timestamp: str,
        respond: bool,
        images: tuple[str, ...] = (),
    ) -> CodexRequest | None:
        self.pending_context.append(
            TranscriptEntry(speaker, text, timestamp, images=tuple(images))
        )
        if not respond:
            del self.pending_context[: -self.max_pending_context]
            return None
        request = CodexRequest(speaker, tuple(self.pending_context))
        self.pending_context.clear()
        return request

    def speculate(
        self,
        speaker: str,
        text: str,
        timestamp: str,
        images: tuple[str, ...] = (),
    ) -> CodexRequest:
        """Build a reply request without consuming the context behind it.

        A speculative turn is abandoned whenever the speaker turns out not to
        have finished. Leaving the entries pending is what makes abandoning it
        lossless: giving up is simply never calling ``commit``, and the next
        request carries everything this one would have.
        """
        self.pending_context.append(
            TranscriptEntry(speaker, text, timestamp, images=tuple(images))
        )
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
