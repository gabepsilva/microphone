#!/usr/bin/env python3
"""Turn streamed transcription events into completed turns for Taga.

The silence timer here is a daemon thread, cancelled on mute and on close, so
a pending flush cannot fire against a torn-down display.
"""

from __future__ import annotations

import threading

from moonshine_voice.transcriber import TranscriptEventListener

from .domain import same_turn_text, silence_for_turn
from .presentation import TranscriptSink


def _flush_match_key(text: str) -> str:
    """Normalize transcript text for post-flush STT catch-up matching."""
    return " ".join(text.split()).casefold().rstrip("?.!,;:")


class ConversationListener(TranscriptEventListener):
    # A deadline reached while its speaker is still audible is pushed back by
    # this much rather than by another whole window: the transcription that
    # will cancel it properly is about half a second behind the speech, so the
    # grace only has to outlast that, and a speaker who really has stopped is
    # answered a grace late instead of a window late.
    EXTENSION_GRACE = 0.5

    def __init__(  # noqa: PLR0913 - pre-existing: audio adapter wiring
        self,
        confidence_threshold,
        turn_silence,
        speaker,
        submit,
        presentation: TranscriptSink,
        on_speech=None,
        countdown=None,
        prefire=None,
        presence=None,
    ):
        self.confidence_threshold = confidence_threshold
        self.turn_silence = turn_silence
        self.speaker = speaker
        self.submit = submit
        self.presentation = presentation
        self.on_speech = on_speech
        # Optional so a listener can run without a display attached to it.
        self.countdown = countdown
        # Optional so a session can wait out the whole window instead, which
        # is what ``--no-codex-prefire`` asks for and what the tests compare
        # every pre-fired path against.
        self.prefire = prefire
        # Optional so a channel with no level tap behind it — and every test
        # that predates one — waits the window out exactly as before.
        self.presence = presence
        self.lock = threading.Lock()
        self.pending = []
        # Live STT line that has not closed yet. Kept so energy-triggered
        # waits and prefires can start before Moonshine commits the last words.
        self._partial = ""
        self.timer = None
        self.prefire_timer = None
        self.prefired = False
        self._prefired_text = ""
        self.timer_generation = 0
        self.extensions = 0
        self.speech_callback_triggered = False
        self.muted = False
        # Text of the turn last accepted by submit/commit, plus the live
        # partial portion of that flush when there was one. Later STT that
        # matches either is catch-up for that utterance, not a new turn.
        self._flushed_text = ""
        self._flushed_partial = ""

    def _stop_counting(self):
        """Take this speaker off the countdown the interface is showing."""
        if self.countdown is not None:
            self.countdown.cleared(self.speaker)

    def _stop_timers(self):
        """Cancel both timers and retire their generation. Caller holds the lock."""
        self.timer_generation += 1
        for timer in (self.timer, self.prefire_timer):
            if timer is not None:
                timer.cancel()
        self.timer = None
        self.prefire_timer = None

    def _drop_prefire(self):
        """Abandon a speculative turn this listener started. Lock not held."""
        if self.prefire is None:
            return
        with self.lock:
            outstanding = self.prefired
            self.prefired = False
            self._prefired_text = ""
        if outstanding:
            self.prefire.cancel(self.speaker)

    def _cancel_prefire_outside(self, cancel: bool) -> None:
        """Cancel Codex speculation after releasing the listener lock."""
        if cancel and self.prefire is not None:
            self.prefire.cancel(self.speaker)

    def set_muted(self, muted):
        """Stop submitting microphone speech while preserving the listener."""
        with self.lock:
            self.muted = muted
            if muted:
                self.pending.clear()
                self._partial = ""
                self._flushed_text = ""
                self._flushed_partial = ""
                self._stop_timers()
                self.extensions = 0
        if muted:
            self._drop_prefire()
            self._stop_counting()

    def _is_muted(self):
        with self.lock:
            return self.muted

    def _speaker_audible(self) -> bool:
        """Whether the level tap says this speaker is talking right now.

        No presence means the session has no tap: treat transcription events
        as the only signal, which is the historical behaviour (always "active"
        so partials and line starts cancel the wait as they used to).
        """
        return self.presence is None or self.presence.speaking()

    def _late_stt_for_flush(self, text: str) -> bool:
        """True when ``text`` is catch-up for the turn already submitted.

        Caller holds the lock. Matches the full flushed string, or the live
        partial that was included in that flush. Trailing sentence punctuation
        is ignored so ``hello`` and ``hello?`` still match. The markers stay
        until this match arrives even if the user has already started again.
        """
        if not text:
            return False
        current = _flush_match_key(text)
        if not current:
            return False
        if self._flushed_text and _flush_match_key(self._flushed_text) == current:
            return True
        return bool(
            self._flushed_partial and _flush_match_key(self._flushed_partial) == current
        )

    def _clear_flush_marker(self) -> None:
        """Caller holds the lock."""
        self._flushed_text = ""
        self._flushed_partial = ""

    def _mark_flushed(self, text: str, partial: str) -> None:
        """Record an accepted flush so late STT can be absorbed. Holds the lock."""
        with self.lock:
            self._flushed_text = text
            self._flushed_partial = partial

    def _text(self, line):
        if line.words:
            return " ".join(
                word.word.strip()
                for word in line.words
                if word.confidence >= self.confidence_threshold
            ).strip()
        return line.text.strip()

    def _buffered_text(self) -> str:
        """Completed lines plus any live partial, as one turn string.

        Caller holds the lock. The partial is what energy-triggered waits and
        prefires use when Moonshine has not closed the last line yet.
        """
        parts = list(self.pending)
        if self._partial:
            parts.append(self._partial)
        return " ".join(parts).strip()

    def _wait_for(self, text: str) -> float:
        """Silence seconds for ``text``, capped by the configured window."""
        return silence_for_turn(text, self.turn_silence.seconds)

    def _revise_prefire_if_stale(self, text: str) -> bool:
        """Drop a speculative turn whose transcript no longer matches ``text``.

        Caller holds the lock for the flag read; cancel runs outside it.
        Returns True when a cancel should be issued after releasing the lock.
        """
        if not self.prefired:
            return False
        if same_turn_text(self._prefired_text, text):
            return False
        self.prefired = False
        self._prefired_text = ""
        return self.prefire is not None

    def _arm_silence(self, text: str) -> bool:
        """Start the wait (and prefire) for ``text``. Caller holds the lock.

        Returns whether an outstanding speculative turn must be cancelled
        outside the lock because its transcript no longer matches.
        """
        cancel = self._revise_prefire_if_stale(text)
        self.extensions = 0
        # Adaptive window from the live buffer (``text`` is that buffer).
        self._start_timer()
        return cancel

    def _flush(self, generation, extendable=True):
        text = ""
        partial = ""
        with self.lock:
            if generation != self.timer_generation:
                return
            extended = extendable and self._extending()
            if extended:
                self.extensions += 1
                # Re-armed while the lock is still held, so the turn is never
                # momentarily left with no deadline on it at all.
                self._start_timer(self.EXTENSION_GRACE, speculate=False)
            else:
                text = self._buffered_text()
                partial = self._partial
                self.pending.clear()
                self._partial = ""
                self.timer = None
            prefired = self.prefired
            self.prefired = False
            self._prefired_text = ""
        if extended:
            # The speaker is still going, so a speculative answer to what they
            # had said by now would be answering half a sentence. The countdown
            # was re-armed above, which is what puts the wait back on screen.
            if prefired and self.prefire is not None:
                self.prefire.cancel(self.speaker)
            return
        self._stop_counting()
        if not text:
            return
        # A partial that never became a completed line still belongs on the
        # transcript: finish_turn alone would clear the live row and lose it.
        if partial:
            self.presentation.commit(self.speaker, partial)
        # A speculative turn that survived to here was right: the window
        # closed without the speaker resuming, so it is the reply. Submitting
        # again would answer the same words twice.
        accepted = False
        if prefired and self.prefire is not None and self.prefire.commit(self.speaker):
            accepted = True
        else:
            # ``False`` means dropped (echo); ``True``/``None`` means taken —
            # tests often wire a bare lambda that returns None.
            accepted = self.submit(self.speaker, text) is not False
        # Completed STT lines are shown immediately, but stay provisional until
        # the submitter has had the chance to reject its own TTS echo. Resolving
        # here keeps genuine speech responsive without leaving rejected echo in
        # either the visible or recorded transcript.
        self.presentation.finish_turn(self.speaker, accepted=accepted)
        if accepted:
            # Only mark catch-up absorption when the turn was actually taken.
            self._mark_flushed(text, partial)

    def _speculate(self, generation):
        """Start answering before the window closes, if it is still this turn."""
        with self.lock:
            if generation != self.timer_generation:
                return
            self.prefire_timer = None
            text = self._buffered_text()
            if not text or self.prefired:
                return
            self.prefired = True
            self._prefired_text = text
        if self.prefire is None or not self.prefire.start(self.speaker, text):
            with self.lock:
                self.prefired = False
                self._prefired_text = ""

    def _prefire_delay(self, window):
        """Seconds to wait before guessing this turn is over, or None to wait.

        A guess that would land at or after the deadline is not a guess; it is
        a slower way of doing what the deadline already does.
        """
        if self.prefire is None:
            return None
        delay = self.prefire.delay(window)
        return delay if 0 < delay < window else None

    def _extension_budget(self) -> int:
        """How many graces one turn may be held open for.

        Bounded by the window itself, so however noisy a room is, a turn waits
        at most twice the silence it was configured with before it is sent. A
        level tap cannot tell speech from a fan, and without a ceiling a room
        loud enough to hold the tap open would never submit anything at all.
        """
        return max(1, int(self.turn_silence.seconds / self.EXTENSION_GRACE))

    def _extending(self) -> bool:
        """Whether this deadline should be pushed back rather than fired.

        Caller holds the lock. An empty buffer is never extended: there is no
        turn to protect, and holding one open would keep a speaker on the
        countdown for sound that is never going to become words.
        """
        return (
            self.presence is not None
            and bool(self._buffered_text())
            and self.extensions < self._extension_budget()
            and self.presence.speaking()
        )

    def _start_timer(self, window=None, speculate=True):
        """Arm the deadline, and the speculative turn that runs ahead of it.

        ``window`` is given for an extension (grace, no speculate) or for an
        adaptive / energy-triggered wait that is shorter than the configured
        ceiling. Omitting it uses the adaptive wait for whatever is buffered.
        """
        if window is None:
            window = self._wait_for(self._buffered_text())
        self._stop_timers()
        generation = self.timer_generation
        self.timer = threading.Timer(window, self._flush, args=(generation,))
        self.timer.daemon = True
        self.timer.start()
        delay = self._prefire_delay(window) if speculate else None
        if delay is not None:
            self.prefire_timer = threading.Timer(
                delay, self._speculate, args=(generation,)
            )
            self.prefire_timer.daemon = True
            self.prefire_timer.start()
        if self.countdown is not None:
            self.countdown.started(self.speaker, window)

    def _cancel_timer(self):
        with self.lock:
            self._stop_timers()
            # Transcription has caught up with the speech the tap heard, so
            # whatever the extensions were spent on has been paid for.
            self.extensions = 0
        self._drop_prefire()
        self._stop_counting()

    def flush_now(self):
        """Submit what is already transcribed without waiting out the silence.

        The silence window exists to decide that a speaker has stopped talking.
        When another speaker's turn is already being answered that question has
        been overtaken: whatever this listener has buffered is context for the
        reply being built now, and holding it until this speaker's own timer
        fires would deliver it one request too late.

        For the same reason this never extends. Holding the buffer back
        because its speaker is still audible is exactly what the caller has
        already decided not to wait for. The same path is the explicit
        "I'm done" control for power users.
        """
        with self.lock:
            self._stop_timers()
            generation = self.timer_generation
        self._flush(generation, extendable=False)

    def on_energy_quiet(self):
        """Arm silence when the level tap falls, without waiting for STT.

        Transcription lags speech by about half a second. Starting the wait
        (and prefire) from the energy drop hides that lag under the silence
        window instead of adding it in front. Late partials and line closes
        revise the buffer; they do not cancel the wait while the tap stays
        quiet. Suppressors on the presence keep TTS / far-end from looking
        like this speaker finishing.
        """
        if self._is_muted():
            return
        if self._speaker_audible():
            return
        cancel = False
        with self.lock:
            text = self._buffered_text()
            if not text:
                return
            if self._late_stt_for_flush(text):
                # Buffer still only holds catch-up for the turn already sent.
                return
            # Already waiting on this turn: a second quiet edge must not
            # restart the clock. Only a stale prefire forces a re-arm.
            stale = self.prefired and not same_turn_text(self._prefired_text, text)
            if self.timer is not None and not stale:
                return
            cancel = self._arm_silence(text)
        self._cancel_prefire_outside(cancel)

    def on_energy_loud(self):
        """Cancel a wait when this speaker is audibly talking again."""
        if self._is_muted():
            return
        if not self._speaker_audible():
            # Tap rose, but suppressors say it is not this speaker (echo /
            # far end / TTS). Leaving the wait alone avoids answering mid-play.
            return
        self.speech_callback_triggered = False
        self._cancel_timer()

    def on_line_started(self, event):  # noqa: ARG002 - Textual/Codex callback signature is fixed
        # When the tap says the speaker is going again, this is a new utterance:
        # clear the live partial and cancel any wait. When the tap is quiet, a
        # new line is usually STT segmenting the same utterance — leave the
        # partial and the energy-armed wait alone so a flush still has text.
        if self._is_muted():
            return
        if self._speaker_audible():
            with self.lock:
                self._partial = ""
            self.speech_callback_triggered = False
            self._cancel_timer()

    def on_line_text_changed(self, event):
        # Partial text updates the live buffer. It only cancels the silence
        # wait while the speaker is still audible: after an energy drop the
        # late STT revisions must not restart the clock.
        if self._is_muted():
            return
        partial = self._text(event.line)
        cancel_wait = False
        cancel_prefire = False
        late = False
        with self.lock:
            if self._late_stt_for_flush(partial):
                # Catch-up for a turn already sent — do not re-open it.
                late = True
            else:
                self._partial = partial
                if self._speaker_audible():
                    cancel_wait = True
                elif self.timer is not None:
                    text = self._buffered_text()
                    cancel_prefire = self._revise_prefire_if_stale(text)
                    desired = self._wait_for(text)
                    # Re-arm when speculation is stale, or when adaptive silence
                    # for the revised partial no longer matches the armed wait.
                    # Presence-quiet here includes TTS suppressors: cancelling on
                    # every quiet partial would also cancel on echo, so audibility
                    # remains the resume signal.
                    if cancel_prefire or desired != self.timer.interval:
                        self.extensions = 0
                        self._start_timer(desired)
        if late:
            return
        self.presentation.update(self.speaker, partial)
        if cancel_wait:
            self._cancel_timer()
        else:
            self._cancel_prefire_outside(cancel_prefire)
        if (
            partial
            and self.on_speech is not None
            and not self.speech_callback_triggered
        ):
            self.speech_callback_triggered = self.on_speech(partial)

    def on_line_completed(self, event):
        if self._is_muted():
            return
        text = self._text(event.line)
        cancel = False
        with self.lock:
            if self._late_stt_for_flush(text):
                # Line close for a partial we already flushed. Leave any newer
                # partial/pending (a follow-up already in progress) alone.
                self._clear_flush_marker()
                return
            self._partial = ""
            if text:
                self.pending.append(text)
            buffered = self._buffered_text()
            if buffered:
                # Energy may already have armed the wait. Fold the completed
                # line into that wait instead of restarting the full window,
                # unless the transcript or adaptive duration changed.
                if self.timer is not None:
                    cancel = self._revise_prefire_if_stale(buffered)
                    desired = self._wait_for(buffered)
                    if cancel or desired != self.timer.interval:
                        self.extensions = 0
                        self._start_timer(desired)
                else:
                    cancel = self._arm_silence(buffered)
        if text:
            self.presentation.commit(self.speaker, text)
        self._cancel_prefire_outside(cancel)

    def close(self):
        with self.lock:
            self._stop_timers()
            self._partial = ""
            self._flushed_text = ""
            self._flushed_partial = ""
        self._drop_prefire()
        self._stop_counting()
        self.presentation.close_speaker(self.speaker)


class PrefireChannel:
    """The four moments a speculative turn has, as one listener sees them.

    A listener knows when a turn is probably over, when it turned out to be
    over, and when it turned out not to be. It does not know about echo
    gating, response policy, or Codex threads. This is the seam between the
    two: schedule on one side, consequences on the other.
    """

    def __init__(self, submitter):
        self.submitter = submitter

    def delay(self, window):
        return self.submitter.prefire_plan.delay(window)

    def start(self, speaker, text) -> bool:
        return self.submitter.prefire(speaker, text)

    def commit(self, speaker) -> bool:
        return self.submitter.commit_prefire(speaker)

    def cancel(self, speaker) -> bool:
        return self.submitter.cancel_prefire(speaker)


class TranscriptSubmitter:
    """Send completed turns to Taga, discarding the assistant's own TTS echo.

    Both microphones can hear Taga speaking. A transcript that matches recent
    speech is dropped rather than answered, and a partial that matches it must
    not interrupt playback either.
    """

    ECHO_PRONE_SPEAKERS = ("Voice", "Audio")

    def __init__(self, conversation, gate, tts, prefire_plan=None):
        self.conversation = conversation
        self.gate = gate
        self.tts = tts
        # Absent when the session waits out every window in full, which is
        # what ``--no-codex-prefire`` selects.
        self.prefire_plan = prefire_plan
        self.listeners = []

    def add_listener(self, listener):
        """Register a channel whose buffer may be swept into a reply's context."""
        self.listeners.append(listener)

    def remove_listener(self, listener):
        """Retire a channel that no longer exists.

        A listener left registered after its transcriber is closed is not
        inert: it still holds whatever it had buffered, and the next reply
        would sweep that stale text in as context for a far end nobody is
        listening to any more.
        """
        if listener in self.listeners:
            self.listeners.remove(listener)

    def channel(  # noqa: PLR0913 - audio adapter wiring, as the listener itself
        self,
        confidence_threshold,
        turn_silence,
        speaker,
        presentation,
        countdown=None,
        presence=None,
    ):
        """Build a listener for a speaker and register it in one step.

        Registration is not left to the caller because a listener that submits
        turns but was never registered still works — it just silently stops
        contributing context, which is the kind of omission a session only
        reveals as a reply that is missing something.
        """
        listener = ConversationListener(
            confidence_threshold,
            turn_silence,
            speaker,
            self.submit,
            presentation,
            on_speech=self.handle_speech,
            countdown=countdown,
            prefire=self._prefire_channel(),
            presence=presence,
        )
        self.add_listener(listener)
        return listener

    def _prefire_channel(self):
        return None if self.prefire_plan is None else PrefireChannel(self)

    def _is_echo(self, speaker, text) -> bool:
        """Report whether a transcript is Taga hearing itself speak."""
        return (
            self.tts is not None
            and speaker in self.ECHO_PRONE_SPEAKERS
            and self.tts.is_likely_echo(text)
        )

    def prefire(self, speaker, text) -> bool:
        """Start answering a turn the silence window has not yet confirmed.

        The same two gates a real submission passes apply here. A speculative
        turn answering the assistant's own echo, or answering a speaker the
        policy stays silent for, would be a turn nobody could have wanted —
        and unlike a late reply, nothing downstream would catch it.
        """
        if not self.gate.should_respond(speaker) or self._is_echo(speaker, text):
            return False
        self._sweep_context(speaker)
        return self.conversation.prefire(speaker, text)

    def commit_prefire(self, speaker) -> bool:
        return self.conversation.commit_prefire(speaker)

    def cancel_prefire(self, speaker) -> bool:
        return self.conversation.cancel_prefire(speaker)

    def _sweep_context(self, replying_to):
        """Flush the channels that only supply context, so this reply carries it.

        Only the channels the policy does not answer are swept. Flushing one
        the policy does answer would queue a second reply from speech its own
        speaker has not finished, which is a turn nobody asked for rather than
        context for this one.
        """
        for listener in self.listeners:
            if listener.speaker != replying_to and not self.gate.should_respond(
                listener.speaker
            ):
                listener.flush_now()

    def end_turn(self, speaker="Voice"):
        """Flush ``speaker``'s buffer now — the explicit "I'm done" control.

        Distinct from ``_sweep_context``, which flushes *other* channels so
        their words ride along as context for someone else's reply.
        """
        for listener in self.listeners:
            if listener.speaker == speaker:
                listener.flush_now()

    def submit(self, speaker, text) -> bool:
        if self._is_echo(speaker, text):
            return False
        respond = self.gate.should_respond(speaker)
        # Swept before this turn is ingested, so the context a speaker supplied
        # earlier is ordered earlier in the request that carries both.
        if respond:
            self._sweep_context(speaker)
        self.conversation.ingest(speaker, text, respond=respond)
        return True

    def handle_speech(self, partial):
        """Interrupt playback for real speech; report whether it was real."""
        if self.tts is None or self.tts.is_likely_echo(partial):
            return False
        self.tts.interrupt()
        return True
