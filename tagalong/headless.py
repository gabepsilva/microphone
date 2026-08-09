"""Headless session host — store write-through without Textual (#102 D9).

Owns the same ``SessionState`` / ``TranscriptStore`` surface the TUI façade
exposes to channels, Codex, and the EventPump, but ``run`` blocks on a stop
event (SIGINT / SIGTERM / ``stop()``) instead of mounting an interface.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from .control.transcript import TranscriptStore
from .domain import TAGA, VOICE
from .presentation import Entry
from .tui import (
    SESSION_STATE_FIELDS,
    OrderedDeltaBuffer,
    SessionState,
    TuiHooks,
    entry_is_open,
)


class SpeechActivity(Protocol):
    def is_speaking(self) -> bool: ...


class HeadlessSession:
    """Presentation + channel host with no Textual app thread."""

    def __init__(
        self,
        state: SessionState | None = None,
        countdown: object = None,
        speech: SpeechActivity | None = None,
        transcript: TranscriptStore | None = None,
        **hooks: object,
    ) -> None:
        del countdown  # accepted for VoiceCodexTUI call-site parity
        self.state = state or SessionState()
        self.hooks = TuiHooks()
        for name, value in hooks.items():
            setattr(self.hooks, name, value)
        self.transcript = transcript if transcript is not None else TranscriptStore()
        self.speech = speech
        self._stop = threading.Event()
        self._thinking_started: float | None = None
        self._partial_pending = False
        self._partial_lock = threading.Lock()
        self._provisional_turns: dict[str, list[Entry]] = {}
        self._answer_deltas = OrderedDeltaBuffer()
        self._reasoning_deltas = OrderedDeltaBuffer()
        self._publish_partial: Callable[[str, str, int], None] | None = None
        self._publish_session_state: Callable[[dict[str, object]], None] | None = None
        self._partial_seq = 0
        self._entries: list[Entry] = []
        self._streaming: Entry | None = None
        self._reasoning: Entry | None = None
        self._command: Entry | None = None

    def bind_partial_publisher(self, publish: Callable[[str, str, int], None]) -> None:
        """Mirror SessionState partials onto controller ``AppState`` (Q3a)."""
        self._publish_partial = publish

    def bind_session_state_publisher(
        self, publish: Callable[[dict[str, object]], None]
    ) -> None:
        """Mirror live session state onto controller ``AppState``."""
        self._publish_session_state = publish

    def _publish_state(self, changed: dict[str, object]) -> None:
        publish = self._publish_session_state
        if publish is not None and changed:
            publish(changed)

    def transcript_entries(self) -> list[Entry]:
        """Accepted-only rows for ``transcript.save`` (F5 / recorded view)."""
        return list(self.transcript.transcript_entries())

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Block until ``stop`` or a terminating signal."""
        previous_int = signal.signal(signal.SIGINT, self._signal_stop)
        previous_term = signal.signal(signal.SIGTERM, self._signal_stop)
        try:
            while not self._stop.wait(0.1):
                self._tick_speaking()
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)

    def stop(self) -> None:
        """Unblock :meth:`run`."""
        self._stop.set()

    def _signal_stop(self, _signum: int, _frame: object) -> None:
        self.stop()

    def _tick_speaking(self) -> None:
        speaking = self.speech is not None and self.speech.is_speaking()
        if speaking == self.state.codex_speaking:
            return
        self.state.codex_speaking = speaking
        self._publish_state({"codex_speaking": speaking})

    # -- store helpers -----------------------------------------------------

    def _stamp(self) -> str:
        return datetime.now(UTC).astimezone().strftime("%H:%M:%S")

    def _add_entry(self, entry: Entry, *, record: bool = True) -> Entry:
        entry.stamp = entry.stamp or self._stamp()
        self.transcript.append(entry, provisional=not record)
        self._entries.append(entry)
        # Same gate as VoiceCodexApp.add_entry — commands stay open until
        # exit_code is set, so output/status reach the session file.
        if record and not entry_is_open(entry):
            self._record(entry)
        return entry

    def _record(self, entry: Entry) -> None:
        if entry.recorded or self.hooks.on_entry is None:
            return
        if self.hooks.on_entry(entry) is not False:
            entry.recorded = True

    def _resolve_entries(self, entries: list[Entry], *, accepted: bool) -> None:
        if accepted:
            self.transcript.accept(entries)
            for entry in entries:
                self._record(entry)
            return
        self.transcript.reject(entries)
        rejected = {id(entry) for entry in entries}
        self._entries = [entry for entry in self._entries if id(entry) not in rejected]

    # -- partials ----------------------------------------------------------

    def _show_partial(self, source: str, text: str) -> None:
        with self._partial_lock:
            self.state.partial_source = source
            self.state.partial_text = text
            self._partial_seq += 1
            seq = self._partial_seq
            publish = self._publish_partial
        if publish is not None:
            publish(source, text, seq)

    def _clear_partial(self) -> None:
        self._show_partial("", "")

    def update(self, speaker: str, text: str) -> None:
        self._show_partial(speaker, text)

    def finish_turn(self, speaker: str, accepted: bool = True) -> None:
        with self._partial_lock:
            clear = self.state.partial_source == speaker
        if clear:
            self._clear_partial()
        entries = self._provisional_turns.pop(speaker, [])
        self._resolve_entries(entries, accepted=accepted)
        if not accepted:
            self.state.echoes_cut += 1
            self._publish_state({"echoes_cut": self.state.echoes_cut})

    def close_speaker(self, speaker: str) -> None:
        self.finish_turn(speaker)

    def commit(self, speaker: str, text: str) -> None:
        self._show_partial("", "")
        entry = Entry(kind="speech", source=speaker, text=text)
        self._provisional_turns.setdefault(speaker, []).append(entry)
        self._add_entry(entry, record=False)

    def show_message(self, speaker: str, text: str) -> None:
        """Draw a message a remote client sent; nothing local drew it."""
        self._add_entry(Entry(kind="speech", source=speaker, text=text))

    def note(self, text: str) -> None:
        self._add_entry(Entry(kind="note", text=text))

    def reset_transcript(self) -> None:
        self._show_partial("", "")
        self._provisional_turns.clear()
        self.finish_recording()
        self.transcript.clear()
        self._entries.clear()
        self._streaming = None
        self._reasoning = None
        self._command = None

    def finish_recording(self) -> None:
        self._flush_answer_deltas()
        self._flush_reasoning_deltas()
        for entry in self._entries:
            if not entry.recorded:
                self._record(entry)

    # -- Codex stream ------------------------------------------------------

    def begin_codex(self) -> None:
        self._clear_partial()
        self._answer_deltas.take()
        self._reasoning_deltas.take()

    def codex_message_open(self, reply_to: str) -> None:
        self.state.codex_state = f"replying to {reply_to}"
        self._publish_state({"codex_state": self.state.codex_state})
        entry = Entry(kind="speech", source=TAGA, reply_to=reply_to, streaming=True)
        self._streaming = self._add_entry(entry)

    def codex_delta(self, delta: str) -> None:
        if not self._answer_deltas.append(delta):
            return
        self._flush_answer_deltas()

    def _flush_answer_deltas(self) -> None:
        text = self._answer_deltas.take()
        if text:
            self._codex_delta_impl(text)

    def codex_message_close(self) -> None:
        return

    def _codex_delta_impl(self, delta: str) -> None:
        entry = self._streaming
        if entry is None:
            self.codex_message_open(VOICE)
            entry = self._streaming
        if entry is None:
            raise RuntimeError("Could not create a streaming Taga transcript row.")
        self.transcript.append_text(entry, delta)

    def end_codex(self) -> None:
        self.reasoning_completed()
        self.state.codex_state = "idle"
        self._publish_state({"codex_state": self.state.codex_state})
        self._flush_answer_deltas()
        entry = self._streaming
        if entry is not None:
            self.transcript.finalize(entry)
            self._record(entry)
            self._streaming = None

    def reasoning_started(self) -> None:
        self.state.codex_state = "thinking"
        self._publish_state({"codex_state": self.state.codex_state})
        self._thinking_started = time.monotonic()
        self._streaming = None
        self._reasoning = self._add_entry(
            Entry(kind="reasoning", source=TAGA, streaming=True)
        )

    def reasoning_delta(self, delta: str) -> None:
        if not self._reasoning_deltas.append(delta):
            return
        self._flush_reasoning_deltas()

    def _flush_reasoning_deltas(self) -> None:
        text = self._reasoning_deltas.take()
        if text:
            entry = self._reasoning
            if entry is None:
                self.reasoning_started()
                entry = self._reasoning
            if entry is None:
                raise RuntimeError("Could not create a streaming reasoning row.")
            self.transcript.append_text(entry, text)

    def reasoning_completed(self) -> None:
        elapsed = (
            None
            if self._thinking_started is None
            else time.monotonic() - self._thinking_started
        )
        self._thinking_started = None
        self._flush_reasoning_deltas()
        entry = self._reasoning
        if entry is None:
            return
        self.transcript.finalize(entry, seconds=elapsed)
        self._record(entry)
        self._reasoning = None

    def command_started(self, command: str) -> None:
        self.state.codex_state = "running command"
        self._publish_state({"codex_state": self.state.codex_state})
        self._streaming = None
        self._command = self._add_entry(Entry(kind="command", text=command))

    def command_output(self, delta: str) -> None:
        entry = self._command
        if entry is None:
            return
        self.transcript.append_command_output(entry, delta)

    def command_completed(self, exit_code: int | None) -> None:
        entry = self._command
        if entry is None:
            return
        code = -1 if exit_code is None else exit_code
        self.transcript.finalize(entry, exit_code=code)
        self._record(entry)
        self._command = None

    def tool_called(self, server: str, tool: str) -> None:
        self.note(f"tool {server}.{tool}")

    def tool_completed(self, status: str) -> None:
        self.note(f"tool status: {status}")

    def token_usage(self, total_tokens: int) -> None:
        self.state.tokens = total_tokens
        self._publish_state({"tokens": total_tokens})

    def error(self, message: str) -> None:
        self.note(message)

    # -- panels / device lists ---------------------------------------------

    def set_audio(self, channel: str, *, active: bool) -> None:
        target = {"mic": self.state.mic, "audio": self.state.audio}.get(channel)
        if target is not None:
            target.active = active

    def set_codex(self, **fields: object) -> None:
        changed: dict[str, object] = {}
        for key, value in fields.items():
            attribute = f"codex_{key}"
            if hasattr(self.state, attribute):
                setattr(self.state, attribute, value)
                if attribute in SESSION_STATE_FIELDS:
                    changed[attribute] = value
        self._publish_state(changed)

    def set_codex_catalog(
        self,
        models: list[tuple[str, str]],
        efforts_by_model: dict[str, list[str]],
        default_effort_by_model: dict[str, str],
    ) -> None:
        self.state.codex_models = models
        self.state.codex_efforts_by_model = efforts_by_model
        self.state.codex_default_effort_by_model = default_effort_by_model
        efforts = efforts_by_model.get(self.state.codex_model, [])
        if efforts:
            self.state.codex_efforts = efforts
            if self.state.codex_effort not in efforts:
                self.state.codex_effort = (
                    default_effort_by_model.get(self.state.codex_model) or efforts[0]
                )

    def set_audio_streams(self, applications: list[tuple[str, str]]) -> None:
        self.state.audio_streams = list(applications)

    def set_microphones(self, microphones: list[tuple[str, str]]) -> None:
        self.state.microphones = list(microphones)

    def set_session(self, **fields: object) -> None:
        changed: dict[str, object] = {}
        for key, value in fields.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                if key in SESSION_STATE_FIELDS:
                    changed[key] = value
        self._publish_state(changed)

    def set_status(self, status: str, live: bool = True) -> None:
        self.state.status = status
        self.state.live = live
