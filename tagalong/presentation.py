"""Display boundary between the runtime and the Textual interface.

The interface implements all of this. Nothing else does: the listener renders
transcript lines, the catalog probe only reports status, and the Codex stream
does neither. Each of those depends on the role it uses rather than on the
whole surface, so a fake in a test — and any future non-Textual display — has
to satisfy only what its collaborator actually calls.
"""

from __future__ import annotations

from typing import Protocol

# What a capture channel shows in place of a device name when it has none. A
# far end can be chosen and dropped while the session runs, so this is a state
# the interface returns to rather than only a value it starts in. It lives here
# because the runtime sets it and the interface draws it, and the runtime must
# not import the interface to name it.
NO_DEVICE = "none"


class TranscriptSink(Protocol):
    """Show speech as it arrives and mark where a turn ends."""

    def update(self, speaker: str, text: str) -> None: ...

    def commit(self, speaker: str, text: str) -> None: ...

    def finish_turn(self, speaker: str) -> None: ...

    def close_speaker(self, speaker: str) -> None: ...


class SessionStatusSink(Protocol):
    """Report what the session is configured to do right now."""

    def note(self, text: str) -> None: ...

    def error(self, message: str) -> None: ...

    def set_codex(self, **fields: object) -> None: ...

    def set_codex_catalog(
        self,
        models: list[tuple[str, str]],
        efforts_by_model: dict[str, list[str]],
        default_effort_by_model: dict[str, str],
    ) -> None: ...


class ApplicationListSink(Protocol):
    """Offer the applications a session can be pointed at as its far end.

    Its own role rather than a line in the status sink: only the refresher
    calls it, and folding it in would make every Codex fake in the tests grow
    a method its subject never reaches for.
    """

    def set_audio_streams(self, applications: list[tuple[str, str]]) -> None: ...


class NewSessionSink(Protocol):
    """Clear the visible transcript after the host starts a fresh session."""

    def reset_transcript(self) -> None: ...


class CodexStreamSink(Protocol):
    """Render one streamed Codex turn: its reasoning, commands and tool calls."""

    def begin_codex(self) -> None: ...

    def codex_message_open(self, reply_to: str) -> None: ...

    def codex_delta(self, delta: str) -> None: ...

    def codex_message_close(self) -> None: ...

    def reasoning_started(self) -> None: ...

    def reasoning_delta(self, delta: str) -> None: ...

    def reasoning_completed(self) -> None: ...

    def command_started(self, command: str) -> None: ...

    def command_output(self, delta: str) -> None: ...

    def command_completed(self, exit_code: int | None) -> None: ...

    def tool_called(self, server: str, tool: str) -> None: ...

    def tool_completed(self, status: str) -> None: ...

    def token_usage(self, total_tokens: int) -> None: ...

    def error(self, message: str) -> None: ...

    def end_codex(self) -> None: ...


class CodexPresentation(CodexStreamSink, SessionStatusSink, Protocol):
    """What a Codex conversation needs: the stream plus its settings display."""


class TranscriptPresentation(
    TranscriptSink, CodexPresentation, ApplicationListSink, NewSessionSink, Protocol
):
    """The whole display surface, as the Textual interface provides it."""
