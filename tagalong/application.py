"""In-process adapter: live session collaborators as catalog action handlers.

Milestone 3 of issue #81 converts the first Textual slice onto the controller:
``message.send``, ``tts.set_enabled``, ``session.interrupt``, and
``session.new``. The controller itself stays UI-neutral; this module is where
the running session's conversation, speech engine, display, and recorder
become the handlers those four actions run.

The Textual interface still speaks through ``TuiHooks``. The hooks installed
here dispatch, so journey tests that drive the interface without a controller
stay valid oracles, and the live path has one execution model.

``session.new`` starts a Codex thread, which is slow work. The handler only
records that a reset was accepted; the caller runs the thread start after
dispatch returns, then settles or fails the request. That is what keeps mute
and interrupt responsive while a new session is opening.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, cast

from .control import (
    Accepted,
    Actor,
    ActorKind,
    AppState,
    Controller,
    Effect,
    EffectFailed,
    Failed,
    Inapplicable,
    Outcome,
    Rejected,
    Request,
    Selection,
)
from .domain import TEXT, UserTextMessage


class ConversationPort(Protocol):
    """The Codex conversation surface this slice needs."""

    generation: int

    def ingest(
        self, speaker: str, text: str, respond: bool, images: tuple[str, ...] = ()
    ) -> None: ...

    def interrupt(self) -> None: ...

    def new_session(self) -> bool: ...


class SessionReset(Protocol):
    """The only conversation method a new-session run needs."""

    def new_session(self) -> bool: ...


class SpeechPort(Protocol):
    """The speech engine surface this slice needs."""

    def set_enabled(self, enabled: bool) -> bool | None: ...


class TranscriptDisplayPort(Protocol):
    """The visible transcript surface a new session has to clear."""

    def reset_transcript(self) -> None: ...


class RecorderPort(Protocol):
    """The session-file recorder a new session has to roll."""

    def roll(self) -> None: ...


class ChannelView(Protocol):
    @property
    def muted(self) -> bool: ...


class SessionView(Protocol):
    """The subset of sidebar state this slice seeds the controller from."""

    microphone: str | None
    audio_stream: str | None
    policy: str
    tts_enabled: bool
    tts_provider: str
    codex_model: str
    codex_effort: str
    turn_silence: float

    @property
    def mic(self) -> ChannelView: ...

    @property
    def audio(self) -> ChannelView: ...


def app_state_from_session(state: SessionView) -> AppState:
    """Seed canonical state from the session the interface already holds.

    Desired device names come from startup; effective ones stay unset until a
    reconciler reports they opened. Mute and the rest of the slice are
    synchronous flags copied as they stand.
    """
    return AppState(
        microphone=Selection(desired=state.microphone),
        microphone_muted=state.mic.muted,
        audio_stream=Selection(desired=state.audio_stream),
        audio_stream_muted=state.audio.muted,
        response_policy=state.policy,
        tts_enabled=state.tts_enabled,
        tts_provider=state.tts_provider,
        codex_model=state.codex_model,
        codex_reasoning=state.codex_effort,
        turn_silence=state.turn_silence,
    )


def bind_first_slice(
    controller: Controller,
    *,
    conversation: ConversationPort,
    tts: SpeechPort,
) -> None:
    """Register the four first-slice handlers on *controller*."""

    def send_message(request: Request, state: AppState) -> Effect:
        if request.actor.kind is not ActorKind.HUMAN:
            raise Inapplicable("agent messages are not enabled in this session")
        text = str(request.payload["text"])
        images = cast(tuple[str, ...], request.payload["images"])
        respond = bool(request.payload["respond"])
        conversation.ingest(TEXT, text, respond=respond, images=images)
        return Effect.applied(state, images)

    def set_tts_enabled(request: Request, state: AppState) -> Effect:
        enabled = bool(request.payload["enabled"])
        if tts.set_enabled(enabled) is False:
            raise EffectFailed("tts could not be changed")
        return Effect.applied(replace(state, tts_enabled=enabled), enabled)

    def interrupt_session(request: Request, state: AppState) -> Effect:
        requested = request.payload["generation"]
        current = conversation.generation
        if requested is not None and requested != current:
            raise Inapplicable(
                f"generation {requested} is no longer current (now {current})"
            )
        conversation.interrupt()
        return Effect.applied(state, current)

    def start_session(_request: Request, state: AppState) -> Effect:
        return Effect.pending(state, settle=lambda current, _effective: current)

    controller.register("message.send", send_message)
    controller.register("tts.set_enabled", set_tts_enabled)
    controller.register("session.interrupt", interrupt_session)
    controller.register("session.new", start_session)


def install_first_slice_hooks(tui: Any, controller: Controller, actor: Actor) -> None:
    """Point the interface's first-slice hooks at *controller*."""

    def on_user_text(message: UserTextMessage) -> None:
        controller.dispatch(
            "message.send",
            {"text": message.text, "images": message.images},
            actor=actor,
        )

    def on_tts(enabled: bool) -> bool:
        outcome = controller.dispatch(
            "tts.set_enabled", {"enabled": enabled}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_interrupt() -> None:
        controller.dispatch("session.interrupt", actor=actor)

    tui.hooks.on_user_text = on_user_text
    tui.hooks.on_tts = on_tts
    tui.hooks.on_interrupt = on_interrupt


def apply_new_session(
    conversation: SessionReset,
    display: TranscriptDisplayPort,
    recorder: RecorderPort,
) -> bool:
    """Start a fresh Codex thread and roll the transcript if it took.

    The visible transcript and the session file move together, and only after
    the new thread exists — a failed start must leave both where they were.
    """
    if not conversation.new_session():
        return False
    display.reset_transcript()
    recorder.roll()
    return True


def run_new_session(
    controller: Controller,
    actor: Actor,
    conversation: SessionReset,
    display: TranscriptDisplayPort,
    recorder: RecorderPort,
) -> Outcome:
    """Dispatch ``session.new`` and reconcile the slow start outside the lock."""
    outcome = controller.dispatch("session.new", actor=actor)
    if not isinstance(outcome, Accepted):
        return outcome
    if apply_new_session(conversation, display, recorder):
        return controller.settle(outcome.request_id)
    return controller.fail(outcome.request_id, "could not start a new session")
