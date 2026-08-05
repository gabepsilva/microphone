"""In-process adapter: live session collaborators as catalog action handlers.

Milestone 3 of issue #81 converts the first Textual slice onto the controller:
``message.send``, ``tts.set_enabled``, ``session.interrupt``, and
``session.new``. The controller itself stays UI-neutral; this module is where
the running session's conversation, speech engine, display, and recorder
become the handlers those four actions run.

The agreed slice is those four actions, not every sidebar transition. Canonical
state therefore carries only what a registered handler keeps true —
``tts_enabled``. Seeding the rest from startup would make ``snapshot()`` report
stale sidebar values as current. Those fields arrive with their actions later.

The Textual interface still speaks through ``TuiHooks``. The hooks installed
here dispatch, so journey tests that drive the interface without a controller
stay valid oracles, and the live path has one execution model.

``session.new`` starts a Codex thread, which is slow work. The handler only
records that a reset was accepted. The caller opens the thread after dispatch
returns, claims the slot if it still holds it, adopts the thread, and only
then announces ``action.applied``. A superseded start is discarded rather
than overwriting the newer session, and a subscriber reacting to the applied
event sees the new thread already live.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, cast

from .control import (
    Accepted,
    Actor,
    ActorKind,
    Applied,
    AppState,
    Controller,
    Effect,
    EffectFailed,
    Failed,
    Inapplicable,
    Outcome,
    Rejected,
    Request,
)
from .domain import TEXT, UserTextMessage


class ConversationPort(Protocol):
    """The Codex conversation surface this slice needs."""

    generation: int

    def ingest(
        self, speaker: str, text: str, respond: bool, images: tuple[str, ...] = ()
    ) -> None: ...

    def interrupt(self) -> None: ...

    def start_fresh_thread(self) -> object | None: ...

    def adopt_fresh_thread(self, started: object) -> None: ...


class SessionReset(Protocol):
    """The conversation methods a new-session run needs."""

    def start_fresh_thread(self) -> object | None: ...

    def adopt_fresh_thread(self, started: object) -> None: ...


class SpeechPort(Protocol):
    """The speech engine surface this slice needs."""

    def set_enabled(self, enabled: bool) -> bool | None: ...


class TranscriptDisplayPort(Protocol):
    """The visible transcript surface a new session has to clear."""

    def reset_transcript(self) -> None: ...


class RecorderPort(Protocol):
    """The session-file recorder a new session has to roll."""

    def roll(self) -> None: ...


class FirstSliceHost(Protocol):
    """Anything that exposes the hook bag this slice overwrites."""

    hooks: Any


class TtsView(Protocol):
    tts_enabled: bool


def app_state_from_session(state: TtsView) -> AppState:
    """Seed only the field this slice will keep true.

    ``tts.set_enabled`` is the one registered handler that writes ``AppState``.
    Copying microphone, model, mute, or silence here would snapshot startup
    values that later sidebar hooks change without telling the controller.
    """
    return AppState(tts_enabled=state.tts_enabled)


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


def install_first_slice_hooks(
    tui: FirstSliceHost, controller: Controller, actor: Actor
) -> None:
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


def run_new_session(
    controller: Controller,
    actor: Actor,
    conversation: SessionReset,
    display: TranscriptDisplayPort,
    recorder: RecorderPort,
) -> Outcome:
    """Dispatch ``session.new`` and adopt the thread only if it still wins.

    The Codex open happens outside the writer lock. After it finishes, claim
    decides whether this request still holds the slot without publishing.
    Transcript, recorder, and ``action.applied`` move only after the winner
    is installed, so a superseded start cannot clear a newer session and a
    subscriber does not observe the previous thread as the applied one.
    """
    outcome = controller.dispatch("session.new", actor=actor)
    if not isinstance(outcome, Accepted):
        return outcome
    started = conversation.start_fresh_thread()
    if started is None:
        return controller.fail(outcome.request_id, "could not start a new session")
    settled = controller.claim(outcome.request_id)
    if not isinstance(settled, Applied):
        return settled
    conversation.adopt_fresh_thread(started)
    display.reset_transcript()
    recorder.roll()
    controller.announce(outcome.request_id)
    return settled
