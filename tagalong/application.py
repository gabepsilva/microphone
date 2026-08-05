"""In-process adapter: live session collaborators as catalog action handlers.

Milestone 3 of issue #81 converts the first Textual slice onto the controller:
``message.send``, ``tts.set_enabled``, ``session.interrupt``, and
``session.new``. Milestone 6 adds the settings and audio actions so every
sidebar field the catalog names is kept true in ``AppState``. The controller
itself stays UI-neutral; this module is where the running session's
conversation, speech engine, gate, silence window, and capture channels
become those handlers.

The Textual interface still speaks through ``TuiHooks``. The hooks installed
here dispatch, so journey tests that drive the interface without a controller
stay valid oracles, and the live path has one execution model.

``session.new`` starts a Codex thread, which is slow work. The handler only
records that a reset was accepted. The caller opens the thread after dispatch
returns, claims the slot if it still holds it, adopts the thread, and only
then announces ``action.applied``. A superseded start is discarded rather
than overwriting the newer session, and a subscriber reacting to the applied
event sees the new thread already live.

Microphone and far-end selection use the same accepted-then-settle shape.
The handler records desired state, wakes the existing reconciler, and folds
the effective name in only if that request still holds the slot. A picker
keystroke that overtook an in-flight open is superseded rather than applied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
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
    Selection,
    with_desired,
    with_effective,
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


class SettingsConversationPort(Protocol):
    """Model and reasoning changes queued for the next Codex turn."""

    def request_model(self, model: str) -> bool | None: ...

    def request_reasoning_effort(self, effort: str) -> bool | None: ...


class SessionReset(Protocol):
    """The conversation methods a new-session run needs."""

    def start_fresh_thread(self) -> object | None: ...

    def adopt_fresh_thread(self, started: object) -> None: ...


class SpeechPort(Protocol):
    """The speech engine surface the first slice needs."""

    def set_enabled(self, enabled: bool) -> bool | None: ...


class SettingsSpeechPort(Protocol):
    """The speech engine surface a provider switch needs."""

    def set_provider(self, provider: str) -> bool | None: ...


class PolicyPort(Protocol):
    """The response-policy gate the sidebar picker drives."""

    def set_policy(self, policy: str) -> None: ...


class SilencePort(Protocol):
    """The shared turn-silence window."""

    def set(self, seconds: float) -> float: ...


class CapturePort(Protocol):
    """A microphone or far-end reconciler the audio actions drive."""

    def select(
        self,
        name: str | None,
        *,
        on_applied: Callable[[str | None], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool: ...

    def set_muted(self, muted: bool) -> None: ...


class TranscriptDisplayPort(Protocol):
    """The visible transcript surface a new session has to clear."""

    def reset_transcript(self) -> None: ...


class RecorderPort(Protocol):
    """The session-file recorder a new session has to roll."""

    def roll(self) -> None: ...


class FirstSliceHost(Protocol):
    """Anything that exposes the hook bag this slice overwrites."""

    hooks: Any


class ChannelMuteView(Protocol):
    @property
    def muted(self) -> bool: ...


class SessionView(Protocol):
    """The live session fields milestone 6 keeps true in ``AppState``."""

    microphone: str | None
    audio_stream: str | None
    policy: str
    tts_enabled: bool
    tts_provider: str
    codex_model: str
    codex_effort: str
    turn_silence: float

    @property
    def mic(self) -> ChannelMuteView: ...

    @property
    def audio(self) -> ChannelMuteView: ...


def app_state_from_session(state: SessionView) -> AppState:
    """Seed every field a registered handler will keep true.

    Desired selections come from the session; effective ones stay empty until
    a reconciler reports what actually opened. Mute, policy, speech, model,
    and silence are synchronous, so the session value is already effective.
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


type Persist = Callable[[str, object], object] | None


def _record(persist: Persist, key: str, value: object) -> None:
    if persist is not None:
        persist(key, value)


def _set_response_policy(
    gate: PolicyPort, persist: Persist, request: Request, state: AppState
) -> Effect:
    policy = str(request.payload["policy"])
    gate.set_policy(policy)
    _record(persist, "taga_after", policy)
    return Effect.applied(replace(state, response_policy=policy), policy)


def _set_tts_provider(
    tts: SettingsSpeechPort, persist: Persist, request: Request, state: AppState
) -> Effect:
    provider = str(request.payload["provider"])
    if tts.set_provider(provider) is False:
        raise EffectFailed("tts provider could not be changed")
    _record(persist, "tts_provider", provider)
    return Effect.applied(replace(state, tts_provider=provider), provider)


def _set_codex_model(
    conversation: SettingsConversationPort,
    persist: Persist,
    request: Request,
    state: AppState,
) -> Effect:
    model = str(request.payload["model"])
    if conversation.request_model(model) is False:
        raise EffectFailed("codex model could not be changed")
    _record(persist, "codex_model", model)
    return Effect.applied(replace(state, codex_model=model), model)


def _set_codex_reasoning(
    conversation: SettingsConversationPort,
    persist: Persist,
    request: Request,
    state: AppState,
) -> Effect:
    effort = str(request.payload["effort"])
    if conversation.request_reasoning_effort(effort) is False:
        raise EffectFailed("codex reasoning could not be changed")
    _record(persist, "codex_reasoning", effort)
    return Effect.applied(replace(state, codex_reasoning=effort), effort)


def _set_turn_silence(
    turn_silence: SilencePort, persist: Persist, request: Request, state: AppState
) -> Effect:
    applied = turn_silence.set(cast(float, request.payload["seconds"]))
    _record(persist, "turn_silence", applied)
    return Effect.applied(replace(state, turn_silence=applied), applied)


def bind_settings_slice(
    controller: Controller,
    collaborators: tuple[
        SettingsConversationPort, SettingsSpeechPort, PolicyPort, SilencePort
    ],
    persist: Persist = None,
) -> None:
    """Register the synchronous settings handlers on *controller*.

    Persistence belongs to the action, not to a TUI hook wrapper. A socket
    caller that sets the model must leave the same startup file a sidebar
    pick would, or the next session starts somewhere only one client went.
    """
    conversation, tts, gate, turn_silence = collaborators
    controller.register(
        "response_policy.set", partial(_set_response_policy, gate, persist)
    )
    controller.register("tts.set_provider", partial(_set_tts_provider, tts, persist))
    controller.register(
        "codex.set_model", partial(_set_codex_model, conversation, persist)
    )
    controller.register(
        "codex.set_reasoning", partial(_set_codex_reasoning, conversation, persist)
    )
    controller.register(
        "turn_silence.set", partial(_set_turn_silence, turn_silence, persist)
    )


def bind_audio_slice(
    controller: Controller,
    *,
    microphone: CapturePort,
    audio: CapturePort,
    on_microphone_applied: Callable[[str | None], object] | None = None,
    on_audio_applied: Callable[[str | None], object] | None = None,
) -> None:
    """Register selection and mute handlers that drive the live reconcilers.

    Select records desired state and wakes the channel. The reconciler reports
    back through ``settle`` or ``fail``; a newer select for the same action
    supersedes the in-flight one the way ``run_new_session`` discards a lost
    reset. Mute is desired state that a later open replays, so it applies even
    when no capture channel is open yet.
    """

    def select_microphone(request: Request, state: AppState) -> Effect:
        name = cast(str | None, request.payload["name"])
        microphone.select(
            name,
            on_applied=_completer(controller, request.id, on_microphone_applied),
        )
        return Effect.pending(
            with_desired(state, "microphone", name),
            settle=lambda current, effective: with_effective(
                current, "microphone", _optional_name(effective)
            ),
        )

    def select_audio(request: Request, state: AppState) -> Effect:
        name = cast(str | None, request.payload["name"])
        audio.select(
            name,
            on_applied=_completer(controller, request.id, on_audio_applied),
            on_failed=lambda detail: controller.fail(request.id, detail),
        )
        return Effect.pending(
            with_desired(state, "audio_stream", name),
            settle=lambda current, effective: with_effective(
                current, "audio_stream", _optional_name(effective)
            ),
        )

    def set_microphone_muted(request: Request, state: AppState) -> Effect:
        muted = bool(request.payload["muted"])
        microphone.set_muted(muted)
        return Effect.applied(replace(state, microphone_muted=muted), muted)

    def set_audio_muted(request: Request, state: AppState) -> Effect:
        muted = bool(request.payload["muted"])
        audio.set_muted(muted)
        return Effect.applied(replace(state, audio_stream_muted=muted), muted)

    controller.register("microphone.select", select_microphone)
    controller.register("audio_stream.select", select_audio)
    controller.register("microphone.set_muted", set_microphone_muted)
    controller.register("audio_stream.set_muted", set_audio_muted)


def _completer(
    controller: Controller,
    request_id: str,
    on_applied: Callable[[str | None], object] | None,
) -> Callable[[str | None], object]:
    def complete(effective: str | None) -> object:
        settled = controller.settle(request_id, effective)
        if isinstance(settled, Applied) and on_applied is not None:
            on_applied(effective)
        return settled

    return complete


def _optional_name(effective: object) -> str | None:
    return None if effective is None else str(effective)


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


def install_settings_hooks(
    tui: FirstSliceHost, controller: Controller, actor: Actor
) -> None:
    """Point the sidebar's settings hooks at *controller*."""

    def on_policy(policy: str) -> bool:
        outcome = controller.dispatch(
            "response_policy.set", {"policy": policy}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_codex_model(model: str) -> bool:
        outcome = controller.dispatch("codex.set_model", {"model": model}, actor=actor)
        return not isinstance(outcome, Rejected | Failed)

    def on_codex_effort(effort: str) -> bool:
        outcome = controller.dispatch(
            "codex.set_reasoning", {"effort": effort}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_tts_provider(provider: str) -> bool:
        outcome = controller.dispatch(
            "tts.set_provider", {"provider": provider}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_turn_silence(seconds: float) -> float | None:
        outcome = controller.dispatch(
            "turn_silence.set", {"seconds": seconds}, actor=actor
        )
        if isinstance(outcome, Applied):
            return float(cast(float, outcome.effective))
        return None

    tui.hooks.on_policy = on_policy
    tui.hooks.on_codex_model = on_codex_model
    tui.hooks.on_codex_effort = on_codex_effort
    tui.hooks.on_tts_provider = on_tts_provider
    tui.hooks.on_turn_silence = on_turn_silence


def install_audio_hooks(
    tui: FirstSliceHost, controller: Controller, actor: Actor
) -> None:
    """Point microphone and far-end hooks at *controller*."""

    def on_microphone(name: str | None) -> bool:
        outcome = controller.dispatch("microphone.select", {"name": name}, actor=actor)
        return not isinstance(outcome, Rejected | Failed)

    def on_audio_stream(name: str | None) -> bool:
        outcome = controller.dispatch(
            "audio_stream.select", {"name": name}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_mute(muted: bool) -> bool:
        outcome = controller.dispatch(
            "microphone.set_muted", {"muted": muted}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    def on_audio_mute(muted: bool) -> bool:
        outcome = controller.dispatch(
            "audio_stream.set_muted", {"muted": muted}, actor=actor
        )
        return not isinstance(outcome, Rejected | Failed)

    tui.hooks.on_microphone = on_microphone
    tui.hooks.on_audio_stream = on_audio_stream
    tui.hooks.on_mute = on_mute
    tui.hooks.on_audio_mute = on_audio_mute


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
    Transcript and recorder move only for the winner. ``action.applied`` is
    announced in ``finally`` so a failed roll cannot leave a claimed request
    without a terminal event.
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
    try:
        conversation.adopt_fresh_thread(started)
        display.reset_transcript()
        recorder.roll()
    finally:
        controller.announce(outcome.request_id)
    return settled
