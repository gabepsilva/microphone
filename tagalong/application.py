"""In-process adapter: live session collaborators as catalog action handlers.

Milestone 3 of issue #81 converts the first Textual slice onto the controller:
``message.send``, ``tts.set_enabled``, ``session.interrupt``, and
``session.new``. Milestone 6 adds the settings and audio actions so every
sidebar field the catalog names is kept true in ``AppState``. Milestone 7
wires session and transcript actions — ``voice.end_turn``, ``transcript.save``,
``transcript.append``, ``attachment.upload``, and ``session.quit``. The
controller itself stays UI-neutral; this module is where the running session's
collaborators become those handlers.

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

import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from .attachments import read_primary_selection
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
from .domain import (
    AGENT,
    TEXT,
    SentenceChunker,
    UserTextMessage,
    markdown_to_speech,
    strip_chrome,
)
from .presentation import Entry
from .recording import default_transcript_dir, write_transcript_export
from .speech import PIPER


class ConversationPort(Protocol):
    """The Codex conversation surface this slice needs."""

    generation: int

    def ingest(
        self,
        speaker: str,
        text: str,
        respond: bool,
        timestamp: str | None = None,
        images: tuple[str, ...] = (),
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


class ReadAloudSpeechPort(Protocol):
    """Speak cleaned selection text; shares the session TurnGate with Codex."""

    def begin_turn(self) -> None: ...

    def speak(self, text: str) -> None: ...


class SettingsSpeechPort(Protocol):
    """The speech engine surface a provider or voice switch needs."""

    def set_provider(
        self,
        provider: str,
        voice: str | None = None,
        *,
        on_applied: Callable[[str], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool | None: ...

    def set_voice(
        self,
        voice: str,
        *,
        on_applied: Callable[[str], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool | None: ...


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


class MessageDisplayPort(Protocol):
    """Draw a message that no local interface drew for itself.

    A typist at the TUI sees their line the moment they press enter, because
    the prompt puts it on screen before the action is dispatched. A socket
    peer has no such prompt: unless the handler draws it, the message reaches
    Codex as context and reaches the transcript nowhere, so both clients
    answer a question neither of them shows being asked.

    ``show_message`` runs inside a handler, so it runs under the controller's
    writer lock — and in a TUI session that lock is the transcript store's
    lock, because ``attach_conversation_hooks`` hands the store to the
    controller. An implementation must therefore **return without waiting on
    another thread**: a display that blocks until, say, a UI thread has drawn
    the row blocks until a thread that has to take the lock its caller is
    still holding. Schedule the draw and return.
    """

    def show_message(self, speaker: str, text: str) -> None: ...


class RecorderPort(Protocol):
    """The session-file recorder a new session has to roll."""

    def roll(self) -> None: ...


class TurnPort(Protocol):
    """Flush a spoken turn without waiting for the silence window."""

    def end_turn(self) -> None: ...


class TranscriptEntriesPort(Protocol):
    """The live transcript rows a save export reads."""

    def transcript_entries(self) -> Sequence[Entry]: ...


class AttachmentPort(Protocol):
    """Opaque attachment ids over validated image bytes."""

    def upload(self, data: bytes) -> str: ...

    def resolve(self, ids: Sequence[str]) -> tuple[str, ...]: ...


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
    tts_voice: str
    piper_voice: str
    edge_voice: str
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
    a reconciler reports what actually opened — except the active TTS engine
    and voice, which are already speaking when the session starts, so both
    sides of those Selections carry the startup values. Mute, policy, speech
    enable, model, and silence are synchronous, so the session value is
    already effective.
    """
    return AppState(
        microphone=Selection(desired=state.microphone),
        microphone_muted=state.mic.muted,
        audio_stream=Selection(desired=state.audio_stream),
        audio_stream_muted=state.audio.muted,
        response_policy=state.policy,
        tts_enabled=state.tts_enabled,
        tts_provider=Selection(
            desired=state.tts_provider, effective=state.tts_provider
        ),
        tts_voice=Selection(desired=state.tts_voice, effective=state.tts_voice),
        piper_voice=state.piper_voice,
        edge_voice=state.edge_voice,
        codex_model=state.codex_model,
        codex_reasoning=state.codex_effort,
        turn_silence=state.turn_silence,
    )


def _show_remote_message(
    display: MessageDisplayPort | None, request: Request, speaker: str, text: str
) -> None:
    """Put a remote actor's message on the transcript.

    Only remote ones: a human typing at the interface has already seen their
    line appear, and drawing it again here would double every typed message.
    """
    if display is not None and request.actor.kind is ActorKind.AGENT:
        display.show_message(speaker, text)


def _send_message(
    conversation: ConversationPort,
    attachments: AttachmentPort | None,
    display: MessageDisplayPort | None,
    request: Request,
    state: AppState,
) -> Effect:
    text = str(request.payload["text"])
    image_ids = cast(tuple[str, ...], request.payload["images"])
    respond = bool(request.payload["respond"])
    if image_ids:
        if attachments is None:
            raise EffectFailed("attachments are not available in this session")
        try:
            images = attachments.resolve(image_ids)
        except KeyError as error:
            raise EffectFailed(f"unknown attachment: {error.args[0]}") from error
    else:
        images = ()
    # Agents share message.send with humans; provenance is the speaker label.
    speaker = AGENT if request.actor.kind is ActorKind.AGENT else TEXT
    # Drawn before ingest so the transcript reads in the order Codex was told.
    _show_remote_message(display, request, speaker, text)
    conversation.ingest(speaker, text, respond=respond, images=images)
    return Effect.applied(state, image_ids)


def bind_first_slice(
    controller: Controller,
    *,
    conversation: ConversationPort,
    tts: SpeechPort,
    attachments: AttachmentPort | None = None,
    display: MessageDisplayPort | None = None,
) -> None:
    """Register the four first-slice handlers on *controller*."""

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

    controller.register(
        "message.send", partial(_send_message, conversation, attachments, display)
    )
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
    tts: SettingsSpeechPort,
    persist: Persist,
    controller: Controller,
    request: Request,
    state: AppState,
) -> Effect:
    provider = str(request.payload["provider"])
    voice = state.piper_voice if provider == PIPER else state.edge_voice

    def settle(current: AppState, effective: object) -> AppState:
        applied_voice = str(effective)
        return replace(
            current,
            tts_provider=Selection(desired=provider, effective=provider),
            tts_voice=Selection(desired=voice, effective=applied_voice),
        )

    def on_applied(effective: str) -> object:
        settled = controller.settle(request.id, effective)
        if isinstance(settled, Applied):
            _record(persist, "tts_provider", provider)
        return settled

    if (
        tts.set_provider(
            provider,
            voice,
            on_applied=on_applied,
            on_failed=lambda detail: controller.fail(request.id, detail),
        )
        is False
    ):
        raise EffectFailed("tts provider could not be changed")
    pending = with_desired(state, "tts_provider", provider)
    return Effect.pending(with_desired(pending, "tts_voice", voice), settle=settle)


def _set_tts_voice(
    tts: SettingsSpeechPort,
    persist: Persist,
    controller: Controller,
    request: Request,
    state: AppState,
) -> Effect:
    voice = str(request.payload["voice"])
    running = state.tts_provider.effective or state.tts_provider.desired
    remember = "piper_voice" if running == PIPER else "edge_voice"

    def settle(current: AppState, effective: object) -> AppState:
        applied = str(effective)
        updated = with_effective(current, "tts_voice", applied)
        return replace(updated, **{remember: applied})

    def on_applied(effective: str) -> object:
        settled = controller.settle(request.id, effective)
        if isinstance(settled, Applied):
            _record(persist, remember, effective)
        return settled

    if (
        tts.set_voice(
            voice,
            on_applied=on_applied,
            on_failed=lambda detail: controller.fail(request.id, detail),
        )
        is False
    ):
        raise EffectFailed("tts voice could not be changed")
    return Effect.pending(with_desired(state, "tts_voice", voice), settle=settle)


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
    """Register the settings handlers on *controller*.

    Persistence belongs to the action, not to a TUI hook wrapper. A socket
    caller that sets the model must leave the same startup file a sidebar
    pick would, or the next session starts somewhere only one client went.
    Provider and voice switches settle asynchronously and persist only on
    success, the same shape as microphone selection (#124 D14).
    """
    conversation, tts, gate, turn_silence = collaborators
    controller.register(
        "response_policy.set", partial(_set_response_policy, gate, persist)
    )
    controller.register(
        "tts.set_provider", partial(_set_tts_provider, tts, persist, controller)
    )
    controller.register(
        "tts.set_voice",
        partial(_set_tts_voice, tts, persist, controller),
    )
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


def bind_read_aloud_slice(
    controller: Controller,
    *,
    tts: ReadAloudSpeechPort,
    read_selection: Callable[[], str | None] = read_primary_selection,
) -> None:
    """Register ``speech.read_selection`` — primary selection → chrome → TTS.

    The handler must not block: ``wl-paste`` is ``subprocess.run(..., timeout=2)``
    and the controller lock is also the transcript lock. Shape mirrors
    ``select_audio`` — validate, return ``Effect.pending``, worker settles/fails.

    Collision with a Codex reply is last-wins: one :class:`~.domain.TurnGate`
    on the shared TTS engine; ``begin_turn`` orphans the other side's queue.
    Spoken text enters ``EchoMemory`` via ``tts.speak`` so the mic does not
    re-transcribe the machine reading aloud.
    """

    def read_aloud(request: Request, state: AppState) -> Effect:
        if not state.tts_enabled:
            raise EffectFailed("TTS is disabled")

        def work() -> None:
            try:
                selection = read_selection()
                if selection is None or not selection.strip():
                    controller.fail(
                        request.id,
                        "Primary selection is empty or no selection helper is available",
                    )
                    return
                cleaned = markdown_to_speech(strip_chrome(selection))
                if not cleaned:
                    controller.fail(
                        request.id,
                        "Nothing speakable remained after cleaning the selection",
                    )
                    return
                # Last-wins vs Codex: begin_turn bumps the shared TurnGate.
                tts.begin_turn()
                chunker = SentenceChunker(tts.speak)
                chunker.feed(cleaned)
                chunker.flush()
                # No effective payload — settle would publish it on action.applied
                # into the shared EventLog (#128b S3); nobody reads it.
                controller.settle(request.id)
            except Exception as error:
                controller.fail(request.id, str(error))

        threading.Thread(target=work, name="speech.read_selection", daemon=True).start()
        return Effect.pending(state, settle=lambda current, _effective: current)

    controller.register("speech.read_selection", read_aloud)


def bind_session_transcript_slice(
    controller: Controller,
    collaborators: tuple[
        ConversationPort, TurnPort, AttachmentPort, TranscriptEntriesPort
    ],
    directory: Path | None = None,
    display: MessageDisplayPort | None = None,
) -> None:
    """Register session and transcript handlers on *controller*.

    ``attachment.upload`` validates bytes and returns opaque ids.
    ``message.send`` resolves those ids; callers never pass filesystem paths.
    ``session.quit`` is refused for agents — capability policy owns that denial.
    """
    conversation, turn, attachments, transcript = collaborators
    export_dir = directory if directory is not None else default_transcript_dir()

    def upload_attachment(request: Request, state: AppState) -> Effect:
        data = cast(bytes, request.payload["data"])
        try:
            attachment_id = attachments.upload(data)
        except ValueError as error:
            raise EffectFailed(str(error)) from error
        return Effect.applied(state, attachment_id)

    def append_transcript(request: Request, state: AppState) -> Effect:
        text = str(request.payload["text"])
        speaker = AGENT if request.actor.kind is ActorKind.AGENT else TEXT
        _show_remote_message(display, request, speaker, text)
        conversation.ingest(speaker, text, respond=False)
        return Effect.applied(state, speaker)

    def save_transcript(_request: Request, state: AppState) -> Effect:
        entries = list(transcript.transcript_entries())
        try:
            path = write_transcript_export(entries, export_dir)
        except OSError as error:
            raise EffectFailed(str(error)) from error
        # Name only — absolute paths would sit oddly beside opaque attachment ids.
        return Effect.applied(state, path.name)

    def end_voice_turn(_request: Request, state: AppState) -> Effect:
        turn.end_turn()
        return Effect.applied(state, None)

    def quit_session(_request: Request, state: AppState) -> Effect:
        # Agents are refused by capability policy before this handler runs.
        return Effect.applied(state, None)

    controller.register("attachment.upload", upload_attachment)
    controller.register("transcript.append", append_transcript)
    controller.register("transcript.save", save_transcript)
    controller.register("voice.end_turn", end_voice_turn)
    controller.register("session.quit", quit_session)


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


def install_session_transcript_hooks(
    tui: FirstSliceHost,
    controller: Controller,
    actor: Actor,
    *,
    on_quit_cleanup: Callable[[], object] | None = None,
) -> None:
    """Point end-turn, save, upload, and quit hooks at *controller*."""

    def on_end_turn() -> None:
        controller.dispatch("voice.end_turn", actor=actor)

    def on_save(_entries: object) -> None:
        controller.dispatch("transcript.save", actor=actor)

    def on_attachment_upload(data: bytes) -> str | None:
        outcome = controller.dispatch("attachment.upload", {"data": data}, actor=actor)
        if isinstance(outcome, Applied):
            return str(outcome.effective)
        return None

    def on_quit() -> bool:
        outcome = controller.dispatch("session.quit", actor=actor)
        if not isinstance(outcome, Applied):
            return False
        if on_quit_cleanup is not None:
            on_quit_cleanup()
        return True

    tui.hooks.on_end_turn = on_end_turn
    tui.hooks.on_save = on_save
    tui.hooks.on_attachment_upload = on_attachment_upload
    tui.hooks.on_quit = on_quit


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
