"""In-process catalog handlers: first slice, settings, and audio."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from tagalong.application import (
    app_state_from_session,
    bind_audio_slice,
    bind_first_slice,
    bind_read_aloud_slice,
    bind_settings_slice,
    install_audio_hooks,
    install_first_slice_hooks,
    install_settings_hooks,
    settle_new_session,
)
from tagalong.control import (
    Accepted,
    Applied,
    AppState,
    Controller,
    Failed,
    Rejected,
    Rejection,
    Selection,
    Superseded,
    agent,
    local_user,
)
from tagalong.control.actions import Scope
from tagalong.control.policy import (
    SOCKET_AGENT_SCOPES,
    denied_actions_for_socket_client,
)
from tagalong.domain import EchoMemory, TurnGate, UserTextMessage
from tagalong.tui import SessionState, TuiHooks


class FakeConversation:
    def __init__(self, *, generation: int = 0, new_session_ok: bool = True) -> None:
        self.generation = generation
        self.new_session_ok = new_session_ok
        self.ingested: list[tuple] = []
        self.interrupts = 0
        self.sessions = 0
        self.started: list[object] = []
        self.adopted: list[object] = []
        self.thread: object | None = None
        self.models: list[str] = []
        self.efforts: list[str] = []
        self.model_ok = True
        self.effort_ok = True

    def ingest(self, speaker, text, respond, timestamp=None, images=()):
        del timestamp
        self.ingested.append((speaker, text, respond, images))

    def interrupt(self) -> None:
        self.interrupts += 1

    def start_fresh_thread(self):
        if not self.new_session_ok:
            return None
        started = f"thread-{len(self.started) + 1}"
        self.started.append(started)
        return started

    def adopt_fresh_thread(self, started) -> None:
        self.generation += 1
        self.sessions += 1
        self.adopted.append(started)
        self.thread = started

    def new_session(self) -> bool:
        started = self.start_fresh_thread()
        if started is None:
            return False
        self.adopt_fresh_thread(started)
        return True

    def request_model(self, model: str) -> bool:
        self.models.append(model)
        return self.model_ok

    def request_reasoning_effort(self, effort: str) -> bool:
        self.efforts.append(effort)
        return self.effort_ok


class FakeSpeech:
    def __init__(self, *, accept: bool = True, provider: str = "piper") -> None:
        self.enabled = True
        self.provider = provider
        self.voice = (
            "en_US-lessac-medium" if provider == "piper" else "en-US-AndrewNeural"
        )
        self.accept = accept
        self.calls: list[bool] = []
        self.providers: list[str] = []
        self.voices: list[str] = []
        self.provider_ok = True
        self.voice_ok = True
        self.voice_fail: str | None = None
        self._pending_provider: str | None = None
        self._pending_voice: str | None = None
        self._provider_applied: Callable[[str], object] | None = None
        self._provider_failed: Callable[[str], object] | None = None
        self._voice_applied: Callable[[str], object] | None = None
        self._voice_failed: Callable[[str], object] | None = None

    def set_enabled(self, enabled: bool) -> bool | None:
        self.calls.append(enabled)
        if not self.accept:
            return False
        self.enabled = enabled
        return True

    def set_provider(
        self,
        provider: str,
        voice: str | None = None,
        *,
        on_applied: Callable[[str], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool | None:
        self.providers.append(provider)
        if not self.provider_ok or provider == self.provider:
            return False
        self._pending_provider = provider
        self._pending_voice = self.voice if voice is None else voice
        self._provider_applied = on_applied
        self._provider_failed = on_failed
        return True

    def complete_provider(self) -> None:
        provider = self._pending_provider
        voice = self._pending_voice
        assert provider is not None
        assert voice is not None
        self.provider = provider
        self.voice = voice
        if self._provider_applied is not None:
            self._provider_applied(voice)

    def set_voice(
        self,
        voice: str,
        *,
        on_applied: Callable[[str], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool | None:
        self.voices.append(voice)
        if not self.voice_ok or voice == self.voice:
            return False
        self._pending_voice = voice
        self._voice_applied = on_applied
        self._voice_failed = on_failed
        return True

    def complete_voice(self) -> None:
        """Settle a pending voice switch the way the real switch thread would."""
        voice = self._pending_voice
        assert voice is not None
        if self.voice_fail is not None:
            if self._voice_failed is not None:
                self._voice_failed(self.voice_fail)
            return
        self.voice = voice
        if self._voice_applied is not None:
            self._voice_applied(voice)


class FakeDisplay:
    def __init__(self) -> None:
        self.resets = 0

    def reset_transcript(self) -> None:
        self.resets += 1


class FakeRecorder:
    def __init__(self) -> None:
        self.rolls = 0

    def roll(self) -> None:
        self.rolls += 1


class FakeTui:
    def __init__(self) -> None:
        self.hooks = TuiHooks()
        self.state = SessionState()


class FakeGate:
    def __init__(self) -> None:
        self.policies: list[str] = []

    def set_policy(self, policy: str) -> None:
        self.policies.append(policy)


class FakeSilence:
    def __init__(self, seconds: float = 3.0) -> None:
        self.seconds = seconds

    def set(self, seconds: float) -> float:
        self.seconds = max(0.25, min(30.0, seconds))
        return self.seconds


class FakeCapture:
    def __init__(self) -> None:
        self.selected: list[str | None] = []
        self.muted = False
        self._applied: list[Callable[[str | None], object] | None] = []
        self._failed: list[Callable[[str], object] | None] = []

    def select(
        self,
        name: str | None,
        *,
        on_applied: Callable[[str | None], object] | None = None,
        on_failed: Callable[[str], object] | None = None,
    ) -> bool:
        self.selected.append(name)
        self._applied.append(on_applied)
        self._failed.append(on_failed)
        return True

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def apply(self, name: str | None, index: int = -1) -> object:
        on_applied = self._applied[index]
        assert on_applied is not None
        return on_applied(name)

    def fail(self, detail: str, index: int = -1) -> object:
        on_failed = self._failed[index]
        assert on_failed is not None
        return on_failed(detail)


OWNER = local_user("tui")


class FakeMessageDisplay:
    """Records what a handler drew on the transcript, in order."""

    def __init__(self) -> None:
        self.shown: list[tuple[str, str]] = []

    def show_message(self, speaker: str, text: str) -> None:
        self.shown.append((speaker, text))


def bound(
    conversation: FakeConversation | None = None,
    tts: FakeSpeech | None = None,
    display: FakeMessageDisplay | None = None,
    transcript: FakeDisplay | None = None,
    recorder: FakeRecorder | None = None,
) -> tuple[Controller, FakeConversation, FakeSpeech]:
    conversation = FakeConversation() if conversation is None else conversation
    tts = FakeSpeech() if tts is None else tts
    controller = Controller()
    bind_first_slice(
        controller,
        conversation=conversation,
        tts=tts,
        display=display,
        transcript=transcript,
        recorder=recorder,
    )
    return controller, conversation, tts


def _bound_session_new(
    conversation: FakeConversation | None = None,
    *,
    transcript: FakeDisplay | None = None,
    recorder: FakeRecorder | None = None,
) -> tuple[Controller, FakeConversation, FakeDisplay, FakeRecorder]:
    display = FakeDisplay() if transcript is None else transcript
    rolls = FakeRecorder() if recorder is None else recorder
    conversation = FakeConversation() if conversation is None else conversation
    controller, conversation, _tts = bound(
        conversation=conversation, transcript=display, recorder=rolls
    )
    return controller, conversation, display, rolls


def test_app_state_seeds_every_field_this_slice_maintains() -> None:
    state = SessionState(
        microphone="Yeti",
        audio_stream="Zoom",
        policy="audio",
        tts_enabled=False,
        tts_provider="edge",
        codex_model="gpt-5.6-sol",
        codex_effort="high",
        codex_thread="thread-9",
        codex_state="thinking",
        codex_speaking=True,
        turn_silence=4.5,
        confidence=0.83,
        language="fr",
        moonshine="small-streaming",
        tokens=17,
        echoes_cut=2,
    )
    state.mic.muted = True
    state.audio.muted = True

    seeded = app_state_from_session(state)

    assert seeded == AppState(
        microphone=Selection(desired="Yeti"),
        microphone_muted=True,
        audio_stream=Selection(desired="Zoom"),
        audio_stream_muted=True,
        response_policy="audio",
        tts_enabled=False,
        tts_provider=Selection(desired="edge", effective="edge"),
        tts_voice=Selection(desired=state.tts_voice, effective=state.tts_voice),
        piper_voice=state.piper_voice,
        edge_voice=state.edge_voice,
        codex_model="gpt-5.6-sol",
        codex_reasoning="high",
        codex_thread="thread-9",
        codex_state="thinking",
        codex_speaking=True,
        turn_silence=4.5,
        confidence=0.83,
        language="fr",
        moonshine="small-streaming",
        tokens=17,
        echoes_cut=2,
    )


def test_a_human_message_is_ingested_as_text_and_applied() -> None:
    controller, conversation, _tts = bound()

    outcome = controller.dispatch(
        "message.send",
        {"text": "hello"},
        actor=OWNER,
    )

    assert outcome == Applied("req-1", ())
    assert conversation.ingested == [("Text", "hello", True, ())]


def test_an_agent_message_is_ingested_as_agent() -> None:
    controller, conversation, _tts = bound()
    caller = agent("notes-bot", {Scope.CONVERSE})

    outcome = controller.dispatch(
        "message.send",
        {"text": "context from a tool", "respond": False},
        actor=caller,
    )

    assert outcome == Applied("req-1", ())
    assert conversation.ingested == [("Agent", "context from a tool", False, ())]


def test_an_agent_message_is_drawn_on_the_transcript() -> None:
    # A socket client has no prompt of its own, so nothing has drawn the
    # message yet. Without this the text reaches Codex as context and reaches
    # the transcript nowhere: both clients answer a question neither shows.
    display = FakeMessageDisplay()
    controller, conversation, _tts = bound(display=display)
    caller = agent("electron", {Scope.CONVERSE})

    controller.dispatch("message.send", {"text": "hello from a client"}, actor=caller)

    assert display.shown == [("Agent", "hello from a client")]
    assert conversation.ingested == [("Agent", "hello from a client", True, ())]


def test_the_display_is_called_under_the_controllers_writer_lock() -> None:
    # Why MessageDisplayPort.show_message must not wait on another thread:
    # the handler runs under the lock, and in a TUI session that lock is the
    # transcript store's (cli.attach_conversation_hooks adopts tui.transcript).
    # A display that waited for a UI thread would wait for a thread that has
    # to take this lock to append the row.
    controller = Controller()
    held: list[bool] = []

    class LockProbe:
        def show_message(self, speaker: str, text: str) -> None:
            del speaker, text

            # Another thread, because the lock is reentrant for this one.
            def probe() -> None:
                held.append(not controller.transcript.lock.acquire(timeout=0.2))
                if not held[-1]:
                    controller.transcript.lock.release()

            worker = threading.Thread(target=probe)
            worker.start()
            worker.join(timeout=2)

    bind_first_slice(
        controller,
        conversation=FakeConversation(),
        tts=FakeSpeech(),
        display=LockProbe(),
    )

    controller.dispatch(
        "message.send", {"text": "hi"}, actor=agent("electron", {Scope.CONVERSE})
    )

    assert held == [True]


def test_a_human_message_is_not_drawn_twice() -> None:
    # The prompt puts the typed line on screen before dispatching.
    display = FakeMessageDisplay()
    controller, _conversation, _tts = bound(display=display)

    controller.dispatch("message.send", {"text": "typed here"}, actor=OWNER)

    assert display.shown == []


def test_an_agent_appending_context_is_drawn_too() -> None:
    from tagalong.application import bind_session_transcript_slice

    class Turns:
        def end_turn(self) -> None:
            return None

    class Rows:
        def transcript_entries(self):
            return []

    class NoAttachments:
        def upload(self, data: bytes) -> str:
            del data
            raise AssertionError("appending context uploads nothing")

        def resolve(self, ids):
            del ids
            raise AssertionError("appending context resolves nothing")

    display = FakeMessageDisplay()
    conversation = FakeConversation()
    controller = Controller(app_state_from_session(SessionState()))
    bind_session_transcript_slice(
        controller,
        (conversation, Turns(), NoAttachments(), Rows()),
        display=display,
    )
    caller = agent("notes-bot", {Scope.TRANSCRIPT})

    controller.dispatch("transcript.append", {"text": "a note"}, actor=caller)

    assert display.shown == [("Agent", "a note")]


def test_an_agent_can_upload_then_send_with_the_id(tmp_path) -> None:
    from tagalong.application import bind_session_transcript_slice
    from tagalong.attachments import AttachmentRegistry, AttachmentStore
    from tagalong.domain import AGENT

    class Turns:
        def end_turn(self) -> None:
            return None

    class Rows:
        def transcript_entries(self):
            return []

    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    conversation = FakeConversation()
    attachments = AttachmentRegistry(store=AttachmentStore(directory=tmp_path))
    controller = Controller(app_state_from_session(SessionState()))
    caller = agent("notes-bot", {Scope.CONVERSE})
    bind_first_slice(
        controller,
        conversation=conversation,
        tts=FakeSpeech(),
        attachments=attachments,
    )
    bind_session_transcript_slice(
        controller,
        (conversation, Turns(), attachments, Rows()),
        directory=tmp_path,
    )

    uploaded = controller.dispatch("attachment.upload", {"data": png}, actor=caller)
    assert isinstance(uploaded, Applied)
    attachment_id = str(uploaded.effective)
    assert controller.dispatch(
        "message.send",
        {"text": "see", "images": (attachment_id,), "respond": True},
        actor=caller,
    ) == Applied("req-2", (attachment_id,))
    assert conversation.ingested[0][:3] == (AGENT, "see", True)
    assert conversation.ingested[0][3] == attachments.resolve((attachment_id,))


def test_tts_updates_canonical_state_with_the_effective_flag() -> None:
    controller, _conversation, tts = bound()

    outcome = controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert outcome == Applied("req-1", False)
    assert tts.enabled is False
    assert controller.state.tts_enabled is False


def test_a_refused_tts_change_fails_and_leaves_state() -> None:
    controller, _conversation, tts = bound(tts=FakeSpeech(accept=False))

    outcome = controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert outcome == Failed("req-1", "tts could not be changed")
    assert tts.enabled is True
    assert controller.state.tts_enabled is True


def test_interrupt_without_a_generation_stops_whatever_is_current() -> None:
    controller, conversation, _tts = bound(conversation=FakeConversation(generation=4))

    outcome = controller.dispatch("session.interrupt", actor=OWNER)

    assert outcome == Applied("req-1", 4)
    assert conversation.interrupts == 1


def test_interrupt_of_a_stale_generation_does_not_cut_a_newer_turn() -> None:
    controller, conversation, _tts = bound(conversation=FakeConversation(generation=4))

    outcome = controller.dispatch("session.interrupt", {"generation": 3}, actor=OWNER)

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.INAPPLICABLE
    assert conversation.interrupts == 0


def test_interrupt_of_the_current_generation_is_applied() -> None:
    controller, conversation, _tts = bound(conversation=FakeConversation(generation=4))

    outcome = controller.dispatch("session.interrupt", {"generation": 4}, actor=OWNER)

    assert outcome == Applied("req-1", 4)
    assert conversation.interrupts == 1


def test_session_new_is_accepted_then_settles_on_the_worker() -> None:
    """Replaces the old dispatch/settle split pin: Accepted, then terminal Applied.

    The deleted ``test_a_new_session_is_accepted_before_the_thread_starts`` pinned
    bare dispatch as a no-op; handler-owned settle makes that obsolete.
    """
    controller, conversation, display, recorder = _bound_session_new()
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.applied"
        assert conversation.sessions == 1
        assert conversation.adopted == conversation.started
        assert display.resets == 1
        assert recorder.rolls == 1
    finally:
        subscription.close()


def test_session_new_refuses_an_unscoped_caller_without_starting() -> None:
    """A refused dispatch spawns no worker, so nothing moves off the app thread."""
    controller, conversation, display, recorder = _bound_session_new()
    caller = agent("notes-bot", {Scope.CONVERSE})

    outcome = controller.dispatch("session.new", actor=caller)

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.FORBIDDEN
    assert conversation.sessions == 0
    assert display.resets == 0
    assert recorder.rolls == 0


def test_session_new_fails_when_the_thread_cannot_start() -> None:
    controller, conversation, display, recorder = _bound_session_new(
        FakeConversation(new_session_ok=False)
    )
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.failed"
        assert "could not start a new session" in str(event.payload.get("detail", ""))
        assert display.resets == 0
        assert recorder.rolls == 0
        assert conversation.sessions == 0
    finally:
        subscription.close()


@pytest.mark.parametrize(
    ("transcript", "recorder"),
    [
        (None, None),
        (FakeDisplay(), None),
        (None, FakeRecorder()),
    ],
    ids=["neither", "no-recorder", "no-transcript"],
)
def test_session_new_fails_when_a_settle_port_is_missing(
    transcript: FakeDisplay | None, recorder: FakeRecorder | None
) -> None:
    """A peer without both settle ports gets a terminal failure, not a hung request.

    Every accepted ``session.new`` must reach a terminal outcome; nothing times
    pending requests out. Both ports are required, so each one missing on its
    own has to fail too.
    """
    controller, conversation, _tts = bound(transcript=transcript, recorder=recorder)
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.failed"
        assert "not available in this session" in str(event.payload.get("detail", ""))
        assert conversation.sessions == 0
    finally:
        subscription.close()


def test_session_new_settles_when_the_thread_starts() -> None:
    controller, conversation, display, recorder = _bound_session_new()
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.applied"
        assert conversation.sessions == 1
        assert conversation.adopted == conversation.started
        assert display.resets == 1
        assert recorder.rolls == 1
    finally:
        subscription.close()


def test_session_new_announces_applied_after_the_thread_is_live(
    monkeypatch,
) -> None:
    controller, conversation, _display, _recorder = _bound_session_new()
    conversation.thread = "thread-0"
    seen: list[tuple[str, object | None]] = []
    publish = controller._events.publish

    def record(name: str, payload=None):
        seen.append((name, conversation.thread))
        return publish(name, payload)

    monkeypatch.setattr(controller._events, "publish", record)
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.applied"
        assert ("action.accepted", "thread-0") in seen
        assert ("action.applied", "thread-1") in seen
        # accepted before the worker adopts; applied after.
        assert seen.index(("action.accepted", "thread-0")) < seen.index(
            ("action.applied", "thread-1")
        )
    finally:
        subscription.close()


def test_a_failed_roll_still_announces_the_adopted_session(monkeypatch) -> None:
    class FullDisk:
        def roll(self) -> None:
            raise OSError("No space left on device")

    # Exercise settle_new_session directly so the OSError is on this thread;
    # the daemon worker would swallow it after announce.
    controller, conversation, _tts = bound()
    conversation.thread = "thread-0"
    seen: list[tuple[str, object | None]] = []
    publish = controller._events.publish

    def record(name: str, payload=None):
        seen.append((name, conversation.thread))
        return publish(name, payload)

    monkeypatch.setattr(controller._events, "publish", record)

    # Pending-only accept via the real handler, then settle on this thread.
    # Re-register a no-spawn handler so the worker does not race settle.
    def accept_only(_request, state):
        from tagalong.control import Effect

        return Effect.pending(state, settle=lambda current, _effective: current)

    controller.register("session.new", accept_only)
    accepted = controller.dispatch("session.new", actor=OWNER)
    assert isinstance(accepted, Accepted)

    with pytest.raises(OSError, match="No space left on device"):
        settle_new_session(
            controller, accepted.request_id, conversation, FakeDisplay(), FullDisk()
        )

    assert conversation.adopted == ["thread-1"]
    assert seen == [
        ("action.accepted", "thread-0"),
        ("action.applied", "thread-1"),
    ]


def test_settle_new_session_does_not_hold_the_controller_lock_for_display() -> None:
    """Invariant: display/recorder run outside the controller (transcript) lock."""
    controller, conversation, _tts = bound()

    def accept_only(_request, state):
        from tagalong.control import Effect

        return Effect.pending(state, settle=lambda current, _effective: current)

    controller.register("session.new", accept_only)
    accepted = controller.dispatch("session.new", actor=OWNER)
    assert isinstance(accepted, Accepted)

    class LockCheckingDisplay:
        def __init__(self) -> None:
            self.resets = 0

        def reset_transcript(self) -> None:
            # Another thread would block; same-thread RLock is re-entrant, so
            # acquire(False) succeeding only proves we are not mid-claim on
            # *another* owner. Check the lock is free for a non-owner by
            # sampling from a helper thread.
            held = threading.Event()
            free = threading.Event()

            def probe() -> None:
                if controller._lock.acquire(blocking=False):
                    controller._lock.release()
                    free.set()
                else:
                    held.set()

            probe_thread = threading.Thread(target=probe)
            probe_thread.start()
            probe_thread.join(timeout=1)
            assert free.is_set(), "controller lock held during reset_transcript"
            assert not held.is_set()
            self.resets += 1

    display = LockCheckingDisplay()
    recorder = FakeRecorder()
    outcome = settle_new_session(
        controller, accepted.request_id, conversation, display, recorder
    )
    assert outcome == Applied(accepted.request_id)
    assert display.resets == 1
    assert recorder.rolls == 1


def test_a_superseded_new_session_does_not_replace_the_live_one() -> None:
    class SlowConversation:
        def __init__(self) -> None:
            self.generation = 0
            self.started: list[str] = []
            self.adopted: list[str] = []
            self._block_first = threading.Event()
            self._first_started = threading.Event()

        def ingest(self, *_args, **_kwargs) -> None:
            return None

        def interrupt(self) -> None:
            return None

        def start_fresh_thread(self):
            name = f"T{len(self.started) + 1}"
            self.started.append(name)
            if name == "T1":
                self._first_started.set()
                self._block_first.wait(timeout=2)
            return name

        def adopt_fresh_thread(self, started) -> None:
            self.adopted.append(started)

    conversation = SlowConversation()
    display = FakeDisplay()
    recorder = FakeRecorder()
    controller = Controller()
    bind_first_slice(
        controller,
        conversation=conversation,
        tts=FakeSpeech(),
        transcript=display,
        recorder=recorder,
    )
    _, subscription = controller.subscribe()
    try:
        first = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(first, Accepted)
        assert conversation._first_started.wait(timeout=2)
        second = controller.dispatch("session.new", actor=OWNER)
        assert isinstance(second, Accepted)
        conversation._block_first.set()
        terminals = _await_terminals(
            subscription, {first.request_id, second.request_id}
        )
        assert terminals[first.request_id].name == "action.superseded"
        assert terminals[second.request_id].name == "action.applied"
        assert conversation.started == ["T1", "T2"]
        assert conversation.adopted == ["T2"]
        assert display.resets == 1
        assert recorder.rolls == 1
    finally:
        subscription.close()


def test_the_tui_hooks_dispatch_the_first_slice() -> None:
    controller, conversation, tts = bound()
    tui = FakeTui()
    install_first_slice_hooks(tui, controller, OWNER)

    assert tui.hooks.on_user_text is not None
    assert tui.hooks.on_tts is not None
    assert tui.hooks.on_interrupt is not None
    tui.hooks.on_user_text(UserTextMessage("typed"))
    assert tui.hooks.on_tts(False) is True
    tui.hooks.on_interrupt()

    assert conversation.ingested == [("Text", "typed", True, ())]
    assert tts.enabled is False
    assert conversation.interrupts == 1
    assert controller.state.tts_enabled is False


def test_a_refused_tts_hook_reports_failure_to_the_interface() -> None:
    controller, _conversation, _tts = bound(tts=FakeSpeech(accept=False))
    tui = FakeTui()
    install_first_slice_hooks(tui, controller, OWNER)

    assert tui.hooks.on_tts is not None
    assert tui.hooks.on_tts(False) is False


def settings_bound(
    conversation: FakeConversation | None = None,
    tts: FakeSpeech | None = None,
    gate: FakeGate | None = None,
    silence: FakeSilence | None = None,
    recorded: list[tuple[str, object]] | None = None,
) -> tuple[Controller, FakeConversation, FakeSpeech, FakeGate, FakeSilence]:
    conversation = FakeConversation() if conversation is None else conversation
    tts = FakeSpeech() if tts is None else tts
    gate = FakeGate() if gate is None else gate
    silence = FakeSilence() if silence is None else silence
    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(controller, conversation=conversation, tts=tts)
    bind_settings_slice(
        controller,
        (conversation, tts, gate, silence),
        persist=(
            (lambda key, value: recorded.append((key, value)))
            if recorded is not None
            else None
        ),
    )
    return controller, conversation, tts, gate, silence


def audio_bound() -> tuple[Controller, FakeCapture, FakeCapture]:
    controller = Controller(app_state_from_session(SessionState()))
    microphone = FakeCapture()
    audio = FakeCapture()
    bind_first_slice(controller, conversation=FakeConversation(), tts=FakeSpeech())
    bind_audio_slice(controller, microphone=microphone, audio=audio)
    return controller, microphone, audio


def test_settings_actions_update_canonical_state() -> None:
    controller, conversation, tts, gate, silence = settings_bound()

    assert controller.dispatch(
        "response_policy.set", {"policy": "voice"}, actor=OWNER
    ) == Applied("req-1", "voice")
    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Accepted,
    )
    tts.complete_provider()
    assert controller.dispatch(
        "codex.set_model", {"model": "gpt-5.6-sol"}, actor=OWNER
    ) == Applied("req-3", "gpt-5.6-sol")
    assert controller.dispatch(
        "codex.set_reasoning", {"effort": "high"}, actor=OWNER
    ) == Applied("req-4", "high")
    assert controller.dispatch(
        "turn_silence.set", {"seconds": 0.01}, actor=OWNER
    ) == Applied("req-5", 0.25)

    assert gate.policies == ["voice"]
    assert tts.provider == "edge"
    assert conversation.models == ["gpt-5.6-sol"]
    assert conversation.efforts == ["high"]
    assert silence.seconds == 0.25
    assert controller.state.response_policy == "voice"
    assert controller.state.tts_provider == Selection(desired="edge", effective="edge")
    assert controller.state.codex_model == "gpt-5.6-sol"
    assert controller.state.codex_reasoning == "high"
    assert controller.state.turn_silence == 0.25


def test_provider_switch_restores_the_remembered_voice() -> None:
    state = SessionState(edge_voice="en-US-JennyNeural")
    controller = Controller(app_state_from_session(state))
    tts = FakeSpeech()
    bind_first_slice(controller, conversation=FakeConversation(), tts=tts)
    bind_settings_slice(
        controller, (FakeConversation(), tts, FakeGate(), FakeSilence())
    )

    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Accepted,
    )
    assert controller.state.tts_provider == Selection(desired="edge", effective="piper")
    tts.complete_provider()
    assert tts.voice == "en-US-JennyNeural"
    assert controller.state.tts_provider == Selection(desired="edge", effective="edge")
    assert controller.state.tts_voice == Selection(
        desired="en-US-JennyNeural", effective="en-US-JennyNeural"
    )


def test_tts_set_voice_persists_only_after_success() -> None:
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)
    prior = controller.state.tts_voice.effective

    outcome = controller.dispatch(
        "tts.set_voice", {"voice": "en_US-sam-medium"}, actor=OWNER
    )
    assert isinstance(outcome, Accepted)
    assert recorded == []
    assert controller.state.tts_voice.desired == "en_US-sam-medium"
    assert controller.state.tts_voice.effective == prior

    tts.complete_voice()
    assert controller.state.tts_voice == Selection(
        desired="en_US-sam-medium", effective="en_US-sam-medium"
    )
    assert controller.state.piper_voice == "en_US-sam-medium"
    assert recorded == [("piper_voice", "en_US-sam-medium")]


def test_tts_set_voice_failure_does_not_persist() -> None:
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    tts.voice_fail = "download failed"
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)
    before_voice = controller.state.piper_voice
    before_effective = controller.state.tts_voice.effective

    assert isinstance(
        controller.dispatch(
            "tts.set_voice", {"voice": "en_US-sam-medium"}, actor=OWNER
        ),
        Accepted,
    )
    tts.complete_voice()
    assert recorded == []
    assert controller.state.piper_voice == before_voice
    assert controller.state.tts_voice.effective == before_effective


def test_a_late_tts_provider_applied_callback_does_not_persist() -> None:
    """Only Applied settle outcomes write the engine; a retired request must not."""
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)

    outcome = controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER)
    assert isinstance(outcome, Accepted)
    assert controller.state.tts_provider == Selection(desired="edge", effective="piper")
    assert controller.fail(outcome.request_id, "cancelled") is not None
    tts.complete_provider()
    assert recorded == []
    # Failure leaves desired where acceptance put it; effective stays on the
    # engine that is still speaking (controller.fail does not roll back).
    assert controller.state.tts_provider == Selection(desired="edge", effective="piper")


def test_a_late_tts_voice_applied_callback_does_not_persist() -> None:
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)
    before = controller.state.piper_voice

    outcome = controller.dispatch(
        "tts.set_voice", {"voice": "en_US-sam-medium"}, actor=OWNER
    )
    assert isinstance(outcome, Accepted)
    assert controller.fail(outcome.request_id, "cancelled") is not None
    tts.complete_voice()
    assert recorded == []
    assert controller.state.piper_voice == before


def test_setting_the_provider_does_not_unmute_speech() -> None:
    """Unmute is a TUI picker composition, not part of ``tts.set_provider``."""
    controller, _conversation, tts, _gate, _silence = settings_bound()
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Accepted,
    )
    tts.complete_provider()
    assert controller.state.tts_enabled is False
    assert controller.state.tts_provider == Selection(desired="edge", effective="edge")


def test_settings_persistence_runs_from_the_handler_not_the_hook() -> None:
    recorded: list[tuple[str, object]] = []
    controller, _conversation, tts, _gate, _silence = settings_bound(recorded=recorded)

    controller.dispatch("response_policy.set", {"policy": "voice"}, actor=OWNER)
    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Accepted,
    )
    assert recorded == [("taga_after", "voice")]
    tts.complete_provider()
    controller.dispatch("codex.set_model", {"model": "gpt-5.6-sol"}, actor=OWNER)
    controller.dispatch("codex.set_reasoning", {"effort": "high"}, actor=OWNER)
    controller.dispatch("turn_silence.set", {"seconds": 0.01}, actor=OWNER)

    assert recorded == [
        ("taga_after", "voice"),
        ("tts_provider", "edge"),
        ("codex_model", "gpt-5.6-sol"),
        ("codex_reasoning", "high"),
        ("turn_silence", 0.25),
    ]


def test_a_refused_settings_change_is_not_persisted() -> None:
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    tts.provider_ok = False
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)

    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Failed,
    )
    assert recorded == []


def test_a_refused_settings_change_fails_and_leaves_state() -> None:
    tts = FakeSpeech()
    tts.provider_ok = False
    conversation = FakeConversation()
    conversation.model_ok = False
    conversation.effort_ok = False
    controller, *_rest = settings_bound(conversation=conversation, tts=tts)
    before = controller.state

    assert isinstance(
        controller.dispatch("tts.set_provider", {"provider": "edge"}, actor=OWNER),
        Failed,
    )
    assert isinstance(
        controller.dispatch("codex.set_model", {"model": "gpt-5.6-sol"}, actor=OWNER),
        Failed,
    )
    assert isinstance(
        controller.dispatch("codex.set_reasoning", {"effort": "high"}, actor=OWNER),
        Failed,
    )
    assert controller.state == before


def test_the_tui_hooks_dispatch_settings() -> None:
    controller, conversation, tts, gate, silence = settings_bound()
    tui = FakeTui()
    install_settings_hooks(tui, controller, OWNER)

    assert tui.hooks.on_policy is not None
    assert tui.hooks.on_codex_model is not None
    assert tui.hooks.on_codex_effort is not None
    assert tui.hooks.on_tts_provider is not None
    assert tui.hooks.on_turn_silence is not None
    assert tui.hooks.on_policy("quiet") is True
    assert tui.hooks.on_codex_model("gpt-5.6-sol") is True
    assert tui.hooks.on_codex_effort("high") is True
    assert tui.hooks.on_tts_provider("edge") is True
    tts.complete_provider()
    assert tui.hooks.on_turn_silence(1.25) == 1.25

    assert gate.policies == ["quiet"]
    assert conversation.models == ["gpt-5.6-sol"]
    assert conversation.efforts == ["high"]
    assert tts.provider == "edge"
    assert silence.seconds == 1.25


def test_a_refused_settings_hook_reports_failure_to_the_interface() -> None:
    tts = FakeSpeech()
    tts.provider_ok = False
    conversation = FakeConversation()
    conversation.model_ok = False
    conversation.effort_ok = False
    controller, *_rest = settings_bound(conversation=conversation, tts=tts)
    tui = FakeTui()
    install_settings_hooks(tui, controller, OWNER)

    assert tui.hooks.on_tts_provider is not None
    assert tui.hooks.on_tts_provider("edge") is False
    assert tui.hooks.on_codex_model is not None
    assert tui.hooks.on_codex_model("gpt-5.6-sol") is False
    assert tui.hooks.on_codex_effort is not None
    assert tui.hooks.on_codex_effort("high") is False
    forbidden = FakeTui()
    install_settings_hooks(forbidden, controller, agent("notes-bot", set()))
    assert forbidden.hooks.on_policy is not None
    assert forbidden.hooks.on_policy("quiet") is False
    assert forbidden.hooks.on_turn_silence is not None
    assert forbidden.hooks.on_turn_silence(1.25) is None


def test_microphone_select_is_accepted_then_settled() -> None:
    controller, microphone, _audio = audio_bound()

    outcome = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)

    assert isinstance(outcome, Accepted)
    assert microphone.selected == ["Yeti"]
    assert controller.state.microphone == Selection(desired="Yeti")
    assert microphone.apply("Yeti") == Applied("req-1", "Yeti")
    assert controller.state.microphone == Selection(desired="Yeti", effective="Yeti")


def test_a_superseded_microphone_select_does_not_apply() -> None:
    controller, microphone, _audio = audio_bound()

    first = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    second = controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)
    assert isinstance(first, Accepted)
    assert isinstance(second, Accepted)

    assert microphone.apply("Yeti", index=0) == Superseded("req-1")
    assert controller.state.microphone.effective is None
    assert microphone.apply("Webcam", index=1) == Applied("req-2", "Webcam")
    assert controller.state.microphone == Selection(
        desired="Webcam", effective="Webcam"
    )


def test_a_failed_audio_select_does_not_settle() -> None:
    controller, _microphone, audio = audio_bound()

    outcome = controller.dispatch("audio_stream.select", {"name": "Zoom"}, actor=OWNER)

    assert isinstance(outcome, Accepted)
    assert audio.fail("could not listen to Zoom") == Failed(
        "req-1", "could not listen to Zoom"
    )
    assert controller.state.audio_stream == Selection(desired="Zoom")
    assert controller.state.audio_stream.effective is None


def test_mute_applies_without_an_open_channel() -> None:
    controller, microphone, audio = audio_bound()

    assert controller.dispatch(
        "microphone.set_muted", {"muted": True}, actor=OWNER
    ) == Applied("req-1", True)
    assert controller.dispatch(
        "audio_stream.set_muted", {"muted": True}, actor=OWNER
    ) == Applied("req-2", True)

    assert microphone.muted is True
    assert audio.muted is True
    assert controller.state.microphone_muted is True
    assert controller.state.audio_stream_muted is True


def test_audio_hooks_report_a_forbidden_actor() -> None:
    controller, _microphone, _audio = audio_bound()
    tui = FakeTui()
    install_audio_hooks(tui, controller, agent("notes-bot", set()))

    assert tui.hooks.on_microphone is not None
    assert tui.hooks.on_microphone("Yeti") is False
    assert tui.hooks.on_audio_stream is not None
    assert tui.hooks.on_audio_stream("Zoom") is False
    assert tui.hooks.on_mute is not None
    assert tui.hooks.on_mute(True) is False
    assert tui.hooks.on_audio_mute is not None
    assert tui.hooks.on_audio_mute(True) is False
    assert tui.hooks.on_turn_silence is None


def test_the_tui_hooks_dispatch_audio_actions() -> None:
    controller, microphone, audio = audio_bound()
    tui = FakeTui()
    install_audio_hooks(tui, controller, OWNER)

    assert tui.hooks.on_microphone is not None
    assert tui.hooks.on_audio_stream is not None
    assert tui.hooks.on_mute is not None
    assert tui.hooks.on_audio_mute is not None
    assert tui.hooks.on_microphone("Yeti") is True
    assert tui.hooks.on_audio_stream("Zoom") is True
    assert tui.hooks.on_mute(True) is True
    assert tui.hooks.on_audio_mute(True) is True

    assert microphone.selected == ["Yeti"]
    assert audio.selected == ["Zoom"]
    assert microphone.muted is True
    assert audio.muted is True


def test_audio_persist_runs_only_for_the_applied_selection() -> None:
    persisted: list[str | None] = []
    controller = Controller(app_state_from_session(SessionState()))
    microphone = FakeCapture()
    bind_first_slice(controller, conversation=FakeConversation(), tts=FakeSpeech())
    bind_audio_slice(
        controller,
        microphone=microphone,
        audio=FakeCapture(),
        on_microphone_applied=persisted.append,
    )

    controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)
    microphone.apply("Yeti", index=0)
    microphone.apply("Webcam", index=1)

    assert persisted == ["Webcam"]


def test_session_and_transcript_actions_are_wired(tmp_path) -> None:
    from tagalong.application import bind_session_transcript_slice
    from tagalong.attachments import AttachmentRegistry, AttachmentStore
    from tagalong.domain import AGENT, TEXT
    from tagalong.presentation import Entry

    class Turns:
        def __init__(self) -> None:
            self.ended = 0

        def end_turn(self) -> None:
            self.ended += 1

    class Rows:
        def transcript_entries(self):
            return [Entry(kind="speech", source="Voice", text="hello", stamp="12:00")]

    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    conversation = FakeConversation()
    turns = Turns()
    attachments = AttachmentRegistry(store=AttachmentStore(directory=tmp_path / "a"))
    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(
        controller,
        conversation=conversation,
        tts=FakeSpeech(),
        attachments=attachments,
    )
    bind_session_transcript_slice(
        controller,
        (conversation, turns, attachments, Rows()),
        directory=tmp_path / "out",
    )

    uploaded = controller.dispatch("attachment.upload", {"data": png}, actor=OWNER)
    assert isinstance(uploaded, Applied)
    attachment_id = str(uploaded.effective)
    assert controller.dispatch(
        "message.send",
        {"text": "see", "images": (attachment_id,)},
        actor=OWNER,
    ) == Applied("req-2", (attachment_id,))
    assert conversation.ingested[0][3] == attachments.resolve((attachment_id,))

    assert controller.dispatch(
        "transcript.append", {"text": "note"}, actor=agent("bot", {Scope.TRANSCRIPT})
    ) == Applied("req-3", AGENT)
    assert conversation.ingested[1][:3] == (AGENT, "note", False)

    assert controller.dispatch(
        "transcript.append", {"text": "human note"}, actor=OWNER
    ) == Applied("req-4", TEXT)

    assert controller.dispatch("voice.end_turn", actor=OWNER) == Applied("req-5", None)
    assert turns.ended == 1

    saved = controller.dispatch("transcript.save", actor=OWNER)
    assert isinstance(saved, Applied)
    name = str(saved.effective)
    assert "/" not in name
    assert (tmp_path / "out" / name).read_text(encoding="utf-8").count("hello") == 1

    assert controller.dispatch("session.quit", actor=OWNER) == Applied("req-7", None)


def test_an_agent_cannot_quit_the_session(tmp_path) -> None:
    from tagalong.application import bind_session_transcript_slice
    from tagalong.attachments import AttachmentRegistry, AttachmentStore

    class Turns:
        def end_turn(self) -> None:
            return None

    class Rows:
        def transcript_entries(self):
            return []

    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(controller, conversation=FakeConversation(), tts=FakeSpeech())
    bind_session_transcript_slice(
        controller,
        (
            FakeConversation(),
            Turns(),
            AttachmentRegistry(store=AttachmentStore(directory=tmp_path)),
            Rows(),
        ),
        directory=tmp_path,
    )

    outcome = controller.dispatch("session.quit", actor=agent("bot", {Scope.SESSION}))
    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.FORBIDDEN
    assert "capability policy" in outcome.detail


def test_message_send_rejects_unknown_and_missing_attachments(tmp_path) -> None:
    from tagalong.attachments import AttachmentRegistry, AttachmentStore

    conversation = FakeConversation()
    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(controller, conversation=conversation, tts=FakeSpeech())

    assert isinstance(
        controller.dispatch(
            "message.send",
            {"text": "x", "images": ("missing",)},
            actor=OWNER,
        ),
        Failed,
    )

    attachments = AttachmentRegistry(store=AttachmentStore(directory=tmp_path))
    bound = Controller(app_state_from_session(SessionState()))
    bind_first_slice(
        bound, conversation=conversation, tts=FakeSpeech(), attachments=attachments
    )
    assert isinstance(
        bound.dispatch(
            "message.send",
            {"text": "x", "images": ("missing",)},
            actor=OWNER,
        ),
        Failed,
    )


def test_upload_and_save_failures_are_reported(monkeypatch) -> None:
    from tagalong.application import bind_session_transcript_slice

    class Turns:
        def end_turn(self) -> None:
            return None

    class Rows:
        def transcript_entries(self):
            return []

    class Broken:
        def upload(self, data: bytes) -> str:
            _ = data
            raise ValueError("not an image")

        def resolve(self, ids):
            del ids
            return ()

    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(controller, conversation=FakeConversation(), tts=FakeSpeech())
    bind_session_transcript_slice(
        controller,
        (
            FakeConversation(),
            Turns(),
            Broken(),
            Rows(),
        ),
    )
    assert isinstance(
        controller.dispatch("attachment.upload", {"data": b"nope"}, actor=OWNER),
        Failed,
    )

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("tagalong.application.write_transcript_export", boom)
    assert isinstance(controller.dispatch("transcript.save", actor=OWNER), Failed)


def test_session_transcript_hooks_dispatch(tmp_path) -> None:
    from tagalong.application import (
        bind_session_transcript_slice,
        install_session_transcript_hooks,
    )
    from tagalong.attachments import AttachmentRegistry, AttachmentStore
    from tagalong.presentation import Entry

    class Turns:
        def __init__(self) -> None:
            self.ended = 0

        def end_turn(self) -> None:
            self.ended += 1

    class Rows:
        def transcript_entries(self):
            return [Entry(kind="note", text="saved", stamp="1")]

    cleaned: list[bool] = []
    conversation = FakeConversation()
    turns = Turns()
    attachments = AttachmentRegistry(store=AttachmentStore(directory=tmp_path))
    controller = Controller(app_state_from_session(SessionState()))
    bind_first_slice(
        controller, conversation=conversation, tts=FakeSpeech(), attachments=attachments
    )
    bind_session_transcript_slice(
        controller,
        (conversation, turns, attachments, Rows()),
        directory=tmp_path,
    )
    tui = FakeTui()
    install_session_transcript_hooks(
        tui, controller, OWNER, on_quit_cleanup=lambda: cleaned.append(True)
    )

    assert tui.hooks.on_end_turn is not None
    tui.hooks.on_end_turn()
    assert turns.ended == 1
    assert tui.hooks.on_save is not None
    tui.hooks.on_save([])
    assert tui.hooks.on_attachment_upload is not None
    assert tui.hooks.on_attachment_upload(b"nope") is None
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    assert tui.hooks.on_attachment_upload(png) is not None
    assert tui.hooks.on_quit is not None
    assert tui.hooks.on_quit() is True
    assert cleaned == [True]

    install_session_transcript_hooks(tui, controller, OWNER)
    assert tui.hooks.on_quit is not None
    assert tui.hooks.on_quit() is True

    install_session_transcript_hooks(tui, controller, agent("bot", {Scope.SESSION}))
    assert tui.hooks.on_quit is not None
    assert tui.hooks.on_quit() is False


def test_a_refused_voice_change_fails_without_persisting() -> None:
    recorded: list[tuple[str, object]] = []
    tts = FakeSpeech()
    tts.voice_ok = False
    controller, *_rest = settings_bound(tts=tts, recorded=recorded)
    before = controller.state

    assert isinstance(
        controller.dispatch(
            "tts.set_voice", {"voice": "en_US-sam-medium"}, actor=OWNER
        ),
        Failed,
    )
    assert recorded == []
    assert controller.state == before


class _ReadAloudTTS:
    """Minimal TTS with the real TurnGate + EchoMemory collision semantics."""

    def __init__(self) -> None:
        self.turns = TurnGate()
        self.echo = EchoMemory()
        self.spoken: list[str] = []

    def begin_turn(self) -> None:
        self.turns.begin_turn()

    def speak(self, text: str) -> None:
        _turn, accepting = self.turns.accepting_turn()
        if text and accepting:
            self.echo.remember(text, retention=120)
            self.spoken.append(text)


def _await_terminal(subscription, request_id: str, timeout: float = 2.0):
    return _await_terminals(subscription, {request_id}, timeout=timeout)[request_id]


def _await_terminals(subscription, request_ids: set[str], timeout: float = 2.0):
    remaining = set(request_ids)
    found = {}
    deadline = timeout
    while deadline > 0 and remaining:
        if not subscription.wait(0.05):
            deadline -= 0.05
            continue
        for event in subscription.drain():
            rid = event.payload.get("request_id")
            if (
                event.name in {"action.applied", "action.failed", "action.superseded"}
                and rid in remaining
            ):
                found[str(rid)] = event
                remaining.discard(str(rid))
        deadline -= 0.05
    if remaining:
        raise AssertionError(f"no terminal event for {sorted(remaining)}")
    return found


def test_speech_read_selection_speaks_cleaned_text_and_echoes() -> None:
    tts = _ReadAloudTTS()
    controller = Controller()
    bind_read_aloud_slice(
        controller,
        tts=tts,
        read_selection=lambda: (
            "Gabriel Silva 10:42 AM Hello **team** — the deploy is live:tada:"
        ),
    )
    # Simulate Codex mid-reply: its queued text will be orphaned by begin_turn.
    tts.begin_turn()
    tts.speak("Codex was saying something long.")
    assert tts.echo.matches("Codex was saying something long.")

    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("speech.read_selection", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.applied"
        # Selection text must not ride action.applied into the EventLog.
        assert event.payload.get("effective") is None
        assert any("deploy is live" in chunk for chunk in tts.spoken)
        assert ":tada:" not in " ".join(tts.spoken)
        assert "Gabriel Silva" not in " ".join(tts.spoken)
        # Last-wins: selection speak is on a newer turn than the Codex probe.
        assert tts.turns.current_turn >= 2
        # EchoMemory must hold the selection so the mic does not re-transcribe it.
        assert tts.echo.matches(tts.spoken[-1])
    finally:
        subscription.close()


def test_speech_read_selection_fails_when_empty_or_tts_disabled() -> None:
    tts = _ReadAloudTTS()
    controller = Controller()
    bind_read_aloud_slice(controller, tts=tts, read_selection=lambda: None)
    _, subscription = controller.subscribe()
    try:
        empty = controller.dispatch("speech.read_selection", actor=OWNER)
        assert isinstance(empty, Accepted)
        event = _await_terminal(subscription, empty.request_id)
        assert event.name == "action.failed"
        assert "empty" in str(event.payload.get("detail", "")).lower()
    finally:
        subscription.close()

    disabled = Controller(AppState(tts_enabled=False))
    bind_read_aloud_slice(disabled, tts=tts, read_selection=lambda: "still selected")
    assert isinstance(
        disabled.dispatch("speech.read_selection", actor=OWNER),
        Failed,
    )


def test_speech_read_selection_fails_when_nothing_speakable_or_worker_raises() -> None:
    tts = _ReadAloudTTS()
    controller = Controller()
    bind_read_aloud_slice(controller, tts=tts, read_selection=lambda: ":tada: :eyes:")
    _, subscription = controller.subscribe()
    try:
        outcome = controller.dispatch("speech.read_selection", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.failed"
        assert "speakable" in str(event.payload.get("detail", "")).lower()
    finally:
        subscription.close()

    exploding = Controller()

    def boom() -> str | None:
        raise RuntimeError("selection helper crashed")

    bind_read_aloud_slice(exploding, tts=tts, read_selection=boom)
    _, subscription = exploding.subscribe()
    try:
        outcome = exploding.dispatch("speech.read_selection", actor=OWNER)
        assert isinstance(outcome, Accepted)
        event = _await_terminal(subscription, outcome.request_id)
        assert event.name == "action.failed"
        assert "crashed" in str(event.payload.get("detail", ""))
    finally:
        subscription.close()


def test_speech_read_selection_mcp_forbidden_electron_allowed() -> None:
    tts = _ReadAloudTTS()
    controller = Controller()
    bind_read_aloud_slice(
        controller, tts=tts, read_selection=lambda: "Hello from the tray."
    )
    mcp = agent("mcp-1", SOCKET_AGENT_SCOPES, denied_actions_for_socket_client("mcp"))
    electron = agent(
        "electron-1",
        SOCKET_AGENT_SCOPES,
        denied_actions_for_socket_client("electron"),
    )
    denied = controller.dispatch("speech.read_selection", actor=mcp)
    assert isinstance(denied, Rejected)
    assert denied.reason is Rejection.FORBIDDEN

    _, subscription = controller.subscribe()
    try:
        allowed = controller.dispatch("speech.read_selection", actor=electron)
        assert isinstance(allowed, Accepted)
        event = _await_terminal(subscription, allowed.request_id)
        assert event.name == "action.applied"
        assert any("Hello from the tray" in chunk for chunk in tts.spoken)
    finally:
        subscription.close()
