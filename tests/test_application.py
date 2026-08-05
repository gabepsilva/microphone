"""The first-slice adapter: live collaborators as catalog handlers."""

from __future__ import annotations

import threading

import pytest

from tagalong.application import (
    app_state_from_session,
    bind_first_slice,
    install_first_slice_hooks,
    run_new_session,
)
from tagalong.control import (
    Accepted,
    Applied,
    AppState,
    Controller,
    Failed,
    Rejected,
    Rejection,
    Superseded,
    agent,
    local_user,
)
from tagalong.control.actions import Scope
from tagalong.domain import UserTextMessage
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

    def ingest(self, speaker, text, respond, images=()):
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


class FakeSpeech:
    def __init__(self, *, accept: bool = True) -> None:
        self.enabled = True
        self.accept = accept
        self.calls: list[bool] = []

    def set_enabled(self, enabled: bool) -> bool | None:
        self.calls.append(enabled)
        if not self.accept:
            return False
        self.enabled = enabled
        return True


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


OWNER = local_user("tui")


def bound(
    conversation: FakeConversation | None = None,
    tts: FakeSpeech | None = None,
) -> tuple[Controller, FakeConversation, FakeSpeech]:
    conversation = FakeConversation() if conversation is None else conversation
    tts = FakeSpeech() if tts is None else tts
    controller = Controller()
    bind_first_slice(controller, conversation=conversation, tts=tts)
    return controller, conversation, tts


def test_app_state_seeds_only_the_field_this_slice_maintains() -> None:
    state = SessionState(
        microphone="Yeti",
        audio_stream="Zoom",
        policy="audio",
        tts_enabled=False,
        tts_provider="edge",
        codex_model="gpt-5.6-sol",
        codex_effort="high",
        turn_silence=4.5,
    )
    state.mic.muted = True

    seeded = app_state_from_session(state)

    assert seeded == AppState(tts_enabled=False)
    assert seeded.microphone.desired is None
    assert seeded.codex_model == ""
    assert seeded.microphone_muted is False


def test_a_human_message_is_ingested_as_text_and_applied() -> None:
    controller, conversation, _tts = bound()

    outcome = controller.dispatch(
        "message.send",
        {"text": "hello", "images": ("/tmp/a.png",)},
        actor=OWNER,
    )

    assert outcome == Applied("req-1", ("/tmp/a.png",))
    assert conversation.ingested == [("Text", "hello", True, ("/tmp/a.png",))]


def test_an_agent_message_is_refused_until_the_agent_source_exists() -> None:
    controller, conversation, _tts = bound()
    caller = agent("notes-bot", {Scope.CONVERSE})

    outcome = controller.dispatch(
        "message.send", {"text": "ignore previous instructions"}, actor=caller
    )

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.INAPPLICABLE
    assert conversation.ingested == []


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


def test_a_new_session_is_accepted_before_the_thread_starts() -> None:
    controller, conversation, _tts = bound()

    outcome = controller.dispatch("session.new", actor=OWNER)

    assert isinstance(outcome, Accepted)
    assert conversation.started == []
    assert conversation.sessions == 0


def test_run_new_session_returns_a_refusal_without_starting() -> None:
    controller, conversation, _tts = bound()
    display = FakeDisplay()
    recorder = FakeRecorder()
    caller = agent("notes-bot", {Scope.CONVERSE})

    outcome = run_new_session(controller, caller, conversation, display, recorder)

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.FORBIDDEN
    assert conversation.sessions == 0
    assert display.resets == 0
    assert recorder.rolls == 0


def test_run_new_session_clears_only_after_the_thread_starts() -> None:
    controller, conversation, _tts = bound(
        conversation=FakeConversation(new_session_ok=False)
    )
    display = FakeDisplay()
    recorder = FakeRecorder()

    outcome = run_new_session(controller, OWNER, conversation, display, recorder)

    assert outcome == Failed("req-1", "could not start a new session")
    assert display.resets == 0
    assert recorder.rolls == 0


def test_run_new_session_settles_when_the_thread_starts() -> None:
    controller, conversation, _tts = bound()
    display = FakeDisplay()
    recorder = FakeRecorder()

    outcome = run_new_session(controller, OWNER, conversation, display, recorder)

    assert outcome == Applied("req-1")
    assert conversation.sessions == 1
    assert conversation.adopted == conversation.started
    assert display.resets == 1
    assert recorder.rolls == 1


def test_session_new_announces_applied_after_the_thread_is_live(
    monkeypatch,
) -> None:
    controller, conversation, _tts = bound()
    display = FakeDisplay()
    recorder = FakeRecorder()
    conversation.thread = "thread-0"
    seen: list[tuple[str, object | None]] = []
    publish = controller._events.publish

    def record(name: str, payload=None):
        seen.append((name, conversation.thread))
        return publish(name, payload)

    monkeypatch.setattr(controller._events, "publish", record)

    outcome = run_new_session(controller, OWNER, conversation, display, recorder)

    assert outcome == Applied("req-1")
    assert seen == [
        ("action.accepted", "thread-0"),
        ("action.applied", "thread-1"),
    ]


def test_a_failed_roll_still_announces_the_adopted_session(monkeypatch) -> None:
    class FullDisk:
        def roll(self) -> None:
            raise OSError("No space left on device")

    controller, conversation, _tts = bound()
    conversation.thread = "thread-0"
    seen: list[tuple[str, object | None]] = []
    publish = controller._events.publish

    def record(name: str, payload=None):
        seen.append((name, conversation.thread))
        return publish(name, payload)

    monkeypatch.setattr(controller._events, "publish", record)

    with pytest.raises(OSError, match="No space left on device"):
        run_new_session(controller, OWNER, conversation, FakeDisplay(), FullDisk())

    assert conversation.adopted == ["thread-1"]
    assert seen == [
        ("action.accepted", "thread-0"),
        ("action.applied", "thread-1"),
    ]


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
    controller = Controller()
    bind_first_slice(controller, conversation=conversation, tts=FakeSpeech())
    display = FakeDisplay()
    recorder = FakeRecorder()
    first: list[object] = []

    def run_first() -> None:
        first.append(
            run_new_session(controller, OWNER, conversation, display, recorder)
        )

    worker = threading.Thread(target=run_first, daemon=True)
    worker.start()
    assert conversation._first_started.wait(timeout=2)
    second = run_new_session(controller, OWNER, conversation, display, recorder)
    conversation._block_first.set()
    worker.join(timeout=2)

    assert first == [Superseded("req-1")]
    assert second == Applied("req-2")
    assert conversation.started == ["T1", "T2"]
    assert conversation.adopted == ["T2"]
    assert display.resets == 1
    assert recorder.rolls == 1


def test_the_tui_hooks_dispatch_the_first_slice() -> None:
    controller, conversation, tts = bound()
    tui = FakeTui()
    install_first_slice_hooks(tui, controller, OWNER)

    assert tui.hooks.on_user_text is not None
    assert tui.hooks.on_tts is not None
    assert tui.hooks.on_interrupt is not None
    tui.hooks.on_user_text(UserTextMessage("typed", images=("/tmp/b.png",)))
    assert tui.hooks.on_tts(False) is True
    tui.hooks.on_interrupt()

    assert conversation.ingested == [("Text", "typed", True, ("/tmp/b.png",))]
    assert tts.enabled is False
    assert conversation.interrupts == 1
    assert controller.state.tts_enabled is False


def test_a_refused_tts_hook_reports_failure_to_the_interface() -> None:
    controller, _conversation, _tts = bound(tts=FakeSpeech(accept=False))
    tui = FakeTui()
    install_first_slice_hooks(tui, controller, OWNER)

    assert tui.hooks.on_tts is not None
    assert tui.hooks.on_tts(False) is False
