"""The /new command's transcript and Codex-session reset contract."""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace

from tagalong import codex as codex_module
from tagalong.application import apply_new_session
from tagalong.cli import (
    build_command_router,
    reset_codex_session,
    show_command_help,
)
from tagalong.codex import (
    CODEX_DEVELOPER_INSTRUCTIONS,
    CodexConversation,
    CodexSettings,
    load_codex_sdk,
)
from tagalong.commands import Command, CommandRouter
from tagalong.domain import TEXT
from tagalong.tui import PromptInput, VoiceCodexTUI


class FakeDisplay:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def begin_codex(self) -> None:
        self.calls.append(("begin_codex",))

    def codex_message_open(self, reply_to) -> None:
        self.calls.append(("codex_message_open", reply_to))

    def codex_delta(self, delta) -> None:
        self.calls.append(("codex_delta", delta))

    def codex_message_close(self) -> None:
        self.calls.append(("codex_message_close",))

    def reasoning_started(self) -> None:
        self.calls.append(("reasoning_started",))

    def reasoning_delta(self, delta) -> None:
        self.calls.append(("reasoning_delta", delta))

    def reasoning_completed(self) -> None:
        self.calls.append(("reasoning_completed",))

    def command_started(self, command) -> None:
        self.calls.append(("command_started", command))

    def command_output(self, delta) -> None:
        self.calls.append(("command_output", delta))

    def command_completed(self, exit_code) -> None:
        self.calls.append(("command_completed", exit_code))

    def tool_called(self, server, tool) -> None:
        self.calls.append(("tool_called", server, tool))

    def tool_completed(self, status) -> None:
        self.calls.append(("tool_completed", status))

    def token_usage(self, total_tokens) -> None:
        self.calls.append(("token_usage", total_tokens))

    def end_codex(self) -> None:
        self.calls.append(("end_codex",))

    def error(self, message) -> None:
        self.calls.append(("error", message))

    def note(self, text) -> None:
        self.calls.append(("note", text))

    def set_codex(self, **fields) -> None:
        self.calls.append(("set_codex", fields))

    def set_codex_catalog(
        self, models, efforts_by_model, default_effort_by_model
    ) -> None:
        self.calls.append(
            ("set_codex_catalog", models, efforts_by_model, default_effort_by_model)
        )


class FakeThread:
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id
        self.turns: list[str] = []
        self.next_turn = FakeTurn()

    def turn(self, prompt, **_kwargs):
        self.turns.append(prompt)
        return self.next_turn


class FakeTurn:
    def __init__(self) -> None:
        self.interrupts = 0

    def stream(self):
        return iter(())

    def interrupt(self) -> None:
        self.interrupts += 1


class FakeCodex:
    def __init__(self) -> None:
        self.threads: list[FakeThread] = []
        self.closed = False
        self.start_error: Exception | None = None

    def thread_start(self, **kwargs):
        if self.start_error is not None:
            raise self.start_error
        self.start_kwargs = kwargs
        thread = FakeThread(f"thread-{len(self.threads) + 1}")
        self.threads.append(thread)
        return thread

    def close(self) -> None:
        self.closed = True


def test_new_session_discards_queued_context_and_keeps_its_settings(
    monkeypatch,
) -> None:
    from openai_codex import ApprovalMode, Sandbox

    load_codex_sdk()
    fake_codex = FakeCodex()
    display = FakeDisplay()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: fake_codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            service_tier="fast",
        ),
        display,
    )
    try:
        conversation.ingest(TEXT, "old context", respond=True, timestamp="T1")
        stale = conversation.requests.get_nowait()
        conversation.request_model("gpt-5.6-sol")
        conversation.request_reasoning_effort("high")

        assert conversation.new_session() is True

        assert conversation.thread.id == "thread-2"
        assert conversation.generation == 1
        assert not conversation.is_current(stale.generation)
        assert conversation.router.pending_context == []
        assert fake_codex.start_kwargs == {
            "model": "gpt-5.6-sol",
            "service_tier": "fast",
            "sandbox": Sandbox("read-only"),
            "approval_mode": ApprovalMode.deny_all,
            "cwd": os.getcwd(),
            "developer_instructions": CODEX_DEVELOPER_INSTRUCTIONS,
        }
        assert conversation.requested_model is None
        assert conversation.requested_reasoning_effort is None
        assert display.calls[-1] == (
            "set_codex",
            {
                "model": "gpt-5.6-sol",
                "effort": "high",
                "thread": "thread-2",
                "state": "idle",
            },
        )

        conversation._run_codex(stale)

        assert fake_codex.threads[0].turns == []
        assert conversation.new_session() is True
        assert conversation.generation == 2
    finally:
        conversation.close()


def test_reset_drops_a_late_reply_from_the_discarded_session(monkeypatch) -> None:
    load_codex_sdk()
    fake_codex = FakeCodex()
    display = FakeDisplay()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: fake_codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )

    class ResettingTurn(FakeTurn):
        def stream(self):
            conversation.new_session()
            yield SimpleNamespace(
                payload=codex_module.AgentMessageDeltaNotification.model_construct(
                    delta="late reply"
                )
            )

    try:
        fake_codex.threads[0].next_turn = ResettingTurn()
        conversation.ingest(TEXT, "start over", respond=True, timestamp="T1")

        conversation._run_codex(conversation.requests.get_nowait())

        assert [name for name, *_ in display.calls] == ["begin_codex", "set_codex"]
    finally:
        conversation.close()


def test_a_failed_reset_keeps_the_current_session_and_pending_settings(
    monkeypatch,
) -> None:
    load_codex_sdk()
    fake_codex = FakeCodex()
    display = FakeDisplay()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: fake_codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )
    try:
        original_thread = conversation.thread
        conversation.request_model("gpt-5.6-sol")
        fake_codex.start_error = RuntimeError("offline")

        assert conversation.new_session() is False

        assert conversation.thread is original_thread
        assert conversation.requested_model == "gpt-5.6-sol"
        assert display.calls[-1] == (
            "error",
            "Could not start a new Codex session: offline",
        )
    finally:
        conversation.close()


def test_a_reset_interrupts_the_pending_guess_and_speech(monkeypatch) -> None:
    class FakeSpeech:
        def __init__(self) -> None:
            self.interrupts = 0

        def interrupt(self) -> None:
            self.interrupts += 1

        def close(self) -> None:
            pass

    load_codex_sdk()
    fake_codex = FakeCodex()
    display = FakeDisplay()
    speech = FakeSpeech()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: fake_codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
        speech,
    )
    try:
        assert conversation.prefire(TEXT, "unfinished", timestamp="T1")
        guess = conversation.speculation
        assert guess is not None
        guess.turn = FakeTurn()
        active = FakeTurn()
        conversation.active_turn = active

        assert conversation.new_session() is True

        assert guess.turn.interrupts == 1
        assert active.interrupts == 1
        assert conversation.speculation is None
        assert speech.interrupts == 1
        assert conversation.warmup_pending is True
    finally:
        conversation.close()


def test_stale_work_cannot_start_a_turn_or_warm_the_new_session(monkeypatch) -> None:
    load_codex_sdk()
    fake_codex = FakeCodex()
    display = FakeDisplay()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: fake_codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )
    try:
        assert (
            conversation._attempt("old prompt", TEXT, conversation.generation + 1) == []
        )
        monkeypatch.setattr(CodexConversation, "_start_turn", lambda *_args: None)

        conversation._warm_up()

        assert fake_codex.threads[0].turns == []
    finally:
        conversation.close()


def test_reset_hook_clears_only_after_a_new_session_starts() -> None:
    class Conversation:
        def __init__(self, started) -> None:
            self.started = started

        def new_session(self) -> bool:
            return self.started

    class Tui:
        def __init__(self) -> None:
            self.resets = 0
            self.notes: list[str] = []

        def reset_transcript(self) -> None:
            self.resets += 1

        def note(self, text: str) -> None:
            self.notes.append(text)

    tui = Tui()

    class Recorder:
        def __init__(self) -> None:
            self.rolls = 0

        def roll(self) -> None:
            self.rolls += 1

    recorder = Recorder()
    reset_codex_session(
        Command("new", ()),
        tui,
        lambda: apply_new_session(Conversation(False), tui, recorder),
    )
    reset_codex_session(
        Command("new", ()),
        tui,
        lambda: apply_new_session(Conversation(True), tui, recorder),
    )
    reset_codex_session(
        Command("new", ("again",)),
        tui,
        lambda: apply_new_session(Conversation(True), tui, recorder),
    )

    assert tui.resets == 1
    assert recorder.rolls == 1
    assert tui.notes == ["usage: /new"]


def test_new_command_clears_the_transcript_without_changing_controls() -> None:
    facade = VoiceCodexTUI()
    commands = CommandRouter(facade)
    commands.register("new", lambda _command: facade.reset_transcript())
    facade.hooks.on_command = commands.handle
    saved_controls = (
        facade.state.codex_model,
        facade.state.turn_silence,
        facade.state.tts_enabled,
    )

    async def exercise() -> None:
        async with facade.app.run_test() as pilot:
            facade._ready.set()
            facade._app_thread = threading.get_ident()
            facade.commit(TEXT, "previous session")
            await pilot.pause()
            facade.app.query_one("#input", PromptInput).value = "/new"
            await pilot.press("enter")
            await pilot.pause()

    asyncio.run(exercise())

    assert facade.app.entries == []
    assert (
        facade.state.codex_model,
        facade.state.turn_silence,
        facade.state.tts_enabled,
    ) == (saved_controls)


def test_build_command_router_registers_new_and_help() -> None:
    class Conversation:
        def new_session(self) -> bool:
            return True

    class Tui:
        def __init__(self) -> None:
            self.notes: list[str] = []
            self.resets = 0

        def note(self, text: str) -> None:
            self.notes.append(text)

        def reset_transcript(self) -> None:
            self.resets += 1

    tui = Tui()

    class Recorder:
        def __init__(self) -> None:
            self.rolls = 0

        def roll(self) -> None:
            self.rolls += 1

    recorder = Recorder()
    commands = build_command_router(tui, Conversation(), recorder)

    assert [spec.name for spec in commands.specs()] == ["new", "help"]
    commands.handle("/help")
    assert tui.notes
    assert tui.notes[0].startswith("commands:")
    assert "/new" in tui.notes[0]
    assert "/help" in tui.notes[0]

    commands.handle("/clear")
    assert tui.resets == 1
    assert recorder.rolls == 1


def test_help_for_one_command_names_aliases_and_unknowns() -> None:
    class Tui:
        def __init__(self) -> None:
            self.notes: list[str] = []

        def note(self, text: str) -> None:
            self.notes.append(text)

    tui = Tui()
    commands = CommandRouter(tui)
    commands.register(
        "new",
        lambda _command: None,
        description="Fresh session",
        aliases=("clear",),
    )

    show_command_help(Command("help", ("clear",)), commands, tui)
    show_command_help(Command("help", ("missing",)), commands, tui)

    assert tui.notes == [
        "/new (aliases: /clear): Fresh session",
        "unknown command: /missing",
    ]
