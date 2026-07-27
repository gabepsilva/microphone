"""Codex turn rendering and conversation lifecycle.

The Codex client is faked at its boundary. Turn events are built from the
real notification types with ``model_construct`` so the renderer's isinstance
dispatch is exercised against the shapes it will actually see.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    CommandExecutionOutputDeltaNotification,
    CommandExecutionThreadItem,
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallThreadItem,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)

WAIT_SECONDS = 10


def event(payload):
    return SimpleNamespace(payload=payload)


def message_item():
    return AgentMessageThreadItem.model_construct()


def command_item(command="ls -la", exit_code=0):
    return CommandExecutionThreadItem.model_construct(
        command=command, exit_code=exit_code
    )


def tool_item(server="files", tool="read", status="completed"):
    return McpToolCallThreadItem.model_construct(
        server=server, tool=tool, status=status
    )


def started(item):
    return event(ItemStartedNotification.model_construct(item=item))


def finished(item):
    return event(ItemCompletedNotification.model_construct(item=item))


def delta(text):
    return event(AgentMessageDeltaNotification.model_construct(delta=text))


def command_output(text):
    return event(CommandExecutionOutputDeltaNotification.model_construct(delta=text))


def usage(total_tokens):
    return event(
        ThreadTokenUsageUpdatedNotification.model_construct(
            token_usage=SimpleNamespace(last=SimpleNamespace(total_tokens=total_tokens))
        )
    )


def failure(message):
    return event(
        ErrorNotification.model_construct(error=SimpleNamespace(message=message))
    )


def turn_completed():
    return event(TurnCompletedNotification.model_construct())


class FakeDisplay:
    """Record every presentation call in the order it arrived."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, *args) if args else (name,))
            if kwargs:
                self.calls[-1] = (name, kwargs)

        return record

    def names(self):
        return [call[0] for call in self.calls]


def render(voice, events, chunker=None, reply_to="Them"):
    display = FakeDisplay()
    voice.CodexTurnRenderer(display, reply_to, chunker).render(events)
    return display


def test_an_assistant_message_is_opened_streamed_and_closed(voice) -> None:
    display = render(
        voice,
        [
            started(message_item()),
            delta("Hello "),
            delta("there."),
            finished(message_item()),
            turn_completed(),
        ],
    )

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Hello "),
        ("codex_delta", "there."),
        ("codex_message_close",),
    ]


def test_a_delta_without_a_start_still_opens_the_message(voice) -> None:
    display = render(voice, [delta("Sudden text."), turn_completed()])

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Sudden text."),
        ("codex_message_close",),
    ]


def test_the_message_is_opened_once_across_many_deltas(voice) -> None:
    display = render(voice, [delta("a"), delta("b"), delta("c")])

    assert display.names().count("codex_message_open") == 1


def test_a_command_closes_the_open_message_before_it_is_shown(voice) -> None:
    display = render(
        voice,
        [
            delta("Let me check."),
            started(command_item("ls -la")),
            command_output("total 0\n"),
            finished(command_item(exit_code=2)),
            turn_completed(),
        ],
    )

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Let me check."),
        ("codex_message_close",),
        ("command_started", "ls -la"),
        ("command_output", "total 0\n"),
        ("command_completed", 2),
    ]


def test_a_tool_call_closes_the_open_message_before_it_is_shown(voice) -> None:
    display = render(
        voice,
        [
            delta("Looking it up."),
            started(tool_item("files", "read")),
            finished(tool_item(status="completed")),
        ],
    )

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Looking it up."),
        ("codex_message_close",),
        ("tool_called", "files", "read"),
        ("tool_completed", "completed"),
    ]


def test_a_command_with_no_message_open_needs_no_close(voice) -> None:
    display = render(voice, [started(command_item("pwd"))])

    assert display.calls == [("command_started", "pwd")]


def test_an_error_notification_is_shown(voice) -> None:
    display = render(voice, [failure("model overloaded")])

    assert display.calls == [("error", "model overloaded")]


def test_token_usage_is_reported_only_when_the_turn_completes(voice) -> None:
    display = render(voice, [delta("Hi."), usage(1234)])

    assert "token_usage" not in display.names()

    display = render(voice, [delta("Hi."), usage(1234), turn_completed()])

    assert display.calls[-1] == ("token_usage", 1234)


def test_the_latest_token_usage_wins(voice) -> None:
    display = render(voice, [usage(10), usage(99), turn_completed()])

    assert display.calls == [("token_usage", 99)]


def test_a_turn_without_usage_reports_none(voice) -> None:
    display = render(voice, [delta("Hi."), turn_completed()])

    assert "token_usage" not in display.names()


def test_a_wrapped_item_is_unwrapped_before_dispatch(voice) -> None:
    display = render(voice, [started(SimpleNamespace(root=command_item("echo hi")))])

    assert display.calls == [("command_started", "echo hi")]


def test_item_root_passes_through_an_unwrapped_item(voice) -> None:
    item = command_item()

    assert voice.item_root(item) is item


def test_an_unrecognised_payload_is_ignored(voice) -> None:
    display = render(voice, [event(SimpleNamespace(kind="something new"))])

    assert display.calls == []


def test_speech_is_flushed_at_every_command_boundary(voice) -> None:
    from voice_codex.domain import SentenceChunker

    spoken: list[str] = []
    chunker = SentenceChunker(spoken.append)

    render(
        voice,
        [
            delta("First sentence. Second one"),
            started(command_item()),
            delta(" resumes after."),
            turn_completed(),
        ],
        chunker=chunker,
    )

    assert spoken == ["First sentence.", "Second one", "resumes after."]


def test_speech_is_flushed_when_the_turn_ends(voice) -> None:
    from voice_codex.domain import SentenceChunker

    spoken: list[str] = []
    render(
        voice,
        [delta("No trailing punctuation"), turn_completed()],
        chunker=SentenceChunker(spoken.append),
    )

    assert spoken == ["No trailing punctuation"]


class FakeTurn:
    def __init__(self, events=(), error=None):
        self.events = list(events)
        self.error = error
        self.interrupted = 0

    def stream(self):
        if self.error is not None:
            raise self.error
        yield from self.events

    def interrupt(self):
        self.interrupted += 1


class FakeThread:
    def __init__(self, thread_id="thread-1"):
        self.id = thread_id
        self.turns: list[str] = []
        self.next_turn = FakeTurn()

    def turn(self, prompt, **kwargs):
        self.turns.append(prompt)
        self.kwargs = kwargs
        return self.next_turn


class FakeCodex:
    def __init__(self):
        self.thread = FakeThread()
        self.closed = False
        self.fork_error = None
        self.forked_from = None

    def thread_start(self, **kwargs):
        self.start_kwargs = kwargs
        return self.thread

    def thread_fork(self, thread_id, **kwargs):
        if self.fork_error is not None:
            raise self.fork_error
        self.forked_from = thread_id
        self.fork_kwargs = kwargs
        return FakeThread("thread-2")

    def close(self):
        self.closed = True


@pytest.fixture
def conversation(voice, monkeypatch):
    codex = FakeCodex()
    monkeypatch.setattr(voice, "Codex", lambda: codex)
    display = FakeDisplay()
    built = voice.CodexConversation(
        voice.CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )
    built.fake_codex = codex
    built.fake_display = display
    yield built
    built.close()


def test_context_accumulates_until_a_reply_is_requested(conversation) -> None:
    conversation.ingest("Them", "A question", respond=False, timestamp="T1")
    conversation.ingest("User Voice", "And this", respond=True, timestamp="T2")

    request = conversation.requests.get(timeout=WAIT_SECONDS)

    assert request.reply_to == "User Voice"
    assert conversation.context_entries(request) == [
        {"timestamp": "T1", "source": "Them", "text": "A question"},
        {"timestamp": "T2", "source": "User Voice", "text": "And this"},
    ]


def test_a_timestamp_is_generated_when_none_is_given(conversation) -> None:
    conversation.ingest("Them", "A question", respond=True)

    request = conversation.requests.get(timeout=WAIT_SECONDS)

    assert request.entries[0].timestamp


def test_a_closed_conversation_ignores_new_input(conversation) -> None:
    conversation.close()

    conversation.ingest("Them", "Too late", respond=True)

    assert conversation.router.pending_context == []


def test_a_requested_model_forks_the_thread_before_the_next_turn(
    conversation,
) -> None:
    assert conversation.request_model("gpt-5.6-nova") is True

    conversation._apply_pending_settings()

    assert conversation.model == "gpt-5.6-nova"
    assert conversation.fake_codex.forked_from == "thread-1"
    assert conversation.thread.id == "thread-2"
    assert ("note", "Codex model → gpt-5.6-nova") in conversation.fake_display.calls


def test_a_failed_model_switch_keeps_the_original_thread(conversation) -> None:
    conversation.fake_codex.fork_error = RuntimeError("model unavailable")
    conversation.request_model("gpt-5.6-nova")

    conversation._apply_pending_settings()

    assert conversation.model == "gpt-5.6-luna"
    assert conversation.thread.id == "thread-1"
    assert (
        "error",
        "Could not switch Codex model: model unavailable",
    ) in conversation.fake_display.calls


def test_reselecting_the_current_model_does_not_fork(conversation) -> None:
    conversation.request_model("gpt-5.6-luna")

    conversation._apply_pending_settings()

    assert conversation.fake_codex.forked_from is None


def test_a_requested_reasoning_effort_is_applied(conversation) -> None:
    assert conversation.request_reasoning_effort("high") is True

    conversation._apply_pending_settings()

    assert conversation.reasoning_effort == "high"
    assert ("set_codex", {"effort": "high"}) in conversation.fake_display.calls


def test_a_turn_renders_between_begin_and_end_markers(conversation) -> None:
    conversation.thread.next_turn = FakeTurn([delta("Answer."), turn_completed()])
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")
    request = conversation.requests.get(timeout=WAIT_SECONDS)

    conversation._run_codex(request)
    names = conversation.fake_display.names()

    assert names[0] == "begin_codex"
    assert names[-1] == "end_codex"
    assert "codex_delta" in names
    assert conversation.active_turn is None


def test_a_failing_turn_is_reported_and_still_ends_cleanly(conversation) -> None:
    conversation.thread.next_turn = FakeTurn(error=RuntimeError("stream broke"))
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")
    request = conversation.requests.get(timeout=WAIT_SECONDS)

    conversation._run_codex(request)

    assert ("error", "Codex error: stream broke") in conversation.fake_display.calls
    assert conversation.fake_display.names()[-1] == "end_codex"
    assert conversation.active_turn is None


def test_the_prompt_names_the_source_to_reply_to(conversation) -> None:
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")
    request = conversation.requests.get(timeout=WAIT_SECONDS)

    conversation._run_codex(request)
    prompt = conversation.thread.turns[0]

    assert "Reply now to the latest Them input" in prompt
    assert '"text": "A question"' in prompt


def test_interrupting_stops_the_active_turn_and_its_speech(voice, monkeypatch) -> None:
    codex = FakeCodex()
    monkeypatch.setattr(voice, "Codex", lambda: codex)
    tts = SimpleNamespace(
        interrupted=0,
        closed=0,
        interrupt=lambda: None,
        close=lambda: None,
    )
    interrupts: list[str] = []
    tts.interrupt = lambda: interrupts.append("tts")
    tts.close = lambda: interrupts.append("closed")
    built = voice.CodexConversation(
        voice.CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
        tts,
    )
    turn = FakeTurn()
    built.active_turn = turn
    try:
        built.interrupt()

        assert turn.interrupted == 1
        assert interrupts == ["tts"]
    finally:
        built.close()

    assert codex.closed
    assert "closed" in interrupts


def test_interrupting_without_an_active_turn_is_harmless(conversation) -> None:
    conversation.interrupt()

    assert conversation.active_turn is None


def test_the_worker_drains_queued_requests(conversation) -> None:
    rendered = threading.Event()

    class WatchedTurn(FakeTurn):
        def stream(self):
            rendered.set()
            return iter([delta("Answer."), turn_completed()])

    conversation.thread.next_turn = WatchedTurn()
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    assert rendered.wait(WAIT_SECONDS)
