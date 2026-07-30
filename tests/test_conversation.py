"""Codex turn rendering and conversation lifecycle.

The Codex client is faked at its boundary. Turn events are built from the
real notification types with ``model_construct`` so the renderer's isinstance
dispatch is exercised against the shapes it will actually see.
"""

from __future__ import annotations

import os
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
    ReasoningSummaryPartAddedNotification,
    ReasoningSummaryTextDeltaNotification,
    ReasoningThreadItem,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)

from tagalong.codex import (
    REASONING_SUMMARY,
    WARMUP_PROMPT,
    CodexConversation,
    CodexSettings,
    CodexTurnRenderer,
    item_root,
    load_codex_sdk,
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


def reasoning_item():
    return ReasoningThreadItem.model_construct()


def summary_delta(text):
    return event(ReasoningSummaryTextDeltaNotification.model_construct(delta=text))


def summary_part(summary_index):
    return event(
        ReasoningSummaryPartAddedNotification.model_construct(
            summary_index=summary_index
        )
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
    """Record every presentation call in the order it arrived.

    Every method is spelled out rather than caught by ``__getattr__``: a
    catch-all fake records calls that do not exist, so a renamed or misspelled
    display call would be swallowed here instead of failing.
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, kwargs) if kwargs else (name, *args))

    def update(self, speaker, text):
        self._record("update", speaker, text)

    def commit(self, speaker, text):
        self._record("commit", speaker, text)

    def finish_turn(self, speaker):
        self._record("finish_turn", speaker)

    def close_speaker(self, speaker):
        self._record("close_speaker", speaker)

    def begin_codex(self):
        self._record("begin_codex")

    def codex_message_open(self, reply_to):
        self._record("codex_message_open", reply_to)

    def codex_delta(self, delta):
        self._record("codex_delta", delta)

    def codex_message_close(self):
        self._record("codex_message_close")

    def reasoning_started(self):
        self._record("reasoning_started")

    def reasoning_delta(self, delta):
        self._record("reasoning_delta", delta)

    def reasoning_completed(self):
        self._record("reasoning_completed")

    def command_started(self, command):
        self._record("command_started", command)

    def command_output(self, delta):
        self._record("command_output", delta)

    def command_completed(self, exit_code):
        self._record("command_completed", exit_code)

    def tool_called(self, server, tool):
        self._record("tool_called", server, tool)

    def tool_completed(self, status):
        self._record("tool_completed", status)

    def token_usage(self, total_tokens):
        self._record("token_usage", total_tokens)

    def error(self, message):
        self._record("error", message)

    def note(self, text):
        self._record("note", text)

    def set_codex(self, **fields):
        self._record("set_codex", **fields)

    def set_codex_catalog(self, models, efforts_by_model, default_effort_by_model):
        self._record(
            "set_codex_catalog", models, efforts_by_model, default_effort_by_model
        )

    def end_codex(self):
        self._record("end_codex")

    def names(self):
        return [call[0] for call in self.calls]


def render(events, chunker=None, reply_to="Them"):
    display = FakeDisplay()
    CodexTurnRenderer(display, reply_to, chunker).render(events)
    return display


def test_an_assistant_message_is_opened_streamed_and_closed() -> None:
    display = render(
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


def test_a_delta_without_a_start_still_opens_the_message() -> None:
    display = render([delta("Sudden text."), turn_completed()])

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Sudden text."),
        ("codex_message_close",),
    ]


def test_the_message_is_opened_once_across_many_deltas() -> None:
    display = render([delta("a"), delta("b"), delta("c")])

    assert display.names().count("codex_message_open") == 1


def test_reasoning_is_shown_as_its_own_section() -> None:
    display = render(
        [
            started(reasoning_item()),
            summary_part(0),
            summary_delta("Weighing the riddle."),
            finished(reasoning_item()),
            started(message_item()),
            delta("Nine."),
            finished(message_item()),
            turn_completed(),
        ],
    )

    assert display.calls == [
        ("reasoning_started",),
        ("reasoning_delta", "Weighing the riddle."),
        ("reasoning_completed",),
        ("codex_message_open", "Them"),
        ("codex_delta", "Nine."),
        ("codex_message_close",),
    ]


def test_every_summary_part_after_the_first_opens_a_paragraph() -> None:
    display = render(
        [
            started(reasoning_item()),
            summary_part(0),
            summary_delta("First thought."),
            summary_part(1),
            summary_delta("Second thought."),
            finished(reasoning_item()),
        ],
    )

    assert display.calls == [
        ("reasoning_started",),
        ("reasoning_delta", "First thought."),
        ("reasoning_delta", "\n\n"),
        ("reasoning_delta", "Second thought."),
        ("reasoning_completed",),
    ]


def test_reasoning_is_never_spoken() -> None:
    from tagalong.domain import SentenceChunker

    spoken: list[str] = []

    render(
        [
            started(reasoning_item()),
            summary_delta("Thinking out loud."),
            finished(reasoning_item()),
            delta("Nine."),
            turn_completed(),
        ],
        SentenceChunker(spoken.append),
    )

    assert spoken == ["Nine."]


def test_reasoning_does_not_time_the_reply() -> None:
    first_deltas = []
    display = FakeDisplay()
    renderer = CodexTurnRenderer(
        display, "Them", on_first_delta=lambda: first_deltas.append(display.names()[-1])
    )

    renderer.render(
        [
            started(reasoning_item()),
            summary_delta("Thinking out loud."),
            finished(reasoning_item()),
            delta("Nine."),
        ],
    )

    # The turn is timed from the first spoken word, so the only call that can
    # precede the report is the message being opened for it.
    assert first_deltas == ["codex_message_open"]


def test_reasoning_closes_the_open_message_before_it_is_shown() -> None:
    display = render(
        [
            delta("Let me think."),
            started(reasoning_item()),
            summary_delta("Still thinking."),
        ],
    )

    assert display.calls == [
        ("codex_message_open", "Them"),
        ("codex_delta", "Let me think."),
        ("codex_message_close",),
        ("reasoning_started",),
        ("reasoning_delta", "Still thinking."),
    ]


def test_a_command_closes_the_open_message_before_it_is_shown() -> None:
    display = render(
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


def test_a_tool_call_closes_the_open_message_before_it_is_shown() -> None:
    display = render(
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


def test_a_command_with_no_message_open_needs_no_close() -> None:
    display = render([started(command_item("pwd"))])

    assert display.calls == [("command_started", "pwd")]


def test_an_error_notification_is_shown() -> None:
    display = render([failure("model overloaded")])

    assert display.calls == [("error", "model overloaded")]


def test_token_usage_is_reported_only_when_the_turn_completes() -> None:
    display = render([delta("Hi."), usage(1234)])

    assert "token_usage" not in display.names()

    display = render([delta("Hi."), usage(1234), turn_completed()])

    assert display.calls[-1] == ("token_usage", 1234)


def test_the_latest_token_usage_wins() -> None:
    display = render([usage(10), usage(99), turn_completed()])

    assert display.calls == [("token_usage", 99)]


def test_a_turn_without_usage_reports_none() -> None:
    display = render([delta("Hi."), turn_completed()])

    assert "token_usage" not in display.names()


def test_a_wrapped_item_is_unwrapped_before_dispatch() -> None:
    display = render([started(SimpleNamespace(root=command_item("echo hi")))])

    assert display.calls == [("command_started", "echo hi")]


def test_item_root_passes_through_an_unwrapped_item() -> None:
    item = command_item()

    assert item_root(item) is item


def test_an_unrecognised_payload_is_ignored() -> None:
    display = render([event(SimpleNamespace(kind="something new"))])

    assert display.calls == []


def test_speech_is_flushed_at_every_command_boundary() -> None:
    from tagalong.domain import SentenceChunker

    spoken: list[str] = []
    chunker = SentenceChunker(spoken.append)

    render(
        [
            delta("First sentence. Second one"),
            started(command_item()),
            delta(" resumes after."),
            turn_completed(),
        ],
        chunker=chunker,
    )

    assert spoken == ["First sentence.", "Second one", "resumes after."]


def test_speech_is_flushed_when_the_turn_ends() -> None:
    from tagalong.domain import SentenceChunker

    spoken: list[str] = []
    render(
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


class RecordedConversation(CodexConversation):
    """A conversation that keeps handles on the fakes it was built against."""

    def __init__(self, codex, display, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_codex = codex
        self.fake_display = display


@pytest.fixture
def conversation(monkeypatch):
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    display = FakeDisplay()
    built = RecordedConversation(
        codex,
        display,
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )
    yield built
    built.close()


def test_the_thread_is_started_from_the_chosen_settings(quiet_conversation) -> None:
    """Every argument here changes what the model is or what it may do."""
    from openai_codex import ApprovalMode, Sandbox

    from tagalong.codex import CODEX_DEVELOPER_INSTRUCTIONS

    assert quiet_conversation.fake_codex.start_kwargs == {
        "model": "gpt-5.6-luna",
        "service_tier": None,
        "sandbox": Sandbox("read-only"),
        "approval_mode": ApprovalMode.deny_all,
        "cwd": os.getcwd(),
        "developer_instructions": CODEX_DEVELOPER_INSTRUCTIONS,
    }


def test_a_new_conversation_is_idle_warm_and_ready_to_guess(
    quiet_conversation,
) -> None:
    """The starting state each later assertion is a departure from."""
    from openai_codex import Sandbox

    assert quiet_conversation.model == "gpt-5.6-luna"
    assert quiet_conversation.reasoning_effort == "low"
    assert quiet_conversation.service_tier is None
    assert quiet_conversation.sandbox == Sandbox("read-only")
    assert quiet_conversation.thread.id == "thread-1"
    assert quiet_conversation.prefire_enabled is True
    # A fresh thread has not paid its slow first turn yet.
    assert quiet_conversation.warmup_pending is True
    assert quiet_conversation.shutdown_requested.is_set() is False
    assert quiet_conversation.active_turn is None
    assert quiet_conversation.speculation is None
    assert quiet_conversation.requested_model is None
    assert quiet_conversation.requested_reasoning_effort is None
    assert quiet_conversation.requests.empty()
    assert quiet_conversation.router.pending_context == []
    assert quiet_conversation.worker.daemon is True


def test_a_silent_session_still_builds_a_conversation(monkeypatch) -> None:
    """The speech pipeline is optional; nothing else may depend on it."""
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    display = FakeDisplay()

    built = CodexConversation(
        CodexSettings(
            sandbox="read-only",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            service_tier="priority",
            prefire=False,
        ),
        display,
    )

    assert built.tts is None
    assert built.prefire_enabled is False
    assert built.service_tier == "priority"
    assert codex.start_kwargs["service_tier"] == "priority"
    built.close()


def test_context_accumulates_until_a_reply_is_requested(conversation) -> None:
    conversation.ingest("Them", "A question", respond=False, timestamp="T1")
    conversation.ingest("User Voice", "And this", respond=True, timestamp="T2")

    request = conversation.requests.get(timeout=WAIT_SECONDS).request

    assert request.reply_to == "User Voice"
    assert conversation.context_entries(request) == [
        {"timestamp": "T1", "source": "Them", "text": "A question"},
        {"timestamp": "T2", "source": "User Voice", "text": "And this"},
    ]


def test_a_timestamp_is_generated_when_none_is_given(conversation) -> None:
    conversation.ingest("Them", "A question", respond=True)

    request = conversation.requests.get(timeout=WAIT_SECONDS).request

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


def test_a_fork_carries_the_whole_thread_configuration(conversation) -> None:
    """A fork that drops an argument silently changes what Codex may do."""
    from openai_codex import ApprovalMode, Sandbox

    from tagalong.codex import CODEX_DEVELOPER_INSTRUCTIONS

    conversation.request_model("gpt-5.6-nova")

    conversation._apply_pending_settings()

    assert conversation.fake_codex.fork_kwargs == {
        "model": "gpt-5.6-nova",
        "service_tier": None,
        "sandbox": Sandbox("read-only"),
        "approval_mode": ApprovalMode.deny_all,
        "developer_instructions": CODEX_DEVELOPER_INSTRUCTIONS,
    }
    assert ("set_codex", {"model": "gpt-5.6-nova", "thread": "thread-2"}) in (
        conversation.fake_display.calls
    )


def test_a_fork_re_arms_the_warm_up_and_leaves_nothing_pending(conversation) -> None:
    conversation.request_model("gpt-5.6-nova")
    conversation.request_reasoning_effort("high")

    conversation._apply_pending_settings()

    assert conversation.warmup_pending is True
    assert conversation.requested_model is None
    assert conversation.requested_reasoning_effort is None
    # A second pass has nothing left to do, so it must not fork again.
    conversation.fake_codex.forked_from = None
    conversation._apply_pending_settings()
    assert conversation.fake_codex.forked_from is None


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


def test_every_turn_asks_for_a_reasoning_summary(conversation) -> None:
    conversation.thread.next_turn = FakeTurn([delta("Answer."), turn_completed()])
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    conversation._run_codex(conversation.requests.get(timeout=WAIT_SECONDS))

    # Without asking, no reasoning is streamed at all and the section stays
    # empty however hard the model thought.
    assert conversation.thread.kwargs["summary"].root.value == REASONING_SUMMARY


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
    queued = conversation.requests.get(timeout=WAIT_SECONDS)

    conversation._run_codex(queued)

    assert ("error", "Codex error: stream broke") in conversation.fake_display.calls
    assert conversation.fake_display.names()[-1] == "end_codex"
    assert conversation.active_turn is None


def test_the_prompt_names_the_source_to_reply_to(conversation) -> None:
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")
    queued = conversation.requests.get(timeout=WAIT_SECONDS)

    conversation._run_codex(queued)
    prompt = conversation.thread.turns[-1]

    assert "Reply now to the latest Them input" in prompt
    assert '"text": "A question"' in prompt


def test_the_prompt_is_exactly_what_codex_is_asked(quiet_conversation) -> None:
    """The whole string, because every word of it is instruction.

    Asserting on fragments leaves the joins, the ordering and the closing
    sentence unpinned, and those are what tell Codex which entries are
    context and which one it is answering.
    """
    quiet_conversation.ingest("Them", "A question", respond=False, timestamp="T1")
    quiet_conversation.ingest("User Voice", "And mine", respond=True, timestamp="T2")
    request = quiet_conversation.requests.get(timeout=WAIT_SECONDS).request

    assert CodexConversation.build_prompt(request) == (
        "Transcript entries since the previous queued reply:\n"
        '[{"timestamp": "T1", "source": "Them", "text": "A question"}, '
        '{"timestamp": "T2", "source": "User Voice", "text": "And mine"}]\n\n'
        "Reply now to the latest User Voice input. "
        "Use the other entries as context."
    )


def test_a_prompt_keeps_non_ascii_speech_as_it_was_heard(quiet_conversation) -> None:
    """Escaping it would hand Codex \\u00e7 where a speaker said ç."""
    quiet_conversation.ingest("Them", "Ação sim", respond=True, timestamp="T1")
    request = quiet_conversation.requests.get(timeout=WAIT_SECONDS).request

    assert '"text": "Ação sim"' in CodexConversation.build_prompt(request)


def test_every_turn_is_started_under_the_chosen_constraints(conversation) -> None:
    """Effort, summary and sandbox are re-sent per turn, not just per thread."""
    from openai_codex import ApprovalMode, Sandbox
    from openai_codex.generated.v2_all import ReasoningEffort, ReasoningSummary

    conversation.thread.next_turn = FakeTurn([delta("Answer."), turn_completed()])
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    conversation._run_codex(conversation.requests.get(timeout=WAIT_SECONDS))

    assert conversation.thread.kwargs == {
        "effort": ReasoningEffort("low"),
        "summary": ReasoningSummary(REASONING_SUMMARY),
        "sandbox": Sandbox("read-only"),
        "approval_mode": ApprovalMode.deny_all,
    }


def test_interrupting_stops_the_active_turn_and_its_speech(monkeypatch) -> None:
    """The worker is stubbed out because this drives ``active_turn`` by hand: a
    live one warms the thread up on the same field and clears it when it is
    done, so the turn under test would be whichever of the two got there last.
    """
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    tts = SimpleNamespace(
        interrupted=0,
        closed=0,
        interrupt=lambda: None,
        close=lambda: None,
    )
    interrupts: list[str] = []
    tts.interrupt = lambda: interrupts.append("tts")
    tts.close = lambda: interrupts.append("closed")
    built = CodexConversation(
        CodexSettings(
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
        # A generator, like FakeTurn and like the SDK stream: ``_warm_up``
        # closes what it was given, and a bare iterator cannot be closed.
        def stream(self):
            rendered.set()
            yield from [delta("Answer."), turn_completed()]

    conversation.thread.next_turn = WatchedTurn()
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    assert rendered.wait(WAIT_SECONDS)


def test_the_first_word_of_a_turn_is_reported_once() -> None:
    """The latency estimate is one sample per turn, not one per delta.

    Reporting every delta would average the whole reply into a figure the
    pre-fire schedule reads as time-to-first-word, and aim its guesses late.
    """
    reports = []
    renderer = CodexTurnRenderer(
        FakeDisplay(), "Them", on_first_delta=lambda: reports.append(1)
    )

    renderer.render([delta("First "), delta("second "), delta("third.")])

    assert reports == [1]


def test_a_guess_carries_the_moment_it_was_heard(quiet_conversation) -> None:
    """Codex reads the timestamps as conversational timing."""
    assert quiet_conversation.prefire("User Voice", "a question", timestamp="T1")

    request = quiet_conversation.requests.get(timeout=WAIT_SECONDS).request

    assert [entry.timestamp for entry in request.entries] == ["T1"]


def test_a_guess_that_was_adopted_is_marked_as_committed(quiet_conversation) -> None:
    """The worker reads this to tell a reply from a guess still in flight."""
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    assert quiet_conversation.commit_prefire("User Voice") is True

    assert queued.speculation.committed is True
    assert queued.speculation.abandoned is False


def test_a_retreat_re_answers_the_very_same_request(quiet_conversation) -> None:
    """The second attempt is the reply; sending anything else answers nothing."""
    refusal = failure('{"param": "reasoning.effort", "code": "unsupported_value"}')
    quiet_conversation.reasoning_effort = "high"
    quiet_conversation.thread.next_turn = FakeTurn([refusal, turn_completed()])
    quiet_conversation.ingest("Them", "A question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    first, second = quiet_conversation.thread.turns
    assert first == second == CodexConversation.build_prompt(queued.request)
    assert ("set_codex", {"effort": "low"}) in quiet_conversation.fake_display.calls


def test_the_retry_still_answers_the_source_that_asked(quiet_conversation) -> None:
    """A retry that loses the source labels the reply as answering nobody."""

    class RefusingOnceTurn(FakeTurn):
        """Refuses the effort on the first stream, answers on the second."""

        def __init__(self):
            super().__init__()
            self.streams = 0

        def stream(self):
            self.streams += 1
            if self.streams == 1:
                yield failure('{"param": "reasoning.effort", "unsupported_value"}')
                yield turn_completed()
                return
            yield delta("Nine.")
            yield turn_completed()

    quiet_conversation.reasoning_effort = "high"
    quiet_conversation.thread.next_turn = RefusingOnceTurn()
    quiet_conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    quiet_conversation._run_codex(quiet_conversation.requests.get(timeout=WAIT_SECONDS))

    assert ("codex_message_open", "Them") in quiet_conversation.fake_display.calls
    assert ("codex_delta", "Nine.") in quiet_conversation.fake_display.calls


def test_the_warm_up_turn_can_be_reached_while_it_runs(quiet_conversation) -> None:
    """Shutdown interrupts through ``active_turn``, mid-warm-up included."""
    seen = []

    class WatchedTurn(FakeTurn):
        def stream(self):
            seen.append(quiet_conversation.active_turn)
            yield turn_completed()

    turn = WatchedTurn()
    quiet_conversation.thread.next_turn = turn

    quiet_conversation._warm_up()

    assert seen == [turn]
    assert quiet_conversation.active_turn is None


def test_closing_survives_a_turn_that_refuses_to_be_interrupted(monkeypatch) -> None:
    """A wedged turn must not take the shutdown down with it."""
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
    )

    class RefusingTurn(FakeTurn):
        def interrupt(self):
            raise RuntimeError("the turn is wedged")

    conversation.active_turn = RefusingTurn()

    conversation.close()

    assert codex.closed is True


def test_a_guess_survives_a_turn_that_refuses_to_be_interrupted(
    quiet_conversation,
) -> None:
    class RefusingTurn(FakeTurn):
        def interrupt(self):
            raise RuntimeError("the turn is wedged")

    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")
    quiet_conversation.speculation.turn = RefusingTurn()

    assert quiet_conversation.cancel_prefire("User Voice") is True
    assert quiet_conversation.speculation is None


def test_closing_does_not_wait_out_a_wedged_worker(monkeypatch) -> None:
    """Shutdown is bounded: a worker that never returns cannot hang the exit."""
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    released = threading.Event()
    monkeypatch.setattr(
        CodexConversation, "_worker", lambda self: released.wait(WAIT_SECONDS)
    )
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
    )

    try:
        conversation.close()
        assert conversation.worker.is_alive() is True
    finally:
        released.set()


def parked(conversation):
    """Report each time the live worker settles down to wait on the queue.

    Waiting for that is what makes these deterministic: acting before the
    worker has reached the queue lets it leave through the loop condition
    instead, which proves nothing about the wait itself.
    """
    waiting = threading.Event()
    original = conversation.requests.get

    def watched(*args, **kwargs):
        waiting.set()
        return original(*args, **kwargs)

    conversation.requests.get = watched
    return waiting


def test_the_worker_gives_up_the_queue_to_notice_a_shutdown(conversation) -> None:
    """Nothing is queued and no sentinel posted; only the timeout wakes it.

    ``close`` does post a sentinel, so an unbounded wait would still end a
    tidy shutdown. This is the untidy one: a flag set by someone who never
    queues anything.
    """
    waiting = parked(conversation)
    assert waiting.wait(WAIT_SECONDS)

    conversation.shutdown_requested.set()
    conversation.worker.join(WAIT_SECONDS)

    assert conversation.worker.is_alive() is False


def test_an_empty_queue_does_not_end_the_worker(conversation) -> None:
    """A quiet moment is not a shutdown: the loop waits again and keeps serving."""
    rendered = threading.Event()

    class WatchedTurn(FakeTurn):
        def stream(self):
            rendered.set()
            yield turn_completed()

    waiting = parked(conversation)
    assert waiting.wait(WAIT_SECONDS)
    waiting.clear()
    # A second visit to the queue means the first empty one was survived.
    assert waiting.wait(WAIT_SECONDS)

    conversation.thread.next_turn = WatchedTurn()
    conversation.ingest("Them", "A late question", respond=True, timestamp="T1")

    assert rendered.wait(WAIT_SECONDS)


def test_the_thread_it_opened_is_announced(monkeypatch, capsys) -> None:
    """The one line printed before the interface exists, for a failed start."""
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)

    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
    )

    assert (
        capsys.readouterr().out
        == "Codex App Server ready. Conversation thread: thread-1\n"
    )
    conversation.close()


def test_a_failed_fork_puts_the_displayed_settings_back(conversation) -> None:
    """The picker moved on its own; a refused switch has to move it back."""
    conversation.fake_codex.fork_error = RuntimeError("model unavailable")
    conversation.request_model("gpt-5.6-nova")

    conversation._apply_pending_settings()

    assert ("set_codex", {"model": "gpt-5.6-luna", "effort": "low"}) in (
        conversation.fake_display.calls
    )


def test_a_fork_keeps_the_service_tier_the_session_pays_for(monkeypatch) -> None:
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            service_tier="priority",
        ),
        FakeDisplay(),
    )

    conversation.request_model("gpt-5.6-nova")
    conversation._apply_pending_settings()

    assert codex.fork_kwargs["service_tier"] == "priority"
    conversation.close()


def test_the_reply_is_addressed_to_the_source_that_asked(conversation) -> None:
    """The renderer labels the row with this; the wrong source misattributes it."""
    conversation.thread.next_turn = FakeTurn([delta("Answer."), turn_completed()])
    conversation.ingest("User Voice", "A question", respond=True, timestamp="T1")

    conversation._run_codex(conversation.requests.get(timeout=WAIT_SECONDS))

    assert ("codex_message_open", "User Voice") in conversation.fake_display.calls


def test_a_running_turn_can_be_reached_while_it_streams(quiet_conversation) -> None:
    """Interrupting depends on it: an unpublished turn cannot be stopped."""
    seen = []
    turn = FakeTurn([delta("Answer."), turn_completed()])
    quiet_conversation.thread.next_turn = turn
    original = quiet_conversation._stream_turn

    def watched(streamed_turn, reply_to):
        seen.append((quiet_conversation.active_turn, reply_to))
        return original(streamed_turn, reply_to)

    quiet_conversation._stream_turn = watched
    quiet_conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    quiet_conversation._run_codex(quiet_conversation.requests.get(timeout=WAIT_SECONDS))

    assert seen == [(turn, "Them")]
    # And it is let go of once the turn is over, so a later interrupt cannot
    # reach a turn that has already finished.
    assert quiet_conversation.active_turn is None


def test_a_refusal_names_the_effort_it_retreated_from(quiet_conversation) -> None:
    quiet_conversation.reasoning_effort = "high"

    assert quiet_conversation._retreat_effort() is True

    assert ("note", "Codex refused reasoning effort 'high' → low") in (
        quiet_conversation.fake_display.calls
    )


def test_the_warm_up_publishes_the_turn_it_is_waiting_on(quiet_conversation) -> None:
    """Shutdown interrupts through ``active_turn``; a warm-up must be reachable."""
    seen = []
    turn = FakeTurn([delta("ready"), turn_completed()])
    quiet_conversation.thread.next_turn = turn
    quiet_conversation.requests.put_nowait(None)  # stop the warm-up at once

    quiet_conversation._warm_up()

    seen.append(quiet_conversation.active_turn)
    assert turn.interrupted == 1
    assert seen == [None]
    assert quiet_conversation.warmup_pending is False


def test_an_idle_worker_keeps_checking_for_shutdown(quiet_conversation) -> None:
    """The queue wait must expire, or close() would block until a turn arrives.

    ``close`` posts a sentinel, so a worker parked on an unbounded ``get``
    would still wake — but only for a session that is shut down cleanly. The
    timeout is what makes the loop notice a shutdown it was not handed.
    """
    quiet_conversation.shutdown_requested.set()

    # The loop condition is checked before the queue is, so a worker started
    # against a shut-down conversation returns rather than waiting at all.
    quiet_conversation._worker()

    assert quiet_conversation.requests.empty()


def test_closing_stops_the_turn_the_speech_and_the_client(monkeypatch) -> None:
    """Shutdown is a sequence; skipping any step leaves something running.

    The worker is stubbed out because this drives ``active_turn`` by hand: a
    live one warms the thread up on the same field and clears it when it is
    done, so the turn under test would be whichever of the two got there last.
    """
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    closed_speech = []
    tts = SimpleNamespace(
        interrupt=lambda: closed_speech.append("interrupt"),
        close=lambda: closed_speech.append("close"),
    )
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
        tts=tts,
    )
    turn = FakeTurn()
    conversation.active_turn = turn

    conversation.close()

    assert conversation.shutdown_requested.is_set() is True
    assert turn.interrupted == 1
    assert conversation.worker.is_alive() is False
    assert codex.closed is True
    assert closed_speech == ["close"]


def test_closing_a_silent_session_has_no_speech_to_close(conversation) -> None:
    """The optional pipeline must not be assumed present at shutdown."""
    conversation.tts = None

    conversation.close()

    assert conversation.fake_codex.closed is True


def test_a_turn_that_ends_on_a_command_speaks_its_text_exactly_once() -> None:
    """The turn completes with no message open, so the close is a no-op.

    The flush that follows it is not: the chunker can still hold the tail of a
    message the command boundary closed. Speaking it twice, or not at all, are
    both audible.
    """
    from tagalong.domain import SentenceChunker

    spoken: list[str] = []
    display = render(
        [
            delta("Running it now"),
            started(command_item()),
            finished(command_item()),
            turn_completed(),
        ],
        chunker=SentenceChunker(spoken.append),
    )

    assert spoken == ["Running it now"]
    closes = [call for call in display.calls if call[0] == "codex_message_close"]
    assert len(closes) == 1


# --------------------------------------------------------------------------
# Answering before the silence window closes
# --------------------------------------------------------------------------


@pytest.fixture
def quiet_conversation(monkeypatch):
    """A conversation whose worker never runs, so turns are driven explicitly.

    The speculation lifecycle is about what the queue and the router hold at
    each moment, and a live worker would consume both before an assertion
    could see them.
    """
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    display = FakeDisplay()
    built = RecordedConversation(
        codex,
        display,
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        display,
    )
    yield built
    built.close()


def test_a_prefired_turn_is_queued_without_consuming_the_context(
    quiet_conversation,
) -> None:
    quiet_conversation.ingest("Them", "context", respond=False, timestamp="T1")

    assert quiet_conversation.prefire("User Voice", "a question", timestamp="T2")

    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)
    assert [entry.text for entry in queued.request.entries] == [
        "context",
        "a question",
    ]
    assert queued.speculation is not None
    assert len(quiet_conversation.router.pending_context) == 2


def test_a_closed_session_reports_that_it_did_not_guess(quiet_conversation) -> None:
    """The caller submits the turn itself when told False; a wrong True is silence."""
    quiet_conversation.shutdown_requested.set()

    assert quiet_conversation.prefire("User Voice", "a question", timestamp="T1") is (
        False
    )
    assert quiet_conversation.requests.empty()


def test_a_second_guess_is_refused_while_one_is_outstanding(
    quiet_conversation,
) -> None:
    assert quiet_conversation.prefire("User Voice", "first", timestamp="T1") is True

    assert quiet_conversation.prefire("User Voice", "second", timestamp="T2") is False
    assert quiet_conversation.requests.qsize() == 1


def test_committing_a_prefired_turn_consumes_its_context(quiet_conversation) -> None:
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")

    assert quiet_conversation.commit_prefire("User Voice")

    assert quiet_conversation.router.pending_context == []
    assert quiet_conversation.speculation is None


def test_cancelling_a_prefired_turn_keeps_its_words_for_the_next_request(
    quiet_conversation,
) -> None:
    quiet_conversation.prefire("User Voice", "half a thought", timestamp="T1")

    assert quiet_conversation.cancel_prefire("User Voice")

    quiet_conversation.requests.get(timeout=WAIT_SECONDS)
    quiet_conversation.ingest("User Voice", "the rest", respond=True, timestamp="T2")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)
    assert [entry.text for entry in queued.request.entries] == [
        "half a thought",
        "the rest",
    ]


def test_cancelling_interrupts_the_turn_that_was_already_running(
    quiet_conversation,
) -> None:
    turn = FakeTurn()
    quiet_conversation.prefire("User Voice", "half a thought", timestamp="T1")
    quiet_conversation.speculation.turn = turn

    quiet_conversation.cancel_prefire("User Voice")

    assert turn.interrupted == 1


def test_another_speakers_cancel_leaves_this_speculation_alone(
    quiet_conversation,
) -> None:
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")

    assert not quiet_conversation.cancel_prefire("Them")
    assert quiet_conversation.speculation is not None


def test_a_committed_turn_can_no_longer_be_cancelled(quiet_conversation) -> None:
    """The reply is already the answer; cutting it off would answer nothing."""
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")
    quiet_conversation.commit_prefire("User Voice")

    assert not quiet_conversation.cancel_prefire("User Voice")


def test_a_cancelled_turn_cannot_be_committed(quiet_conversation) -> None:
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")
    quiet_conversation.cancel_prefire("User Voice")

    assert not quiet_conversation.commit_prefire("User Voice")


def test_only_one_turn_is_guessed_at_a_time(quiet_conversation) -> None:
    quiet_conversation.prefire("User Voice", "first", timestamp="T1")

    assert not quiet_conversation.prefire("Them", "second", timestamp="T2")


def test_a_settled_request_supersedes_an_outstanding_guess(
    quiet_conversation,
) -> None:
    """Both running would answer the same transcript twice."""
    turn = FakeTurn()
    quiet_conversation.prefire("User Voice", "half a thought", timestamp="T1")
    quiet_conversation.speculation.turn = turn

    quiet_conversation.ingest("User Text", "never mind", respond=True, timestamp="T2")

    assert turn.interrupted == 1
    assert quiet_conversation.speculation is None


def test_prefiring_is_refused_when_the_session_turned_it_off(monkeypatch) -> None:
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    display = FakeDisplay()
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            prefire=False,
        ),
        display,
    )

    try:
        assert not conversation.prefire("User Voice", "a question")
        assert conversation.requests.empty()
    finally:
        conversation.close()


def test_a_closed_conversation_does_not_guess(quiet_conversation) -> None:
    quiet_conversation.close()

    assert not quiet_conversation.prefire("User Voice", "a question")


def test_a_turn_abandoned_before_it_started_is_interrupted_at_once(
    quiet_conversation,
) -> None:
    """Cancelled while still queued: the worker must not render a dead turn."""
    turn = FakeTurn(events=[delta("Half an answer"), turn_completed()])
    quiet_conversation.fake_codex.thread.next_turn = turn
    quiet_conversation.prefire("User Voice", "half a thought", timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)
    quiet_conversation.cancel_prefire("User Voice")

    quiet_conversation._run_codex(queued)

    assert turn.interrupted == 1
    assert "codex_delta" not in quiet_conversation.fake_display.names()


# --------------------------------------------------------------------------
# Paying the first turn's cost before anyone is waiting
# --------------------------------------------------------------------------


def test_a_new_thread_is_warmed_before_the_first_real_turn(quiet_conversation) -> None:
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(
        events=[delta("ready"), turn_completed()]
    )

    quiet_conversation._warm_up()

    assert quiet_conversation.fake_codex.thread.turns == [WARMUP_PROMPT]
    assert quiet_conversation.warmup_pending is False
    # Nothing is shown or spoken: the answer exists only to have been asked for.
    assert quiet_conversation.fake_display.names() == []


def test_the_warm_up_yields_as_soon_as_there_is_real_work(quiet_conversation) -> None:
    """A speaker who starts talking during startup waits for nobody."""
    turn = FakeTurn(events=[delta("rea"), delta("dy"), turn_completed()])
    quiet_conversation.fake_codex.thread.next_turn = turn
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")

    quiet_conversation._warm_up()

    assert turn.interrupted == 1


def test_a_failed_warm_up_is_not_reported(quiet_conversation) -> None:
    """The turn it was warming will complain loudly enough on its own."""
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(
        error=RuntimeError("no thread")
    )

    quiet_conversation._warm_up()

    assert quiet_conversation.fake_display.names() == []
    assert quiet_conversation.warmup_pending is False


def test_switching_models_re_arms_the_warm_up(quiet_conversation) -> None:
    """A fork is a new thread, and carries a new thread's slow first turn."""
    quiet_conversation.warmup_pending = False
    quiet_conversation.requested_model = "gpt-5.6-sol"

    quiet_conversation._apply_pending_settings()

    assert quiet_conversation.warmup_pending is True


def test_a_failed_fork_does_not_arm_the_warm_up(quiet_conversation) -> None:
    quiet_conversation.warmup_pending = False
    quiet_conversation.fake_codex.fork_error = RuntimeError("nope")
    quiet_conversation.requested_model = "gpt-5.6-sol"

    quiet_conversation._apply_pending_settings()

    assert quiet_conversation.warmup_pending is False


# --------------------------------------------------------------------------
# An effort the model will not take
# --------------------------------------------------------------------------


REFUSAL = '{"error": {"code": "unsupported_value", "param": "reasoning.effort"}}'


def test_a_refused_effort_retreats_and_answers_anyway(quiet_conversation) -> None:
    """A refusal produces no reply at all, so a silent session is the failure."""
    quiet_conversation.reasoning_effort = "none"
    refused = FakeTurn(events=[failure(REFUSAL), turn_completed()])
    answered = FakeTurn(events=[delta("Here it is."), turn_completed()])
    turns = iter([refused, answered])
    quiet_conversation.fake_codex.thread.turn = lambda prompt, **kwargs: (
        quiet_conversation.fake_codex.thread.turns.append(prompt) or next(turns)
    )
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    assert quiet_conversation.reasoning_effort == "low"
    assert len(quiet_conversation.fake_codex.thread.turns) == 2
    assert ("codex_delta", "Here it is.") in quiet_conversation.fake_display.calls


def test_an_effort_refused_at_the_fallback_is_not_retried(quiet_conversation) -> None:
    """Retrying the same effort would ask the same refused question forever."""
    quiet_conversation.reasoning_effort = "low"
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(
        events=[failure(REFUSAL), turn_completed()]
    )
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    assert len(quiet_conversation.fake_codex.thread.turns) == 1


def test_an_unrelated_error_does_not_change_the_effort(quiet_conversation) -> None:
    quiet_conversation.reasoning_effort = "none"
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(
        events=[failure("the network went away"), turn_completed()]
    )
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    assert quiet_conversation.reasoning_effort == "none"
    assert len(quiet_conversation.fake_codex.thread.turns) == 1


# --------------------------------------------------------------------------
# Learning how long a reply takes to start
# --------------------------------------------------------------------------


def test_the_time_to_the_first_word_is_recorded(quiet_conversation) -> None:
    seeded = quiet_conversation.latency.estimate
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(
        events=[delta("Here"), delta(" it is."), turn_completed()]
    )
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    # A fake turn answers instantly, so the estimate can only have fallen.
    assert quiet_conversation.latency.estimate < seeded


def test_a_turn_that_never_speaks_teaches_nothing(quiet_conversation) -> None:
    seeded = quiet_conversation.latency.estimate
    quiet_conversation.fake_codex.thread.next_turn = FakeTurn(events=[turn_completed()])
    quiet_conversation.ingest("Them", "a question", respond=True, timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    assert quiet_conversation.latency.estimate == seeded


def test_a_cancelled_guess_stops_the_speech_it_had_already_started(
    monkeypatch,
) -> None:
    """The half-spoken reply must stop with the turn that was producing it."""
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    interrupted: list[int] = []
    tts = SimpleNamespace(
        interrupt=lambda: interrupted.append(1),
        close=lambda: None,
        begin_turn=lambda: None,
        speak=lambda text: None,
    )
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
        tts,
    )
    try:
        conversation.prefire("User Voice", "half a thought", timestamp="T1")

        conversation.cancel_prefire("User Voice")

        assert interrupted == [1]
    finally:
        conversation.close()


def test_a_started_turn_is_attached_to_the_speculation_that_owns_it(
    quiet_conversation,
) -> None:
    """Attaching is what lets a later cancel reach a turn already streaming."""
    turn = FakeTurn(events=[delta("Here it is."), turn_completed()])
    quiet_conversation.fake_codex.thread.next_turn = turn
    quiet_conversation.prefire("User Voice", "a question", timestamp="T1")
    queued = quiet_conversation.requests.get(timeout=WAIT_SECONDS)

    quiet_conversation._run_codex(queued)

    assert queued.speculation.turn is turn
    assert ("codex_delta", "Here it is.") in quiet_conversation.fake_display.calls


def test_a_turn_speaks_through_the_session_speech(monkeypatch) -> None:
    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    spoken: list[str] = []
    turns: list[int] = []
    tts = SimpleNamespace(
        interrupt=lambda: None,
        close=lambda: None,
        begin_turn=lambda: turns.append(1),
        speak=spoken.append,
    )
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only", model="gpt-5.6-luna", reasoning_effort="low"
        ),
        FakeDisplay(),
        tts,
    )
    try:
        codex.thread.next_turn = FakeTurn(
            events=[delta("Here it is. "), turn_completed()]
        )
        conversation.ingest("Them", "a question", respond=True, timestamp="T1")
        queued = conversation.requests.get(timeout=WAIT_SECONDS)

        conversation._run_codex(queued)

        assert turns == [1]
        assert spoken == ["Here it is."]
    finally:
        conversation.close()


def test_a_warm_up_whose_turn_will_not_start_is_given_up_on(
    quiet_conversation,
) -> None:
    def refuse(_prompt, **_kwargs):
        raise RuntimeError("thread is gone")

    quiet_conversation.fake_codex.thread.turn = refuse

    quiet_conversation._warm_up()

    assert quiet_conversation.warmup_pending is False
    assert quiet_conversation.active_turn is None
