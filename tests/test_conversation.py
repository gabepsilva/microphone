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

from voice_codex.codex import (
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
    from voice_codex.domain import SentenceChunker

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
    from voice_codex.domain import SentenceChunker

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
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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


def test_interrupting_stops_the_active_turn_and_its_speech(monkeypatch) -> None:
    codex = FakeCodex()
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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
        def stream(self):
            rendered.set()
            return iter([delta("Answer."), turn_completed()])

    conversation.thread.next_turn = WatchedTurn()
    conversation.ingest("Them", "A question", respond=True, timestamp="T1")

    assert rendered.wait(WAIT_SECONDS)


def test_a_turn_that_ends_on_a_command_speaks_its_text_exactly_once() -> None:
    """The turn completes with no message open, so the close is a no-op.

    The flush that follows it is not: the chunker can still hold the tail of a
    message the command boundary closed. Speaking it twice, or not at all, are
    both audible.
    """
    from voice_codex.domain import SentenceChunker

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
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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
    monkeypatch.setattr("voice_codex.codex.Codex", lambda: codex)
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
