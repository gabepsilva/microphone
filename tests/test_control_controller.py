from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from tagalong.control.actions import CATALOG, PROTOCOL_VERSION, Scope
from tagalong.control.actors import agent, local_user
from tagalong.control.controller import (
    IDEMPOTENCY_MEMORY,
    SUPERSEDED_MEMORY,
    UNPUBLISHED_MEMORY,
    Controller,
    Request,
)
from tagalong.control.outcomes import (
    Accepted,
    Applied,
    EffectFailed,
    Failed,
    Inapplicable,
    Rejected,
    Rejection,
    Superseded,
)
from tagalong.control.state import (
    AppState,
    Effect,
    Selection,
    with_desired,
    with_effective,
)
from tagalong.domain import TurnSilence

OWNER = local_user()
STAMP = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def selecting(field: str):
    """A device selection, shaped the way an adapter has to write one.

    The handler records what was asked for and returns; opening a capture
    device is the reconciler's job, and doing it here would hold the writer.
    """

    def settle(state: AppState, effective: object) -> AppState:
        # The adapter that started the work is the one place that knows what
        # the reconciler reports back, so it is where the value is narrowed.
        assert effective is None or isinstance(effective, str)
        return with_effective(state, field, effective)

    def handle(request, state):
        return Effect.pending(
            with_desired(state, field, request.payload["name"]), settle
        )

    return handle


def setting_turn_silence(request, state):
    """A synchronous setting that coerces, using the session's own bounds."""
    seconds = TurnSilence.clamp(request.payload["seconds"])
    return Effect.applied(replace(state, turn_silence=seconds), seconds)


def setting_tts(request, state):
    enabled = request.payload["enabled"]
    return Effect.applied(replace(state, tts_enabled=enabled), enabled)


def refusing(message: str):
    def handle(_request, _state):
        raise Inapplicable(message)

    return handle


def breaking(message: str):
    def handle(_request, _state):
        raise EffectFailed(message)

    return handle


def never_called(_request, _state):
    raise AssertionError("the handler ran for a request that should not reach it")


def sending(request, state):
    """A message, whose effective value is the attachments it went out with."""
    return Effect.applied(state, request.payload["images"])


def wired() -> Controller:
    """A controller with the handlers these tests dispatch against."""
    controller = Controller()
    controller.register("message.send", sending)
    controller.register("microphone.select", selecting("microphone"))
    controller.register("audio_stream.select", selecting("audio_stream"))
    controller.register("turn_silence.set", setting_turn_silence)
    controller.register("tts.set_enabled", setting_tts)
    return controller


def events(subscription) -> list[tuple[str, dict]]:
    return [(event.name, dict(event.payload)) for event in subscription.drain()]


# -- discovery ----------------------------------------------------------


def test_a_snapshot_carries_the_cursor_its_state_was_read_at() -> None:
    controller = wired()
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    snapshot = controller.snapshot()

    assert snapshot.state.tts_enabled is False
    assert snapshot.sequence == 2  # state.changed, then action.applied
    assert snapshot.protocol_version == PROTOCOL_VERSION
    assert snapshot.instance


def test_subscribing_leaves_no_gap_between_the_snapshot_and_the_stream() -> None:
    controller = wired()

    snapshot, subscription = controller.subscribe()
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert snapshot.state.tts_enabled is True
    assert [name for name, _ in events(subscription)] == [
        "state.changed",
        "action.applied",
    ]


def test_capabilities_answer_for_the_actor_that_asked() -> None:
    """The catalog is static; who may call an entry is not."""
    controller = wired()
    reader = agent("notes-bot", {Scope.TRANSCRIPT})

    entries = controller.capabilities(reader)

    assert [entry.action.id for entry in entries] == [spec.id for spec in CATALOG]
    allowed = {entry.action.id for entry in entries if entry.allowed}
    assert allowed == {"transcript.append", "transcript.save"}
    assert all(entry.allowed for entry in controller.capabilities(OWNER))


def test_capabilities_deny_agent_quit_even_with_the_session_scope() -> None:
    controller = wired()
    caller = agent("bot", {Scope.SESSION})

    allowed = {
        entry.action.id: entry.allowed for entry in controller.capabilities(caller)
    }

    assert allowed["session.interrupt"] is True
    assert allowed["session.new"] is True
    assert allowed["session.quit"] is False


def test_an_unhandled_action_stays_in_the_catalog() -> None:
    """It is inapplicable in this session, not unknown to the protocol."""
    controller = wired()

    outcome = controller.dispatch("session.new", actor=OWNER)

    assert outcome == Rejected(
        "req-1", Rejection.INAPPLICABLE, "session.new is not available in this session"
    )
    assert {entry.action.id for entry in controller.capabilities(OWNER)} >= {
        "session.new"
    }


def test_a_handler_can_only_be_registered_for_a_catalog_action() -> None:
    with pytest.raises(KeyError, match=r"no such action: microphone\.explode"):
        wired().register("microphone.explode", never_called)


def test_registered_lists_only_actions_with_handlers() -> None:
    controller = Controller()
    assert controller.registered() == frozenset()
    controller.register("tts.set_enabled", setting_tts)
    assert controller.registered() == frozenset({"tts.set_enabled"})


# -- refusals -----------------------------------------------------------


def test_an_action_outside_the_catalog_is_unknown() -> None:
    outcome = wired().dispatch("microphone.explode", actor=OWNER)

    assert outcome == Rejected(
        "req-1", Rejection.UNKNOWN_ACTION, "no such action: microphone.explode"
    )


def test_an_actor_without_the_scope_never_reaches_the_handler() -> None:
    controller = wired()
    controller.register("microphone.select", never_called)
    reader = agent("notes-bot", {Scope.TRANSCRIPT})

    outcome = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=reader)

    assert outcome == Rejected(
        "req-1", Rejection.FORBIDDEN, "notes-bot does not hold the audio scope"
    )
    assert controller.state.microphone == Selection()


def test_a_payload_that_does_not_match_the_schema_never_reaches_the_handler() -> None:
    controller = wired()
    controller.register("turn_silence.set", never_called)

    outcome = controller.dispatch("turn_silence.set", {"seconds": "soon"}, actor=OWNER)

    assert outcome == Rejected(
        "req-1",
        Rejection.INVALID,
        "turn_silence.set: seconds: expected a number",
    )


def test_a_handler_may_refuse_a_valid_request_the_session_cannot_honour() -> None:
    controller = wired()
    controller.register("microphone.select", refusing("no such microphone: Yeti"))

    outcome = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)

    assert outcome == Rejected(
        "req-1", Rejection.INAPPLICABLE, "no such microphone: Yeti"
    )
    assert controller.state == AppState()


def test_a_refusal_changes_no_state_but_is_still_recorded() -> None:
    """An audit trail of successes only cannot show an agent being turned away."""
    controller = wired()
    _, subscription = controller.subscribe()

    controller.dispatch("turn_silence.set", {"seconds": "soon"}, actor=OWNER)

    assert events(subscription) == [
        (
            "action.rejected",
            {
                "request_id": "req-1",
                "action": "turn_silence.set",
                "actor": "local",
                "reason": Rejection.INVALID,
                "detail": "turn_silence.set: seconds: expected a number",
            },
        )
    ]


def test_a_request_that_was_tried_and_broke_is_not_a_refusal() -> None:
    """A retry is meaningful after a failure and pointless after a refusal."""
    controller = wired()
    controller.register("tts.set_enabled", breaking("speech engine is gone"))
    _, subscription = controller.subscribe()

    outcome = controller.dispatch("tts.set_enabled", {"enabled": True}, actor=OWNER)

    assert outcome == Failed("req-1", "speech engine is gone")
    assert events(subscription) == [
        (
            "action.failed",
            {
                "request_id": "req-1",
                "action": "tts.set_enabled",
                "actor": "local",
                "detail": "speech engine is gone",
            },
        )
    ]


def test_a_handler_is_told_who_asked_and_exactly_what_they_asked_for() -> None:
    """Provenance comes from the dispatch, never from something in the payload."""
    controller = Controller()
    seen: list[Request] = []

    def recording(request, state):
        seen.append(request)
        return Effect.applied(state, None)

    controller.register("tts.set_enabled", recording)

    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    (request,) = seen
    assert request.id == "req-1"
    assert request.action.id == "tts.set_enabled"
    assert request.actor is OWNER
    assert request.payload == {"enabled": False}


# -- the immediate path -------------------------------------------------


def test_a_finished_action_answers_with_what_the_session_actually_holds() -> None:
    """A slider showing 45 while the session runs 30 is the failure here."""
    controller = wired()

    outcome = controller.dispatch("turn_silence.set", {"seconds": 45.0}, actor=OWNER)

    assert outcome == Applied("req-1", 30.0)
    assert controller.state.turn_silence == 30.0


def test_state_is_published_before_the_outcome_that_describes_it() -> None:
    controller = wired()
    _, subscription = controller.subscribe()

    controller.dispatch("turn_silence.set", {"seconds": 45.0}, actor=OWNER)

    assert events(subscription) == [
        ("state.changed", {"turn_silence": 30.0}),
        (
            "action.applied",
            {
                "request_id": "req-1",
                "action": "turn_silence.set",
                "actor": "local",
                "effective": 30.0,
            },
        ),
    ]


def test_only_the_fields_that_changed_are_published() -> None:
    controller = wired()
    _, subscription = controller.subscribe()

    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert events(subscription)[0] == ("state.changed", {"tts_enabled": False})


def test_an_action_that_changes_nothing_publishes_no_state_event() -> None:
    controller = wired()
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)
    _, subscription = controller.subscribe()

    outcome = controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert outcome == Applied("req-2", False)
    assert [name for name, _ in events(subscription)] == ["action.applied"]


# -- the accepted path --------------------------------------------------


def test_a_selection_is_accepted_before_the_device_is_open() -> None:
    """The interface has to keep drawing while a capture model loads."""
    controller = wired()
    _, subscription = controller.subscribe()

    outcome = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)

    assert outcome == Accepted("req-1")
    assert controller.state.microphone == Selection(desired="Yeti", effective=None)
    assert events(subscription) == [
        ("state.changed", {"microphone": Selection("Yeti", None)}),
        (
            "action.accepted",
            {"request_id": "req-1", "action": "microphone.select", "actor": "local"},
        ),
    ]


def test_a_settled_selection_reports_what_the_device_became() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    _, subscription = controller.subscribe()

    outcome = controller.settle(accepted.request_id, "Yeti")

    assert outcome == Applied("req-1", "Yeti")
    assert controller.state.microphone == Selection(desired="Yeti", effective="Yeti")
    assert events(subscription) == [
        ("state.changed", {"microphone": Selection("Yeti", "Yeti")}),
        (
            "action.applied",
            {
                "request_id": "req-1",
                "action": "microphone.select",
                "actor": "local",
                "effective": "Yeti",
            },
        ),
    ]


def test_settling_on_nothing_still_says_what_it_settled_on() -> None:
    """Deselecting reaches None, which an omitted key could not express."""
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": None}, actor=OWNER)
    _, subscription = controller.subscribe()

    controller.settle(accepted.request_id, None)

    applied = dict(events(subscription)[-1][1])
    assert applied["effective"] is None
    assert controller.state.microphone == Selection(desired=None, effective=None)


def test_a_failed_selection_keeps_what_was_asked_for() -> None:
    """Whether to give up is the adapter's decision; one rule here would
    overrule whichever adapter disagreed with it."""
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    _, subscription = controller.subscribe()

    outcome = controller.fail(accepted.request_id, "device is busy")

    assert outcome == Failed("req-1", "device is busy")
    assert controller.state.microphone == Selection(desired="Yeti", effective=None)
    assert events(subscription) == [
        (
            "action.failed",
            {
                "request_id": "req-1",
                "action": "microphone.select",
                "actor": "local",
                "detail": "device is busy",
            },
        )
    ]


def test_a_newer_selection_supersedes_the_one_still_opening() -> None:
    """Arrowing through a device list produces one request per keystroke."""
    controller = wired()
    first = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    second = controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)

    assert controller.settle(first.request_id, "Yeti") == Superseded("req-1")
    assert controller.state.microphone == Selection(desired="Webcam", effective=None)
    assert controller.settle(second.request_id, "Webcam") == Applied("req-2", "Webcam")
    assert controller.state.microphone == Selection("Webcam", "Webcam")


def test_the_superseded_request_is_named_when_it_loses_its_slot() -> None:
    """The client waiting on req-1 has to learn that req-2 took it over."""
    controller = wired()
    controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    _, subscription = controller.subscribe()

    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)

    assert events(subscription) == [
        ("state.changed", {"microphone": Selection("Webcam", None)}),
        (
            "action.superseded",
            {"request_id": "req-1", "action": "microphone.select", "actor": "local"},
        ),
        (
            "action.accepted",
            {"request_id": "req-2", "action": "microphone.select", "actor": "local"},
        ),
    ]


def test_a_superseded_request_that_fails_also_changes_nothing() -> None:
    controller = wired()
    first = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)

    assert controller.fail(first.request_id, "device is busy") == Superseded("req-1")
    assert controller.state.microphone == Selection(desired="Webcam", effective=None)


def test_selections_of_different_things_do_not_supersede_each_other() -> None:
    controller = wired()
    mic = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    far = controller.dispatch("audio_stream.select", {"name": "Zoom"}, actor=OWNER)

    assert controller.settle(mic.request_id, "Yeti") == Applied("req-1", "Yeti")
    assert controller.settle(far.request_id, "Zoom") == Applied("req-2", "Zoom")
    assert controller.state.microphone == Selection("Yeti", "Yeti")
    assert controller.state.audio_stream == Selection("Zoom", "Zoom")


def test_claiming_a_request_commits_without_publishing_anything() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    _, subscription = controller.subscribe()

    outcome = controller.claim(accepted.request_id, "Yeti")

    assert outcome == Applied("req-1", "Yeti")
    assert controller.state.microphone == Selection("Yeti", "Yeti")
    assert events(subscription) == []


def test_announcing_a_claimed_request_publishes_state_then_applied() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.claim(accepted.request_id, "Yeti")
    _, subscription = controller.subscribe()

    assert controller.announce(accepted.request_id) == Applied("req-1", "Yeti")

    assert events(subscription) == [
        ("state.changed", {"microphone": Selection("Yeti", "Yeti")}),
        (
            "action.applied",
            {
                "request_id": "req-1",
                "action": "microphone.select",
                "actor": "local",
                "effective": "Yeti",
            },
        ),
    ]


def test_announcing_a_request_that_was_not_claimed_is_rejected() -> None:
    controller = wired()
    _, subscription = controller.subscribe()

    outcome = controller.announce("req-1")

    assert outcome == Rejected(
        "req-1", Rejection.INVALID, "no request in flight: req-1"
    )
    assert events(subscription) == []


def test_announcing_a_settled_request_returns_applied_without_republishing() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.settle(accepted.request_id, "Yeti")
    _, subscription = controller.subscribe()

    outcome = controller.announce(accepted.request_id)

    assert outcome == Applied("req-1", "Yeti")
    assert events(subscription) == []


def test_claiming_unchanged_state_publishes_no_state_event() -> None:
    def settle(state: AppState, _effective: object) -> AppState:
        return state

    def handle(_request, state):
        return Effect.pending(state, settle)

    controller = wired()
    controller.register("session.new", handle)
    accepted = controller.dispatch("session.new", actor=OWNER)
    controller.claim(accepted.request_id)
    _, subscription = controller.subscribe()

    assert controller.announce(accepted.request_id) == Applied("req-1")
    assert events(subscription) == [
        (
            "action.applied",
            {
                "request_id": "req-1",
                "action": "session.new",
                "actor": "local",
                "effective": None,
            },
        ),
    ]


def test_a_partially_superseded_fragment_keeps_fields_that_are_still_current() -> None:
    def settle(state: AppState, _effective: object) -> AppState:
        return replace(state, codex_model="opus", codex_reasoning="high")

    def handle(_request, state):
        return Effect.pending(state, settle)

    def set_reasoning(_request, state):
        return Effect.applied(replace(state, codex_reasoning="low"), "low")

    controller = wired()
    controller.register("session.new", handle)
    controller.register("turn_silence.set", set_reasoning)
    accepted = controller.dispatch("session.new", actor=OWNER)
    controller.claim(accepted.request_id)
    controller.dispatch("turn_silence.set", {"seconds": 1.0}, actor=OWNER)
    _, subscription = controller.subscribe()

    assert controller.announce(accepted.request_id) == Applied("req-1")
    assert controller.state.codex_model == "opus"
    assert controller.state.codex_reasoning == "low"
    assert events(subscription) == [
        ("state.changed", {"codex_model": "opus"}),
        (
            "action.applied",
            {
                "request_id": "req-1",
                "action": "session.new",
                "actor": "local",
                "effective": None,
            },
        ),
    ]


def test_announcing_twice_returns_the_same_applied_outcome() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.claim(accepted.request_id, "Yeti")
    first = controller.announce(accepted.request_id)
    _, subscription = controller.subscribe()

    second = controller.announce(accepted.request_id)

    assert first == second == Applied("req-1", "Yeti")
    assert events(subscription) == []


def test_a_claimed_request_cannot_be_settled_again() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.claim(accepted.request_id, "Yeti")

    outcome = controller.settle(accepted.request_id, "Webcam")

    assert outcome == Rejected(
        "req-1", Rejection.INVALID, "no request in flight: req-1"
    )
    assert controller.state.microphone == Selection("Yeti", "Yeti")


def test_a_request_that_already_settled_cannot_settle_twice() -> None:
    controller = wired()
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.settle(accepted.request_id, "Yeti")

    outcome = controller.settle(accepted.request_id, "Webcam")

    assert outcome == Rejected(
        "req-1", Rejection.INVALID, "no request in flight: req-1"
    )
    assert controller.state.microphone == Selection("Yeti", "Yeti")


def test_a_reconciler_reporting_a_request_nobody_made_is_told_so() -> None:
    outcome = wired().fail("req-99", "device is busy")

    assert outcome == Rejected(
        "req-99", Rejection.INVALID, "no request in flight: req-99"
    )


def test_a_superseded_request_is_still_recognised_when_it_reports_late() -> None:
    """A device that opens minutes later gets an answer, not a shrug."""
    controller = wired()
    first = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)
    controller.settle(first.request_id, "Yeti")

    assert controller.settle(first.request_id, "Yeti") == Superseded("req-1")


# -- idempotency --------------------------------------------------------


def test_a_retry_with_the_same_key_does_not_repeat_the_work() -> None:
    controller = Controller()
    calls: list[str] = []

    def counting(request, state):
        calls.append(request.id)
        return Effect.applied(replace(state, tts_enabled=True), True)

    controller.register("tts.set_enabled", counting)

    first = controller.dispatch(
        "tts.set_enabled", {"enabled": True}, actor=OWNER, idempotency_key="k1"
    )
    second = controller.dispatch(
        "tts.set_enabled", {"enabled": True}, actor=OWNER, idempotency_key="k1"
    )

    assert first == second == Applied("req-1", True)
    assert calls == ["req-1"]


def test_a_retried_selection_keeps_the_request_id_the_reconciler_owns() -> None:
    controller = wired()

    first = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    second = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert first == second == Accepted("req-1")
    assert controller.settle("req-1", "Yeti") == Applied("req-1", "Yeti")


def test_two_clients_that_both_count_from_one_do_not_collide() -> None:
    controller = wired()
    other = agent("notes-bot", set(Scope))

    mine = controller.dispatch(
        "tts.set_enabled", {"enabled": False}, actor=OWNER, idempotency_key="1"
    )
    theirs = controller.dispatch(
        "tts.set_enabled", {"enabled": True}, actor=other, idempotency_key="1"
    )

    assert mine == Applied("req-1", False)
    assert theirs == Applied("req-2", True)
    assert controller.state.tts_enabled is True


def test_without_a_key_a_repeated_request_runs_again() -> None:
    controller = wired()

    first = controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)
    second = controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert (first, second) == (Applied("req-1", False), Applied("req-2", False))


# -- one writer ---------------------------------------------------------


def test_concurrent_requests_each_see_what_the_last_one_left() -> None:
    """The read-modify-write below loses an update unless dispatch is serialized."""
    controller = Controller()

    def counting(_request, state):
        seen = state.turn_silence
        time.sleep(0.001)
        return Effect.applied(replace(state, turn_silence=seen + 1.0), seen + 1.0)

    controller.register("turn_silence.set", counting)
    callers = [
        threading.Thread(
            target=controller.dispatch,
            args=("turn_silence.set", {"seconds": 1.0}),
            kwargs={"actor": OWNER},
        )
        for _ in range(20)
    ]

    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=30)

    assert controller.state.turn_silence == AppState().turn_silence + 20.0


def test_every_request_is_given_its_own_identity() -> None:
    controller = wired()

    ids = {
        controller.dispatch(
            "tts.set_enabled", {"enabled": index % 2 == 0}, actor=OWNER
        ).request_id
        for index in range(5)
    }

    assert len(ids) == 5


# -- bounded memory -----------------------------------------------------


def test_the_oldest_idempotency_key_is_forgotten_first() -> None:
    """A long session must not accumulate every key a client ever sent."""
    controller = wired()
    for number in range(IDEMPOTENCY_MEMORY + 1):
        controller.dispatch(
            "tts.set_enabled",
            {"enabled": True},
            actor=OWNER,
            idempotency_key=f"k{number}",
        )

    retried = controller.dispatch(
        "tts.set_enabled", {"enabled": True}, actor=OWNER, idempotency_key="k0"
    )
    remembered = controller.dispatch(
        "tts.set_enabled",
        {"enabled": True},
        actor=OWNER,
        idempotency_key=f"k{IDEMPOTENCY_MEMORY}",
    )

    assert retried.request_id != "req-1"
    assert remembered.request_id == f"req-{IDEMPOTENCY_MEMORY + 1}"


def test_a_key_is_remembered_until_the_memory_is_really_full() -> None:
    controller = wired()
    for number in range(IDEMPOTENCY_MEMORY):
        controller.dispatch(
            "tts.set_enabled",
            {"enabled": True},
            actor=OWNER,
            idempotency_key=f"k{number}",
        )

    retried = controller.dispatch(
        "tts.set_enabled", {"enabled": True}, actor=OWNER, idempotency_key="k0"
    )

    assert retried.request_id == "req-1"


def test_only_recent_superseded_requests_are_remembered() -> None:
    controller = wired()
    lost = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=OWNER)
    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)
    controller.settle(lost.request_id, "Yeti")

    for _ in range(SUPERSEDED_MEMORY):
        overtaken = controller.dispatch(
            "microphone.select", {"name": "Yeti"}, actor=OWNER
        )
        controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)
        controller.settle(overtaken.request_id, "Yeti")

    outcome = controller.settle(lost.request_id, "Yeti")

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.INVALID


# -- provenance ---------------------------------------------------------


def test_every_lifecycle_event_names_the_actor_that_asked() -> None:
    """Who attempted what, and when, is the record an operator reads."""
    controller = wired()
    bot = agent("notes-bot", set(Scope))
    _, subscription = controller.subscribe()

    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=bot)
    controller.settle(accepted.request_id, "Yeti")

    assert [(name, payload.get("actor")) for name, payload in events(subscription)] == [
        ("state.changed", None),
        ("action.accepted", "notes-bot"),
        ("state.changed", None),
        ("action.applied", "notes-bot"),
    ]


def test_a_result_reported_much_later_is_still_attributed_to_who_asked() -> None:
    controller = wired()
    bot = agent("notes-bot", set(Scope))
    accepted = controller.dispatch("microphone.select", {"name": "Yeti"}, actor=bot)
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)
    _, subscription = controller.subscribe()

    controller.fail(accepted.request_id, "device is busy")

    assert events(subscription) == [
        (
            "action.failed",
            {
                "request_id": "req-1",
                "action": "microphone.select",
                "actor": "notes-bot",
                "detail": "device is busy",
            },
        )
    ]


def test_a_refusal_records_the_actor_that_was_turned_away() -> None:
    controller = wired()
    reader = agent("notes-bot", {Scope.TRANSCRIPT})
    _, subscription = controller.subscribe()

    controller.dispatch("microphone.select", {"name": "Yeti"}, actor=reader)

    assert events(subscription) == [
        (
            "action.rejected",
            {
                "request_id": "req-1",
                "action": "microphone.select",
                "actor": "notes-bot",
                "reason": Rejection.FORBIDDEN,
                "detail": "notes-bot does not hold the audio scope",
            },
        )
    ]


# -- retries after a disconnect -----------------------------------------


def test_a_key_reused_for_a_different_request_is_refused() -> None:
    """Answering it would drop the second action and report the first result."""
    controller = wired()
    controller.dispatch(
        "tts.set_enabled", {"enabled": False}, actor=OWNER, idempotency_key="k1"
    )

    outcome = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert isinstance(outcome, Rejected)
    assert outcome.reason is Rejection.INVALID
    assert "already used for tts.set_enabled" in outcome.detail
    assert controller.state.microphone == Selection()


def test_a_key_reused_for_the_same_request_with_a_new_value_is_refused() -> None:
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    outcome = controller.dispatch(
        "microphone.select", {"name": "Webcam"}, actor=OWNER, idempotency_key="k1"
    )

    assert isinstance(outcome, Rejected)
    assert controller.state.microphone == Selection(desired="Yeti")


def test_a_retry_learns_the_outcome_the_request_has_since_reached() -> None:
    """The event announcing it may have been published before the reconnect."""
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    controller.settle("req-1", "Yeti")

    retried = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert retried == Applied("req-1", "Yeti")


def test_a_keyed_retry_stays_accepted_until_the_claim_is_announced() -> None:
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    controller.claim("req-1", "Yeti")

    between = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    controller.announce("req-1")
    after = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert between == Accepted("req-1")
    assert after == Applied("req-1", "Yeti")


def test_the_oldest_unannounced_claim_is_released_first() -> None:
    controller = wired()
    first = controller.dispatch("microphone.select", {"name": "mic-0"}, actor=OWNER)
    controller.claim(first.request_id, "mic-0")
    _, subscription = controller.subscribe()
    for number in range(1, UNPUBLISHED_MEMORY):
        accepted = controller.dispatch(
            "microphone.select", {"name": f"mic-{number}"}, actor=OWNER
        )
        controller.claim(accepted.request_id, f"mic-{number}")

    assert not any(name == "action.applied" for name, _ in events(subscription))

    overflow = controller.dispatch(
        "microphone.select", {"name": "mic-overflow"}, actor=OWNER
    )
    controller.claim(overflow.request_id, "mic-overflow")

    assert any(
        name == "action.applied" and payload["request_id"] == first.request_id
        for name, payload in events(subscription)
    )
    assert controller.announce(first.request_id) == Applied(first.request_id, "mic-0")


def test_remembered_announcements_are_bounded() -> None:
    controller = wired()
    first = controller.dispatch("microphone.select", {"name": "mic-0"}, actor=OWNER)
    controller.settle(first.request_id, "mic-0")
    for number in range(1, UNPUBLISHED_MEMORY):
        accepted = controller.dispatch(
            "microphone.select", {"name": f"mic-{number}"}, actor=OWNER
        )
        controller.settle(accepted.request_id, f"mic-{number}")

    assert controller.announce(first.request_id) == Applied(first.request_id, "mic-0")

    overflow = controller.dispatch(
        "microphone.select", {"name": "mic-overflow"}, actor=OWNER
    )
    controller.settle(overflow.request_id, "mic-overflow")

    assert controller.announce(first.request_id) == Rejected(
        first.request_id,
        Rejection.INVALID,
        f"no request in flight: {first.request_id}",
    )


def test_a_retry_of_a_request_that_was_overtaken_says_so() -> None:
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    controller.dispatch("microphone.select", {"name": "Webcam"}, actor=OWNER)

    retried = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert retried == Superseded("req-1")


def test_a_retry_of_a_request_that_failed_says_so() -> None:
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    controller.fail("req-1", "device is busy")

    retried = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert retried == Failed("req-1", "device is busy")


# -- bounded supersession -----------------------------------------------


def test_an_overtaken_request_is_retired_the_moment_it_loses_its_slot() -> None:
    """Selection reconcilers coalesce, so the overtaken one may never report."""
    controller = wired()

    for number in range(1000):
        controller.dispatch(
            "microphone.select", {"name": f"device-{number}"}, actor=OWNER
        )

    assert len(controller._requests.pending) == 1
    assert len(controller._requests.superseded) == SUPERSEDED_MEMORY
    assert controller.settle("req-999", "device-998") == Superseded("req-999")


@pytest.mark.parametrize(
    ("action_id", "payload", "reason", "detail"),
    [
        (
            "microphone.explode",
            {},
            Rejection.UNKNOWN_ACTION,
            "no such action: microphone.explode",
        ),
        (
            "turn_silence.set",
            {"seconds": "soon"},
            Rejection.INVALID,
            "turn_silence.set: seconds: expected a number",
        ),
        (
            "session.new",
            {},
            Rejection.INAPPLICABLE,
            "session.new is not available in this session",
        ),
        (
            "audio_stream.select",
            {"name": "Zoom"},
            Rejection.INAPPLICABLE,
            "no such application: Zoom",
        ),
    ],
)
def test_a_rejection_records_the_action_it_was_sent_to(
    action_id, payload, reason, detail
) -> None:
    """The audit record has to say what was attempted, not only that it failed."""
    controller = wired()
    controller.register("audio_stream.select", refusing("no such application: Zoom"))
    _, subscription = controller.subscribe()

    outcome = controller.dispatch(action_id, payload, actor=OWNER)

    assert outcome == Rejected("req-1", reason, detail)
    assert events(subscription) == [
        (
            "action.rejected",
            {
                "request_id": "req-1",
                "action": action_id,
                "actor": "local",
                "reason": reason,
                "detail": detail,
            },
        )
    ]


def test_a_conflicting_key_is_recorded_as_its_own_attempt() -> None:
    controller = wired()
    controller.dispatch(
        "tts.set_enabled", {"enabled": False}, actor=OWNER, idempotency_key="k1"
    )
    _, subscription = controller.subscribe()

    outcome = controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )

    assert outcome.request_id == "req-2"
    assert events(subscription) == [
        (
            "action.rejected",
            {
                "request_id": "req-2",
                "action": "microphone.select",
                "actor": "local",
                "reason": Rejection.INVALID,
                "detail": (
                    "idempotency key 'k1' was already used for tts.set_enabled "
                    "with a different payload"
                ),
            },
        )
    ]


def test_a_key_evicted_while_its_request_was_in_flight_still_settles() -> None:
    """The request outlives its key; settling it must not go looking for one."""
    controller = wired()
    controller.dispatch(
        "microphone.select", {"name": "Yeti"}, actor=OWNER, idempotency_key="k1"
    )
    for number in range(IDEMPOTENCY_MEMORY):
        controller.dispatch(
            "tts.set_enabled",
            {"enabled": True},
            actor=OWNER,
            idempotency_key=f"filler-{number}",
        )

    assert controller.settle("req-1", "Yeti") == Applied("req-1", "Yeti")
    assert controller.state.microphone == Selection("Yeti", "Yeti")


def test_only_a_request_still_in_flight_is_tracked_by_its_key() -> None:
    """A terminal outcome has nothing left to update, so it is not indexed."""
    controller = wired()

    for number in range(20):
        controller.dispatch(
            "tts.set_enabled",
            {"enabled": True},
            actor=OWNER,
            idempotency_key=f"k{number}",
        )

    assert controller._keyed == {}


def test_the_controller_stamps_its_events_from_the_clock_it_was_given() -> None:
    controller = Controller(clock=lambda: STAMP)
    controller.register("tts.set_enabled", setting_tts)
    _, subscription = controller.subscribe()

    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    assert [event.at for event in subscription.drain()] == [STAMP, STAMP]


def test_a_key_is_bound_to_the_request_the_session_understood() -> None:
    """A caller that keeps editing its own list must not rewrite what it asked."""
    controller = wired()
    images = ["img-a"]

    first = controller.dispatch(
        "message.send",
        {"text": "hi", "images": images},
        actor=OWNER,
        idempotency_key="k1",
    )
    images[0] = "img-b"
    retried = controller.dispatch(
        "message.send",
        {"text": "hi", "images": ["img-a"]},
        actor=OWNER,
        idempotency_key="k1",
    )

    assert first == retried == Applied("req-1", ("img-a",))


def test_two_spellings_of_one_request_are_one_request() -> None:
    """The key matches the validated payload, not the container it arrived in."""
    controller = wired()
    controller.dispatch(
        "message.send",
        {"text": "hi", "images": ["img-a"]},
        actor=OWNER,
        idempotency_key="k1",
    )

    retried = controller.dispatch(
        "message.send",
        {"text": "hi", "images": ("img-a",), "respond": True},
        actor=OWNER,
        idempotency_key="k1",
    )

    assert retried == Applied("req-1", ("img-a",))


def test_a_key_used_on_a_refused_payload_is_free_for_the_corrected_retry() -> None:
    """Nothing ran, so nothing is bound; the caller fixes it and sends again."""
    controller = wired()
    refused = controller.dispatch(
        "message.send", {"text": 3}, actor=OWNER, idempotency_key="k1"
    )

    corrected = controller.dispatch(
        "message.send", {"text": "hi"}, actor=OWNER, idempotency_key="k1"
    )

    assert isinstance(refused, Rejected)
    assert corrected == Applied("req-2", ())


def test_a_handler_cannot_edit_the_payload_it_was_given() -> None:
    controller = Controller()
    payloads: list[Mapping[str, object]] = []

    def recording(request, state):
        payloads.append(request.payload)
        return Effect.applied(state, None)

    controller.register("tts.set_enabled", recording)
    controller.dispatch("tts.set_enabled", {"enabled": False}, actor=OWNER)

    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(dict, payloads[0])["enabled"] = True


def test_set_partial_drops_superseded_stamps() -> None:
    """Older partial publishes must not overwrite a newer AppState line (#102)."""
    controller = Controller()
    controller.set_partial("Voice", "newest", seq=2)
    controller.set_partial("Voice", "stale", seq=1)
    assert controller.state.partial_source == "Voice"
    assert controller.state.partial_text == "newest"


def test_set_session_state_publishes_known_fields_and_ignores_other_fields() -> None:
    controller = Controller()
    _snapshot, subscription = controller.subscribe()

    controller.set_session_state(
        {
            "tokens": 42,
            "echoes_cut": 3,
            "codex_thread": "thread-9",
            "not_a_session_field": "ignored",
        }
    )

    assert controller.state.tokens == 42
    assert controller.state.echoes_cut == 3
    assert controller.state.codex_thread == "thread-9"
    assert not hasattr(controller.state, "not_a_session_field")
    changed = [event for event in subscription.drain() if event.name == "state.changed"]
    assert [dict(event.payload) for event in changed] == [
        {"tokens": 42, "echoes_cut": 3, "codex_thread": "thread-9"}
    ]

    controller.set_session_state({"not_a_session_field": "still ignored"})
    assert subscription.drain() == ()
