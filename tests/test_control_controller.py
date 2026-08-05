from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from tagalong.control.actions import CATALOG, PROTOCOL_VERSION, Scope
from tagalong.control.actors import agent, local_user
from tagalong.control.controller import (
    IDEMPOTENCY_MEMORY,
    SUPERSEDED_MEMORY,
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


def wired() -> Controller:
    """A controller with the handlers these tests dispatch against."""
    controller = Controller()
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


def test_a_refusal_changes_nothing_and_publishes_nothing() -> None:
    """Nothing happened, so there is nothing for another client to render."""
    controller = wired()
    _, subscription = controller.subscribe()

    controller.dispatch("turn_silence.set", {"seconds": "soon"}, actor=OWNER)

    assert events(subscription) == []


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
            {"request_id": "req-1", "action": "turn_silence.set", "effective": 30.0},
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
        ("action.accepted", {"request_id": "req-1", "action": "microphone.select"}),
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
        ("action.superseded", {"request_id": "req-1", "action": "microphone.select"}),
        ("action.accepted", {"request_id": "req-2", "action": "microphone.select"}),
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
