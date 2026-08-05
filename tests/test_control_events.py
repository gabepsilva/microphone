from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import cast

import pytest

from tagalong.control.events import (
    Event,
    EventLog,
    Subscription,
    frozen,
    utc_now,
)

STAMP = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def names(events: tuple[Event, ...]) -> list[str]:
    return [event.name for event in events]


def test_events_are_numbered_in_the_order_they_happened() -> None:
    log = EventLog()
    subscription = log.subscribe()

    log.publish("state.changed", {"tts_enabled": False})
    log.publish("action.applied", {"request_id": "req-1"})

    events = subscription.drain()
    assert [event.sequence for event in events] == [1, 2]
    assert names(events) == ["state.changed", "action.applied"]
    assert events[0].payload == {"tts_enabled": False}


def test_the_sequence_says_how_much_a_client_has_missed() -> None:
    log = EventLog()

    assert log.sequence == 0
    log.publish("state.changed")
    assert log.sequence == 1


def test_a_restarted_runtime_is_not_a_gap_in_the_old_one() -> None:
    """Two logs both count from one, so identity is what tells them apart."""
    assert EventLog().instance != EventLog().instance


def test_a_subscriber_hears_only_what_happened_after_it_subscribed() -> None:
    log = EventLog()
    log.publish("state.changed", {"before": True})

    subscription = log.subscribe()
    log.publish("state.changed", {"after": True})

    assert [event.payload for event in subscription.drain()] == [{"after": True}]


def test_every_open_subscriber_hears_the_same_event() -> None:
    log = EventLog()
    first, second = log.subscribe(), log.subscribe()

    log.publish("action.applied")

    assert names(first.drain()) == names(second.drain()) == ["action.applied"]


def test_draining_takes_each_event_once() -> None:
    log = EventLog()
    subscription = log.subscribe()
    log.publish("action.applied")

    assert len(subscription.drain()) == 1
    assert subscription.drain() == ()


def test_a_client_that_falls_too_far_behind_is_told_it_lost_events() -> None:
    """Rendering from half the updates is worse than resynchronizing."""
    log = EventLog(capacity=2)
    subscription = log.subscribe()

    for index in range(3):
        log.publish("state.changed", {"index": index})

    assert subscription.lost is True
    assert [event.payload["index"] for event in subscription.drain()] == [1, 2]


def test_a_client_that_keeps_up_is_never_told_it_lost_anything() -> None:
    log = EventLog(capacity=2)
    subscription = log.subscribe()

    for index in range(4):
        log.publish("state.changed", {"index": index})
        subscription.drain()

    assert subscription.lost is False


def test_a_waiting_client_is_woken_by_the_next_event() -> None:
    log = EventLog()
    subscription = log.subscribe()
    woken = threading.Event()

    def listen() -> None:
        if subscription.wait(timeout=5):
            woken.set()

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    log.publish("action.applied")
    listener.join(timeout=5)

    assert woken.is_set()


def test_waiting_for_nothing_ends_rather_than_blocking() -> None:
    assert Subscription().wait(timeout=0.01) is False


def test_a_closed_subscription_stops_receiving_and_stops_waiting() -> None:
    """A client thread parked during shutdown leaves instead of sitting it out."""
    log = EventLog()
    subscription = log.subscribe()
    subscription.close()

    log.publish("action.applied")

    assert subscription.wait(timeout=0.01) is True
    assert subscription.drain() == ()
    assert subscription.open is False


def test_a_closed_subscriber_stops_costing_the_publisher() -> None:
    log = EventLog()
    gone = log.subscribe()
    staying = log.subscribe()
    gone.close()

    log.publish("action.applied")
    log.publish("action.applied")

    assert len(staying.drain()) == 2
    assert log.subscribers == 1


def test_a_container_inside_a_payload_is_frozen_too() -> None:
    """A read-only mapping still hands out the list inside it."""
    log = EventLog()
    mine, theirs = log.subscribe(), log.subscribe()

    log.publish("example", {"items": ["kept"], "nested": {"inner": ["kept"]}})

    (event,) = mine.drain()
    assert event.payload["items"] == ("kept",)
    with pytest.raises(AttributeError, match="append"):
        cast(list, event.payload["items"]).append("corrupted")
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(dict, event.payload["nested"])["inner"] = "corrupted"

    (theirs_event,) = theirs.drain()
    assert theirs_event.payload["nested"] == {"inner": ("kept",)}


def test_freezing_leaves_the_values_a_payload_is_made_of_alone() -> None:
    """Strings, numbers, and frozen dataclasses are values; only holders change."""
    assert frozen("Yeti") == "Yeti"
    assert frozen(3.0) == 3.0
    assert frozen(None) is None
    assert frozen({"a", "b"}) == frozenset({"a", "b"})
    assert frozen(("kept",)) == ("kept",)


def test_publishing_reports_the_event_it_numbered() -> None:
    log = EventLog(clock=lambda: STAMP)

    event = log.publish("state.changed", {"tts_enabled": True})

    assert event == Event(1, "state.changed", {"tts_enabled": True}, STAMP)


def test_an_event_says_when_it_happened_as_well_as_in_what_order() -> None:
    """A sequence number says how much was missed and nothing about when."""
    log = EventLog(clock=lambda: STAMP)
    subscription = log.subscribe()

    log.publish("action.applied")

    (event,) = subscription.drain()
    assert event.at == STAMP
    assert event.at.tzinfo is UTC


def test_the_clock_is_read_at_the_moment_of_publication() -> None:
    ticks = iter([STAMP, STAMP.replace(second=30)])
    log = EventLog(clock=lambda: next(ticks))
    subscription = log.subscribe()

    log.publish("state.changed")
    log.publish("action.applied")

    assert [event.at.second for event in subscription.drain()] == [0, 30]


def test_one_client_cannot_rewrite_what_another_has_queued() -> None:
    """One event object reaches every subscriber, so its payload is read-only."""
    log = EventLog()
    mine, theirs = log.subscribe(), log.subscribe()
    log.publish("state.changed", {"tts_enabled": True})

    # Cast because that is the mistake being guarded against: a client that
    # believes it holds a plain dict and writes to it.
    (event,) = mine.drain()
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(dict, event.payload)["tts_enabled"] = "corrupted"

    assert theirs.drain()[0].payload == {"tts_enabled": True}


def test_the_default_clock_is_an_aware_utc_reading() -> None:
    """A naive stamp read back later cannot be placed against anything else."""
    assert utc_now().tzinfo is UTC


def test_an_event_payload_is_a_copy_of_what_was_published() -> None:
    """A caller reusing its payload dict must not rewrite delivered history."""
    log = EventLog()
    payload = {"tts_enabled": True}

    event = log.publish("state.changed", payload)
    payload["tts_enabled"] = False

    assert event.payload == {"tts_enabled": True}


def test_an_event_racing_a_close_is_dropped_rather_than_queued() -> None:
    """The publisher sweeps closed subscribers between events, not during one."""
    subscription = Subscription()
    subscription.close()

    subscription.deliver(Event(1, "action.applied"))

    assert subscription.drain() == ()
