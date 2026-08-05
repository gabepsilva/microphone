from __future__ import annotations

import threading

from tagalong.control.events import Event, EventLog, Subscription


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


def test_publishing_reports_the_event_it_numbered() -> None:
    log = EventLog()

    event = log.publish("state.changed", {"tts_enabled": True})

    assert event == Event(1, "state.changed", {"tts_enabled": True})


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
