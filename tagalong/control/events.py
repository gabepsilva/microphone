"""Ordered notifications, and how a client catches up after missing some.

Every event carries a sequence number assigned in the order the state changed,
so a client can tell "nothing happened" from "I did not hear about it". The
numbers belong to a runtime instance: a restarted session starts counting from
one again, and a client comparing only sequence numbers would take the second
event 7 for the first. Comparing ``instance`` first is what makes a restart
distinguishable from a gap.

Delivery is a bounded queue per subscriber and nothing else. The publisher
appends and returns; it never calls into client code, so a slow or stopped
client cannot hold up the state transition that produced the event — the
property this package exists to protect. When a client falls far enough behind
to overflow its queue, the oldest events are dropped and the subscription says
so. Dropping is the only honest option left at that point, and reporting it
lets the client take a fresh snapshot rather than render a state built from
updates it half received.

The log itself does no locking. Its owner serializes ``publish`` and
``subscribe`` under the same lock that guards the state, because an event
delivered in a different order from the state change it describes is worse
than no event at all. Draining is safe from any thread.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

# Room for a client that stops reading for a moment without losing anything,
# and small enough that a client that stopped for good is not a memory leak.
DEFAULT_CAPACITY = 512


@dataclass(frozen=True)
class Event:
    """One thing that happened, numbered in the order it happened."""

    sequence: int
    name: str
    payload: Mapping[str, object] = field(default_factory=dict)


class Subscription:
    """A client's ordered view of what happened after it took its snapshot."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._events: deque[Event] = deque()
        self._capacity = capacity
        self._lost = False
        self._arrived = threading.Event()
        self._open = True

    def deliver(self, event: Event) -> None:
        """Queue *event*, dropping the oldest when this client is too far behind."""
        with self._lock:
            if not self._open:
                return
            self._events.append(event)
            while len(self._events) > self._capacity:
                self._events.popleft()
                self._lost = True
            self._arrived.set()

    def drain(self) -> tuple[Event, ...]:
        """Take everything queued so far, in order."""
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            self._arrived.clear()
            return events

    @property
    def lost(self) -> bool:
        """True once events were dropped; the client must take a fresh snapshot."""
        with self._lock:
            return self._lost

    def wait(self, timeout: float | None = None) -> bool:
        """Block until something is queued. Returns False when the wait timed out."""
        return self._arrived.wait(timeout)

    def close(self) -> None:
        """Stop receiving. Whatever was queued is dropped with the subscription.

        Closing wakes anyone waiting, so a client thread parked on ``wait``
        during shutdown leaves rather than sitting out its timeout.
        """
        with self._lock:
            self._open = False
            self._events.clear()
            self._arrived.set()

    @property
    def open(self) -> bool:
        with self._lock:
            return self._open


class EventLog:
    """Numbers events and fans them out to the subscribers that are still open."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self.instance = uuid.uuid4().hex
        self.capacity = capacity
        self._sequence = 0
        self._subscribers: list[Subscription] = []

    @property
    def sequence(self) -> int:
        """The number of the most recent event; 0 before anything happened."""
        return self._sequence

    @property
    def subscribers(self) -> int:
        """How many subscriptions still receive; closed ones are swept on publish."""
        return len(self._subscribers)

    def publish(self, name: str, payload: Mapping[str, object] | None = None) -> Event:
        """Number an event and hand it to every open subscriber."""
        self._sequence += 1
        event = Event(self._sequence, name, dict(payload or {}))
        # Closed subscriptions are swept here rather than on close, so a client
        # that disconnects without closing costs one pass and not a leak.
        self._subscribers = [
            subscriber for subscriber in self._subscribers if subscriber.open
        ]
        for subscriber in self._subscribers:
            subscriber.deliver(event)
        return event

    def subscribe(self) -> Subscription:
        """Start receiving events published from now on."""
        subscription = Subscription(self.capacity)
        self._subscribers.append(subscription)
        return subscription
