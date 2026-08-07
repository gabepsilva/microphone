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
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

# Room for a client that stops reading for a moment without losing anything,
# and small enough that a client that stopped for good is not a memory leak.
DEFAULT_CAPACITY = 512

# Issue #102 D4: stay on one EventLog while steady-state publish stays at or
# below this rate during streaming Codex + both capture channels. Above it,
# split transcript to its own log. Measured via ``publishes_last_second``.
PUBLISH_RATE_TRIPWIRE_PER_SECOND = 50

_NOTHING: Mapping[str, object] = MappingProxyType({})


def frozen(value: object) -> object:
    """Return *value* with every container in it replaced by a read-only one.

    Deep rather than shallow because the demonstrated corruption is one level
    down: a read-only mapping still hands out the list inside it, and a client
    that appends to that list has edited what every other subscriber is about
    to read. Values that are not containers are returned as they are — see
    :class:`Event` for why that is the boundary.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(frozen(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(frozen(item) for item in value)
    return value


def frozen_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Freeze a whole payload, so publication owns a copy nobody else can edit."""
    return MappingProxyType({key: frozen(value) for key, value in payload.items()})


def utc_now() -> datetime:
    """The wall clock the log stamps with, aware so a transcript can be read."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class Event:
    """One thing that happened, numbered and stamped in the order it happened.

    The sequence number is what a client resumes from; the timestamp is what an
    operator reads. Both are needed — a number says how much was missed and
    nothing about when, and a clock can repeat or run backwards.

    One event object is delivered to every subscriber, so its payload is
    frozen on publication, all the way down: a client that drained it cannot
    rewrite history queued for the others. Lists become tuples, sets become
    frozensets, and nested mappings become read-only, so the containers a
    payload is built from cannot be edited in place by whoever drained it
    first.

    What that cannot reach is a mutable object of some other type. Publishing
    one is a mistake this module can neither detect nor undo, so payload values
    are the values the session already deals in — strings, numbers, byte
    strings, and frozen dataclasses like the state fragments.
    """

    sequence: int
    name: str
    payload: Mapping[str, object] = _NOTHING
    at: datetime = field(default_factory=utc_now)


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

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        clock: Callable[[], datetime] = utc_now,
        *,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.instance = uuid.uuid4().hex
        self.capacity = capacity
        self._clock = clock
        self._mono = mono
        self._sequence = 0
        self._subscribers: list[Subscription] = []
        self._publish_times: deque[float] = deque()

    @property
    def sequence(self) -> int:
        """The number of the most recent event; 0 before anything happened."""
        return self._sequence

    @property
    def subscribers(self) -> int:
        """How many subscriptions still receive; closed ones are swept on publish."""
        return len(self._subscribers)

    @property
    def publishes_last_second(self) -> int:
        """How many ``publish`` calls landed in the trailing one-second window.

        Read-only: does not prune. Snapshot the deque before iterating —
        ``publish`` may append/popleft on another thread under the controller
        lock, and iterating a live deque raises ``RuntimeError``.
        """
        now = self._mono()
        stamps = tuple(self._publish_times)
        return sum(1 for stamp in stamps if now - stamp < 1.0)

    def publish(self, name: str, payload: Mapping[str, object] | None = None) -> Event:
        """Number an event and hand it to every open subscriber."""
        now = self._mono()
        self._publish_times.append(now)
        while self._publish_times and now - self._publish_times[0] >= 1.0:
            self._publish_times.popleft()
        self._sequence += 1
        # Copied, then frozen: the caller keeps its own dict, and no consumer
        # can reach into what every other consumer is holding.
        event = Event(
            self._sequence,
            name,
            frozen_payload(payload or {}),
            self._clock(),
        )
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
