"""What a caller learns about one request, from refusal to effect.

A request has exactly one terminal outcome, and the five here are the whole
vocabulary. The split that matters is between refusing and trying: a
:class:`Rejected` request never touched the session, so a caller can correct
it and send it again, while a :class:`Failed` one ran and broke, and repeating
it unchanged will most likely break the same way.

:class:`Accepted` is the only non-terminal outcome. It means the desired state
was recorded and a reconciler owns the rest, so the eventual answer arrives
later as :class:`Applied`, :class:`Failed`, or :class:`Superseded` carrying
the same ``request_id``. Nothing else in this package waits for slow work, and
this outcome is why it does not have to.

:class:`Applied` carries the effective value rather than the requested one.
The two differ whenever a setting coerces — ``TurnSilence.set`` bounds the
window it is handed and returns what it adopted — and a client that renders
its own request shows a number the session is not running. That holds on the
immediate path too: coercion is not something only reconcilers produce.

Every outcome carries a ``request_id``, including the refusals. An audit trail
that omits what was turned away is the half of the record an operator most
often needs, and a client that correlates by id should not need a second rule
for the case where the id is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rejection(StrEnum):
    """Why a request was refused before anything was attempted."""

    UNKNOWN_ACTION = "unknown_action"
    FORBIDDEN = "forbidden"
    INVALID = "invalid"
    INAPPLICABLE = "inapplicable"


@dataclass(frozen=True)
class Rejected:
    """Refused before any effect ran: nothing in the session changed."""

    request_id: str
    reason: Rejection
    detail: str


@dataclass(frozen=True)
class Accepted:
    """Desired state recorded; a reconciler will report the effect later."""

    request_id: str


@dataclass(frozen=True)
class Applied:
    """The effect succeeded. ``effective`` is what the session actually holds."""

    request_id: str
    effective: object = None


@dataclass(frozen=True)
class Failed:
    """The effect was attempted and did not succeed."""

    request_id: str
    detail: str


@dataclass(frozen=True)
class Superseded:
    """A newer request for the same setting won before this one landed."""

    request_id: str


Outcome = Rejected | Accepted | Applied | Failed | Superseded


class Inapplicable(Exception):
    """Raised by a handler when the session cannot honour a valid request now."""


class EffectFailed(Exception):
    """Raised by a handler whose effect was attempted and broke."""
