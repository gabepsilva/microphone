"""UI-neutral application core: typed actions, one ordered writer, and events.

This package is what a client talks to. It owns the canonical state, decides
what an actor is allowed to do, applies changes in one definite order, and
publishes what happened. It knows nothing about Textual, sockets, or agents —
each of those is an adapter that translates its own input into the same typed
actions and renders the same events.

Nothing in here opens a device, calls a model, or writes a file. Those live
behind reconcilers that answer :meth:`~.controller.Controller.settle` or
:meth:`~.controller.Controller.fail` when the slow work is over, which is what
keeps the ordered core responsive while a capture device is loading.
"""

from .actions import (
    CATALOG,
    PROTOCOL_VERSION,
    ActionSpec,
    Capability,
    InvalidPayload,
    Kind,
    Parameter,
    Scope,
)
from .actors import Actor, ActorKind, agent, local_user
from .controller import Controller, Handler, Request, Snapshot
from .events import Event, EventLog, Subscription
from .outcomes import (
    Accepted,
    Applied,
    EffectFailed,
    Failed,
    Inapplicable,
    Outcome,
    Rejected,
    Rejection,
    Superseded,
)
from .state import AppState, Effect, Selection, with_desired, with_effective

__all__ = [
    "CATALOG",
    "PROTOCOL_VERSION",
    "Accepted",
    "ActionSpec",
    "Actor",
    "ActorKind",
    "AppState",
    "Applied",
    "Capability",
    "Controller",
    "Effect",
    "EffectFailed",
    "Event",
    "EventLog",
    "Failed",
    "Handler",
    "Inapplicable",
    "InvalidPayload",
    "Kind",
    "Outcome",
    "Parameter",
    "Rejected",
    "Rejection",
    "Request",
    "Scope",
    "Selection",
    "Snapshot",
    "Subscription",
    "Superseded",
    "agent",
    "local_user",
    "with_desired",
    "with_effective",
]
