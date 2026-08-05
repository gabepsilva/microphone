"""The one place session state changes, whoever asked for it.

Every client — the Textual interface, a future window, an agent — sends the
same typed actions here and renders what comes back. That is what keeps three
frontends from growing three slightly different ideas of what "mute" means,
and it is why the interface is expected to dispatch and re-render rather than
change state and tell the runtime afterwards.

**One writer.** A lock serializes dispatch, so requests arriving from an audio
callback, a keystroke, and an agent apply in some definite order and each one
sees the state the last one left. The lock is the single writer; a queue and a
worker thread would give the same ordering and cost every caller an extra hop
to learn what happened.

**The writer never waits.** Handlers run holding that lock, so a handler that
blocks blocks the session — including the mute the user pressed to stop the
noise. Slow work therefore does not belong in a handler: a handler records the
desired state, answers :class:`~.outcomes.Accepted`, and hands the work to a
reconciler that reports back through :meth:`Controller.settle` or
:meth:`Controller.fail`. Publishing an event is only an append to each
subscriber's queue, so notification cannot block the writer either.

**A newer request wins.** Two selections in flight for the same setting is the
normal case — someone arrowing through a device list produces one per
keystroke — so a pending request for an action is superseded by the next, and
a superseded request never applies its result. This is what stops a slow
device that finally opened from overwriting the choice made after it, and what
keeps a failed or overtaken request from being persisted as though it ran.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .actions import (
    CATALOG,
    PROTOCOL_VERSION,
    ActionSpec,
    Capability,
    InvalidPayload,
)
from .actors import Actor
from .events import DEFAULT_CAPACITY, EventLog, Subscription
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
from .state import AppState, Effect

# Enough for a client to retry the handful of requests a disconnect can leave
# unanswered, and bounded so a long session cannot accumulate keys forever.
IDEMPOTENCY_MEMORY = 256
# Superseded requests are remembered only so a reconciler reporting one gets a
# straight answer instead of "unknown". A few are always in flight; thousands
# never are.
SUPERSEDED_MEMORY = 64


@dataclass(frozen=True)
class Request:
    """One validated call: who made it, what it asks for, and its identity."""

    id: str
    action: ActionSpec
    payload: Mapping[str, object]
    actor: Actor


Handler = Callable[[Request, AppState], Effect]


@dataclass(frozen=True)
class Snapshot:
    """Complete state plus the cursor a client resumes its event stream from."""

    instance: str
    sequence: int
    state: AppState
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class _Pending:
    """A request a reconciler still owes an answer for."""

    action_id: str
    settle: Callable[[AppState, object], AppState]
    superseded: bool = False


@dataclass
class _Requests:
    """In-flight requests, one slot per action, plus what recently lost its slot."""

    pending: dict[str, _Pending] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)
    superseded: OrderedDict[str, None] = field(default_factory=OrderedDict)

    def start(self, request_id: str, pending: _Pending) -> str | None:
        """Take the slot for this action, returning whoever held it before."""
        displaced = self.slots.get(pending.action_id)
        if displaced is not None:
            self.pending[displaced] = _Pending(
                pending.action_id, self.pending[displaced].settle, superseded=True
            )
        self.slots[pending.action_id] = request_id
        self.pending[request_id] = pending
        return displaced

    def finish(self, request_id: str) -> _Pending | None:
        """Retire a request, freeing its slot when it still held one."""
        pending = self.pending.pop(request_id, None)
        if pending is None:
            return None
        if self.slots.get(pending.action_id) == request_id:
            del self.slots[pending.action_id]
        if pending.superseded:
            self.superseded[request_id] = None
            while len(self.superseded) > SUPERSEDED_MEMORY:
                self.superseded.popitem(last=False)
        return pending


class Controller:
    """The ordered core: validate, authorize, change state, publish."""

    def __init__(
        self,
        state: AppState | None = None,
        *,
        catalog: tuple[ActionSpec, ...] = CATALOG,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        self._lock = threading.Lock()
        self._state = AppState() if state is None else state
        self._catalog = catalog
        self._by_id = {action.id: action for action in catalog}
        self._events = EventLog(capacity)
        self._handlers: dict[str, Handler] = {}
        self._requests = _Requests()
        self._answered: OrderedDict[tuple[str, str], Outcome] = OrderedDict()
        self._counter = 0

    # -- wiring ---------------------------------------------------------

    def register(self, action_id: str, handler: Handler) -> None:
        """Install the handler for a catalog action.

        An action with no handler is refused as inapplicable rather than as
        unknown: the operation exists for every client at this protocol
        version, and whether this session can carry it out is a different
        question from whether it is in the catalog.
        """
        if action_id not in self._by_id:
            raise KeyError(f"no such action: {action_id}")
        with self._lock:
            self._handlers[action_id] = handler

    # -- reading --------------------------------------------------------

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot()

    def subscribe(self) -> tuple[Snapshot, Subscription]:
        """Take a snapshot and start listening, with no gap between the two.

        Both happen under the writer's lock, so no event can land after the
        snapshot was read and before the subscription exists. A client that
        did these separately would silently miss whatever fell between them.
        """
        with self._lock:
            return self._snapshot(), self._events.subscribe()

    def capabilities(self, actor: Actor) -> tuple[Capability, ...]:
        """The whole catalog, each entry marked with whether *actor* may use it."""
        return tuple(Capability(action, actor.may(action)) for action in self._catalog)

    def _snapshot(self) -> Snapshot:
        return Snapshot(self._events.instance, self._events.sequence, self._state)

    # -- writing --------------------------------------------------------

    def dispatch(
        self,
        action_id: str,
        payload: Mapping[str, object] | None = None,
        *,
        actor: Actor,
        idempotency_key: str | None = None,
    ) -> Outcome:
        """Run one action for *actor* and answer with its outcome.

        An ``idempotency_key`` makes a retry safe: the second call with a key
        this actor already used returns the first call's outcome without
        running anything. Keys are scoped to the actor, so two clients that
        both count from one cannot answer each other's requests.
        """
        with self._lock:
            remembered = self._remembered(actor, idempotency_key)
            if remembered is not None:
                return remembered
            outcome = self._dispatch(action_id, payload or {}, actor)
            self._remember(actor, idempotency_key, outcome)
            return outcome

    def _remembered(self, actor: Actor, key: str | None) -> Outcome | None:
        return None if key is None else self._answered.get((actor.id, key))

    def _remember(self, actor: Actor, key: str | None, outcome: Outcome) -> None:
        if key is None:
            return
        self._answered[(actor.id, key)] = outcome
        while len(self._answered) > IDEMPOTENCY_MEMORY:
            self._answered.popitem(last=False)

    def _dispatch(
        self, action_id: str, payload: Mapping[str, object], actor: Actor
    ) -> Outcome:
        self._counter += 1
        request_id = f"req-{self._counter}"

        action = self._by_id.get(action_id)
        if action is None:
            return Rejected(
                request_id, Rejection.UNKNOWN_ACTION, f"no such action: {action_id}"
            )
        if not actor.may(action):
            return Rejected(
                request_id,
                Rejection.FORBIDDEN,
                f"{actor.id} does not hold the {action.scope} scope",
            )
        try:
            checked = action.validate(payload)
        except InvalidPayload as error:
            return Rejected(request_id, Rejection.INVALID, str(error))

        handler = self._handlers.get(action_id)
        if handler is None:
            return Rejected(
                request_id,
                Rejection.INAPPLICABLE,
                f"{action_id} is not available in this session",
            )

        request = Request(request_id, action, checked, actor)
        try:
            effect = handler(request, self._state)
        except Inapplicable as error:
            return Rejected(request_id, Rejection.INAPPLICABLE, str(error))
        except EffectFailed as error:
            self._publish_outcome("action.failed", request_id, action_id, str(error))
            return Failed(request_id, str(error))

        if effect.settle is None:
            self._commit(effect.state)
            self._publish_applied(request_id, action_id, effect.effective)
            return Applied(request_id, effect.effective)

        displaced = self._requests.start(request_id, _Pending(action_id, effect.settle))
        self._commit(effect.state)
        if displaced is not None:
            self._publish_outcome("action.superseded", displaced, action_id)
        self._publish_outcome("action.accepted", request_id, action_id)
        return Accepted(request_id)

    # -- reconciler callbacks -------------------------------------------

    def settle(self, request_id: str, effective: object = None) -> Outcome:
        """Report that an accepted request finished, with what it settled on.

        A request that lost its slot to a newer one answers
        :class:`~.outcomes.Superseded` and changes nothing. That is the rule
        that keeps a device which finally opened from overwriting the choice
        made while it was opening, and the reason a superseded selection is
        never written to the configuration file.
        """
        with self._lock:
            pending = self._requests.finish(request_id)
            if pending is None:
                return self._unknown_request(request_id)
            if pending.superseded:
                return Superseded(request_id)
            self._commit(pending.settle(self._state, effective))
            self._publish_applied(request_id, pending.action_id, effective)
            return Applied(request_id, effective)

    def fail(self, request_id: str, detail: str) -> Outcome:
        """Report that an accepted request was attempted and did not succeed.

        Desired state is left as it was. Whether a failure should roll the
        choice back is the adapter's decision — the microphone keeps asking
        because its reconciler retries, while the far end gives up — and the
        controller inventing one rule for both would overrule whichever
        adapter it disagreed with.
        """
        with self._lock:
            pending = self._requests.finish(request_id)
            if pending is None:
                return self._unknown_request(request_id)
            if pending.superseded:
                return Superseded(request_id)
            self._publish_outcome(
                "action.failed", request_id, pending.action_id, detail
            )
            return Failed(request_id, detail)

    def _unknown_request(self, request_id: str) -> Outcome:
        if request_id in self._requests.superseded:
            return Superseded(request_id)
        return Rejected(
            request_id, Rejection.INVALID, f"no request in flight: {request_id}"
        )

    # -- publication ----------------------------------------------------

    def _commit(self, state: AppState) -> None:
        """Adopt new state and publish what changed, if anything did."""
        changed = state.changed_from(self._state)
        self._state = state
        if changed:
            self._events.publish("state.changed", changed)

    def _publish_outcome(
        self, name: str, request_id: str, action_id: str, detail: str = ""
    ) -> None:
        """Announce an outcome after the state it describes is already visible.

        State first, then the lifecycle event: a client that reacts to
        ``action.applied`` by reading the snapshot it has been maintaining
        finds the change already there rather than one event too early.
        """
        payload: dict[str, object] = {"request_id": request_id, "action": action_id}
        if detail:
            payload["detail"] = detail
        self._events.publish(name, payload)

    def _publish_applied(
        self, request_id: str, action_id: str, effective: object
    ) -> None:
        """Announce a finished effect, always carrying the effective value.

        Carried even when it is ``None``: deselecting a microphone settles on
        nothing, and an event that omitted the key would be indistinguishable
        from one that forgot to say.
        """
        self._events.publish(
            "action.applied",
            {"request_id": request_id, "action": action_id, "effective": effective},
        )
