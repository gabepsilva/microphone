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
Losing the slot is the end of that request: it is retired there and then, and
only its id is kept, because the reconcilers this serves coalesce and an
overtaken request is exactly the one that may never call back.

**Every attempt is on the record.** Each lifecycle event names the actor whose
connection authenticated it and the moment it happened, refusals included. A
record of successes only cannot show an agent being turned away, which is the
question an operator asks first.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime

from .actions import (
    CATALOG,
    PROTOCOL_VERSION,
    ActionSpec,
    Capability,
    InvalidPayload,
)
from .actors import Actor
from .events import DEFAULT_CAPACITY, EventLog, Subscription, utc_now
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
class _Answer:
    """What an idempotency key was used for, and where that request stands.

    The action and payload are kept so a reused key can be recognised as the
    same request rather than answered with an outcome belonging to a different
    one. The outcome is replaced as the request progresses, so a retry after a
    disconnect learns the terminal result rather than the acceptance forever.
    """

    action_id: str
    payload: Mapping[str, object]
    outcome: Outcome


@dataclass(frozen=True)
class _Pending:
    """A request a reconciler still owes an answer for, and who asked for it."""

    action_id: str
    actor_id: str
    settle: Callable[[AppState, object], AppState]


@dataclass
class _Requests:
    """In-flight requests, one slot per action, plus what recently lost its slot."""

    pending: dict[str, _Pending] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)
    superseded: OrderedDict[str, None] = field(default_factory=OrderedDict)

    def start(self, request_id: str, pending: _Pending) -> tuple[str, _Pending] | None:
        """Take the slot for this action, returning whoever held it before.

        Losing the slot is terminal, so the displaced request is retired here
        rather than left waiting for a callback. The reconcilers this serves
        coalesce on purpose — a run of picker keystrokes is meant to cost one
        device open — so an overtaken request is precisely the one that may
        never be reported. Keeping it would hold its settle closure for the
        rest of the session, once per keystroke.
        """
        displaced_id = self.slots.get(pending.action_id)
        displaced = None
        if displaced_id is not None:
            displaced = (displaced_id, self.pending.pop(displaced_id))
            self._remember_superseded(displaced_id)
        self.slots[pending.action_id] = request_id
        self.pending[request_id] = pending
        return displaced

    def finish(self, request_id: str) -> _Pending | None:
        """Retire a request and free the slot it held.

        A pending request always holds its action's slot: taking the slot is
        what makes it pending, and losing it retires it in ``start``. So there
        is no case here of a pending request whose slot belongs to someone
        else, and freeing it unconditionally is the invariant rather than an
        assumption.
        """
        pending = self.pending.pop(request_id, None)
        if pending is None:
            return None
        del self.slots[pending.action_id]
        return pending

    def _remember_superseded(self, request_id: str) -> None:
        """Keep the id, not the request, so a late callback gets an answer."""
        self.superseded[request_id] = None
        while len(self.superseded) > SUPERSEDED_MEMORY:
            self.superseded.popitem(last=False)


class Controller:
    """The ordered core: validate, authorize, change state, publish."""

    def __init__(
        self,
        state: AppState | None = None,
        *,
        catalog: tuple[ActionSpec, ...] = CATALOG,
        capacity: int = DEFAULT_CAPACITY,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._lock = threading.Lock()
        self._state = AppState() if state is None else state
        self._catalog = catalog
        self._by_id = {action.id: action for action in catalog}
        self._events = EventLog(capacity, clock)
        self._handlers: dict[str, Handler] = {}
        self._requests = _Requests()
        self._answered: OrderedDict[tuple[str, str], _Answer] = OrderedDict()
        # Which key, if any, is waiting on a request that has not ended yet.
        self._keyed: dict[str, tuple[str, str]] = {}
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

        An ``idempotency_key`` makes a retry safe: repeating a key this actor
        already used for the same request answers with what that request has
        come to, without running anything again. Keys are scoped to the actor,
        so two clients that both count from one cannot answer each other's
        requests, and a key is bound to the action and payload it was first
        used for — reusing it for something else is refused rather than
        answered with an outcome that belongs to a different operation.

        What comes back is the request's latest outcome, not a frozen copy of
        the first answer: a keyed selection that has since applied or been
        superseded says so. A client that retries after a disconnect therefore
        learns the terminal result even though the event announcing it was
        published before the client resubscribed.
        """
        with self._lock:
            sent = dict(payload or {})
            slot = None if idempotency_key is None else (actor.id, idempotency_key)
            if slot is not None:
                answered = self._answered.get(slot)
                if answered is not None:
                    return self._replay(answered, slot, action_id, sent, actor)
            outcome = self._dispatch(action_id, sent, actor)
            if slot is not None:
                self._remember(slot, _Answer(action_id, sent, outcome))
            return outcome

    def _replay(
        self,
        answered: _Answer,
        slot: tuple[str, str],
        action_id: str,
        payload: Mapping[str, object],
        actor: Actor,
    ) -> Outcome:
        """Answer a repeated key, or refuse it if it now means something else."""
        if (answered.action_id, answered.payload) == (action_id, payload):
            return answered.outcome
        return self._reject(
            self._next_id(),
            action_id,
            actor,
            Rejection.INVALID,
            f"idempotency key {slot[1]!r} was already used for "
            f"{answered.action_id} with a different payload",
        )

    def _remember(self, slot: tuple[str, str], answer: _Answer) -> None:
        """Record what this key was used for, and what it is still waiting on.

        Only an accepted request is tracked in ``_keyed``: every other outcome
        is already terminal, so there is nothing left to update it with. That
        keeps the reverse index the size of the in-flight set — at most one
        request per action — rather than growing with every key ever used.
        """
        self._answered[slot] = answer
        if isinstance(answer.outcome, Accepted):
            self._keyed[answer.outcome.request_id] = slot
        while len(self._answered) > IDEMPOTENCY_MEMORY:
            self._answered.popitem(last=False)

    def _conclude(self, request_id: str, outcome: Outcome) -> Outcome:
        """Record a terminal outcome against the key that asked for it, if any.

        The key may have been evicted while its request was still in flight —
        a long-running session can use a great many keys — so the entry has to
        still be there, not merely have been there once.
        """
        slot = self._keyed.pop(request_id, None)
        if slot is not None and slot in self._answered:
            self._answered[slot] = replace(self._answered[slot], outcome=outcome)
        return outcome

    def _next_id(self) -> str:
        self._counter += 1
        return f"req-{self._counter}"

    def _dispatch(
        self, action_id: str, payload: Mapping[str, object], actor: Actor
    ) -> Outcome:
        request_id = self._next_id()

        action = self._by_id.get(action_id)
        if action is None:
            return self._reject(
                request_id,
                action_id,
                actor,
                Rejection.UNKNOWN_ACTION,
                f"no such action: {action_id}",
            )
        if not actor.may(action):
            return self._reject(
                request_id,
                action_id,
                actor,
                Rejection.FORBIDDEN,
                f"{actor.id} does not hold the {action.scope} scope",
            )
        try:
            checked = action.validate(payload)
        except InvalidPayload as error:
            return self._reject(
                request_id, action_id, actor, Rejection.INVALID, str(error)
            )

        handler = self._handlers.get(action_id)
        if handler is None:
            return self._reject(
                request_id,
                action_id,
                actor,
                Rejection.INAPPLICABLE,
                f"{action_id} is not available in this session",
            )

        request = Request(request_id, action, checked, actor)
        try:
            effect = handler(request, self._state)
        except Inapplicable as error:
            return self._reject(
                request_id, action_id, actor, Rejection.INAPPLICABLE, str(error)
            )
        except EffectFailed as error:
            self._publish_outcome(
                "action.failed", request_id, action_id, actor.id, str(error)
            )
            return Failed(request_id, str(error))

        if effect.settle is None:
            self._commit(effect.state)
            self._publish_applied(request_id, action_id, actor.id, effect.effective)
            return Applied(request_id, effect.effective)

        displaced = self._requests.start(
            request_id, _Pending(action_id, actor.id, effect.settle)
        )
        self._commit(effect.state)
        if displaced is not None:
            self._supersede(*displaced)
        self._publish_outcome("action.accepted", request_id, action_id, actor.id)
        return Accepted(request_id)

    def _supersede(self, request_id: str, pending: _Pending) -> None:
        """Retire the request that just lost its slot, under its own actor."""
        self._publish_outcome(
            "action.superseded", request_id, pending.action_id, pending.actor_id
        )
        self._conclude(request_id, Superseded(request_id))

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
            self._commit(pending.settle(self._state, effective))
            self._publish_applied(
                request_id, pending.action_id, pending.actor_id, effective
            )
            return self._conclude(request_id, Applied(request_id, effective))

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
            self._publish_outcome(
                "action.failed", request_id, pending.action_id, pending.actor_id, detail
            )
            return self._conclude(request_id, Failed(request_id, detail))

    def _unknown_request(self, request_id: str) -> Outcome:
        """Answer a reconciler about a request this controller no longer holds.

        A request that lost its slot was retired the moment it lost it, so it
        is recognised here rather than in ``pending``. Anything else is a
        report about a request that was never accepted.
        """
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
        self,
        name: str,
        request_id: str,
        action_id: str,
        actor_id: str,
        detail: str = "",
    ) -> None:
        """Announce an outcome after the state it describes is already visible.

        State first, then the lifecycle event: a client that reacts to
        ``action.applied`` by reading the snapshot it has been maintaining
        finds the change already there rather than one event too early.

        Every lifecycle event names the actor whose request it belongs to, and
        the event log stamps it. With the request id, that is the audit record:
        who attempted what, when, and how it ended. The actor is the one the
        connection authenticated — a payload never supplies it — and it is
        carried through the pending lifecycle so a result reported minutes
        later is still attributed to whoever asked for it.
        """
        payload: dict[str, object] = {
            "request_id": request_id,
            "action": action_id,
            "actor": actor_id,
        }
        if detail:
            payload["detail"] = detail
        self._events.publish(name, payload)

    def _reject(
        self,
        request_id: str,
        action_id: str,
        actor: Actor,
        reason: Rejection,
        detail: str,
    ) -> Rejected:
        """Refuse a request, and record the attempt.

        A refusal changes no state, so it publishes no ``state.changed``. It is
        still published: an audit trail that omits what was turned away is
        missing the half an operator needs, and an agent whose scopes are wrong
        is invisible in a record of successes only.
        """
        self._events.publish(
            "action.rejected",
            {
                "request_id": request_id,
                "action": action_id,
                "actor": actor.id,
                "reason": reason,
                "detail": detail,
            },
        )
        return Rejected(request_id, reason, detail)

    def _publish_applied(
        self, request_id: str, action_id: str, actor_id: str, effective: object
    ) -> None:
        """Announce a finished effect, always carrying the effective value.

        Carried even when it is ``None``: deselecting a microphone settles on
        nothing, and an event that omitted the key would be indistinguishable
        from one that forgot to say.
        """
        self._events.publish(
            "action.applied",
            {
                "request_id": request_id,
                "action": action_id,
                "actor": actor_id,
                "effective": effective,
            },
        )
