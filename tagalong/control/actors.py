"""Who is acting, and what they are allowed to do.

An actor is derived from the connection that carried the request and never
read out of the request itself. That rule is the whole point of the type: a
payload field naming its own caller is a claim, and an agent that can write
its own identity can grant itself any scope it likes.

Actor is not the same question as source. Actor answers "who is allowed to do
this", so it governs authorization and the audit trail. Source answers "how
should this content be interpreted", so an agent's message enters the
transcript as ``Agent`` rather than passing for something a human typed. The
two are separate because a trusted actor can still send content that must not
be read as a person speaking in the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .actions import ActionSpec, Scope


class ActorKind(StrEnum):
    """What kind of principal a connection authenticated.

    This is not a scope and not a transcript source. Kind answers how the
    session should treat the actor's content — a human message is ``Text``, an
    agent message is ``Agent`` — while scopes answer what the actor may ask
    the session to do. An agent granted ``converse`` still cannot impersonate
    a person in the room.
    """

    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass(frozen=True)
class Actor:
    """An authenticated caller and the scopes its connection was granted."""

    id: str
    kind: ActorKind = ActorKind.HUMAN
    scopes: frozenset[Scope] = field(default_factory=frozenset)

    def may(self, action: ActionSpec) -> bool:
        """Return whether this actor holds the scope *action* requires."""
        return action.scope in self.scopes


def local_user(id: str = "local") -> Actor:
    """The person sitting at the session: every scope, because they own it."""
    return Actor(id, ActorKind.HUMAN, frozenset(Scope))


def agent(id: str, scopes: frozenset[Scope] | set[Scope]) -> Actor:
    """An external agent, holding only the scopes it was granted.

    There is no default here on purpose. An agent's authority is a decision
    someone made, and a signature that supplies one silently would make the
    unconsidered case the permissive one.
    """
    return Actor(id, ActorKind.AGENT, frozenset(scopes))
