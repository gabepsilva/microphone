"""Capability policy: who may invoke which catalog actions.

Scopes answer "which families of actions this connection was granted".
Explicit denials answer "which actions are refused even when the scope is
held". Together they are the load-bearing half of the issue #81 invariant —
authorization — and stay separate from runtime applicability (device missing,
generation stale, and so on).

**Where the grant comes from.** Socket peers are authenticated by uid alone.
The runtime therefore decides scopes in :func:`scopes_for_socket_client`, keyed
by connection class (the ``client`` label is a name for the actor id, not a
way to mint scopes the peer did not earn). The person at the TUI is
:func:`~.actors.local_user` and holds every scope because they own the
session. A client cannot enlarge its grant by sending a scopes field.
"""

from __future__ import annotations

from .actions import ActionSpec, Scope
from .actors import Actor, ActorKind

# Socket agents share one session with the human operator under the same-uid
# threat model. The grant is therefore the full scope set — decided here, not
# inherited by "give them everything by accident". Narrowing a future client
# class means adding a branch, not deleting a silent default.
SOCKET_AGENT_SCOPES: frozenset[Scope] = frozenset(Scope)

# Agents may hold ``session`` for interrupt and ``session.new``, but shutting
# the runtime down is the human's. Denied by policy (FORBIDDEN), not left as a
# handler-level INAPPLICABLE that MCP would still advertise.
AGENT_DENIED_ACTIONS: frozenset[str] = frozenset({"session.quit"})


def scopes_for_socket_client(client: str) -> frozenset[Scope]:
    """Return the scopes the runtime grants a same-uid socket peer.

    *client* labels the actor id (``mcp``, ``electron``, …). Today every
    socket class receives :data:`SOCKET_AGENT_SCOPES`; the parameter exists so
    a future class can be narrower without inventing a second grant path.
    """
    _ = client
    return SOCKET_AGENT_SCOPES


def authorizes(actor: Actor, action: ActionSpec) -> bool:
    """True when *actor* holds *action*'s scope and is not explicitly denied."""
    if action.scope not in actor.scopes:
        return False
    denied = actor.kind is ActorKind.AGENT and action.id in AGENT_DENIED_ACTIONS
    return not denied
