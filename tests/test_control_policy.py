"""Capability policy: scopes granted by the runtime, plus explicit denials."""

from __future__ import annotations

from tagalong.control.actions import CATALOG, ActionSpec, Scope
from tagalong.control.actors import ActorKind, agent, local_user
from tagalong.control.policy import (
    AGENT_DENIED_ACTIONS,
    SOCKET_AGENT_SCOPES,
    authorizes,
    scopes_for_socket_client,
)


def _action(action_id: str) -> ActionSpec:
    return next(spec for spec in CATALOG if spec.id == action_id)


def test_socket_scopes_are_an_explicit_runtime_grant() -> None:
    assert scopes_for_socket_client("mcp") == SOCKET_AGENT_SCOPES
    assert scopes_for_socket_client("electron") == frozenset(Scope)
    # The label does not mint a narrower or wider set by itself today.
    assert scopes_for_socket_client("mcp") == scopes_for_socket_client("other")


def test_authorizes_requires_scope_and_honours_agent_denials() -> None:
    human = local_user()
    bot = agent("bot", {Scope.SESSION, Scope.CONVERSE})

    assert authorizes(human, _action("session.quit"))
    assert authorizes(bot, _action("session.interrupt"))
    assert not authorizes(bot, _action("session.quit"))
    assert _action("session.quit").id in AGENT_DENIED_ACTIONS
    assert not authorizes(agent("bot", {Scope.TRANSCRIPT}), _action("message.send"))


def test_every_catalog_action_is_decidable_for_a_socket_agent() -> None:
    caller = agent("mcp-1", SOCKET_AGENT_SCOPES)
    assert caller.kind is ActorKind.AGENT
    decisions = {spec.id: authorizes(caller, spec) for spec in CATALOG}
    assert decisions["session.quit"] is False
    assert all(
        allowed
        for action_id, allowed in decisions.items()
        if action_id != "session.quit"
    )
