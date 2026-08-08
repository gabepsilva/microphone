"""Capability policy: scopes granted by the runtime, plus explicit denials."""

from __future__ import annotations

from tagalong.control.actions import CATALOG, ActionSpec, Scope
from tagalong.control.actors import ActorKind, agent, local_user
from tagalong.control.policy import (
    AGENT_DENIED_ACTIONS,
    MCP_DENIED_ACTIONS,
    SOCKET_AGENT_SCOPES,
    authorizes,
    denied_actions_for_socket_client,
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


def test_client_keyed_denial_binds_mcp_not_electron() -> None:
    """#128 D12: grant-time denial — mcp FORBIDDEN, electron allowed."""
    assert "speech.read_selection" in MCP_DENIED_ACTIONS
    assert denied_actions_for_socket_client("mcp") == MCP_DENIED_ACTIONS
    assert denied_actions_for_socket_client("electron") == frozenset()
    # Blank / whitespace labels are not mcp — same empty denial as electron.
    assert denied_actions_for_socket_client("  ") == frozenset()
    assert denied_actions_for_socket_client("") == frozenset()
    assert denied_actions_for_socket_client("MCP") == frozenset()
    # strip() then exact match — trailing/leading space still counts as mcp.
    assert denied_actions_for_socket_client("mcp ") == MCP_DENIED_ACTIONS
    assert denied_actions_for_socket_client(" mcp") == MCP_DENIED_ACTIONS

    # Synthetic spec — denial is keyed on action id, not catalog membership.
    action = ActionSpec(
        "speech.read_selection",
        "Read the primary selection aloud after chrome cleanup",
        Scope.SETTINGS,
    )
    mcp_denied = denied_actions_for_socket_client("mcp")
    assert mcp_denied == frozenset({"speech.read_selection"})
    mcp = agent("mcp-1", SOCKET_AGENT_SCOPES, mcp_denied)
    electron = agent(
        "electron-1", SOCKET_AGENT_SCOPES, denied_actions_for_socket_client("electron")
    )
    # actor.denied must actually be the grant — ignoring it would leave mcp open.
    assert mcp.denied == MCP_DENIED_ACTIONS
    assert electron.denied == frozenset()
    assert not authorizes(mcp, action)
    assert authorizes(electron, action)
    # A same-scope agent with an empty denied set is allowed (denial is not kind-wide).
    assert authorizes(agent("other-1", SOCKET_AGENT_SCOPES), action)
