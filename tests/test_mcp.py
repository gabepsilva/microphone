"""MCP tool schemas are generated from the static catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from tagalong.application import bind_first_slice
from tagalong.control import Controller
from tagalong.control.actions import CATALOG
from tagalong.control.policy import SOCKET_AGENT_SCOPES
from tagalong.mcp import McpBridge, mcp_tools, tool_name
from tagalong.transport import LocalServer, socket_path
from tests.test_transport import Conversation, Speech


def test_every_catalog_action_has_one_static_tool() -> None:
    tools = mcp_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert [tool["name"] for tool in tools] == [tool_name(spec.id) for spec in CATALOG]
    assert by_name["tagalong_tts_set_enabled"]["inputSchema"]["required"] == ["enabled"]
    assert by_name["tagalong_tts_set_enabled"]["inputSchema"]["properties"][
        "enabled"
    ] == {"type": "boolean"}
    assert by_name["tagalong_message_send"]["inputSchema"]["properties"]["images"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert by_name["tagalong_attachment_upload"]["inputSchema"]["properties"][
        "data"
    ] == {
        "type": "string",
        "contentEncoding": "base64",
    }
    assert by_name["tagalong_turn_silence_set"]["inputSchema"]["properties"][
        "seconds"
    ] == {"type": "number"}
    assert by_name["tagalong_session_interrupt"]["inputSchema"]["properties"][
        "generation"
    ] == {"type": ["number", "null"]}
    assert by_name["tagalong_microphone_select"]["inputSchema"]["properties"][
        "name"
    ] == {"type": ["string", "null"]}
    assert all("command.invoke" not in tool["name"] for tool in tools)


def test_mcp_tools_can_be_limited_to_allowed_ids() -> None:
    tools = mcp_tools(allowed_ids={"tts.set_enabled", "session.quit"})
    assert [tool["name"] for tool in tools] == [
        tool_name("session.quit"),
        tool_name("tts.set_enabled"),
    ]


def test_mcp_bridge_omits_tools_capability_policy_denies(tmp_path: Path) -> None:
    from tagalong.application import (
        bind_audio_slice,
        bind_session_transcript_slice,
        bind_settings_slice,
    )
    from tagalong.attachments import AttachmentRegistry, AttachmentStore

    class Turns:
        def end_turn(self) -> None:
            return None

    class Rows:
        def transcript_entries(self):
            return []

    class Policy:
        def set_policy(self, policy: str) -> None:
            del policy

    class Silence:
        def set(self, seconds: float) -> float:
            return seconds

    class Capture:
        def select(self, name, *, on_applied=None, on_failed=None) -> bool:
            del name, on_applied, on_failed
            return True

        def set_muted(self, muted: bool) -> None:
            del muted

    class SettingsSpeech(Speech):
        def set_provider(self, provider: str) -> bool:
            del provider
            return True

    class SettingsTalk(Conversation):
        def request_model(self, model: str) -> bool:
            del model
            return True

        def request_reasoning_effort(self, effort: str) -> bool:
            del effort
            return True

    talk = SettingsTalk()
    speech = SettingsSpeech()
    attachments = AttachmentRegistry(store=AttachmentStore(directory=tmp_path / "a"))
    controller = Controller()
    bind_first_slice(controller, conversation=talk, tts=speech, attachments=attachments)
    bind_settings_slice(controller, (talk, speech, Policy(), Silence()))
    bind_audio_slice(controller, microphone=Capture(), audio=Capture())
    bind_session_transcript_slice(
        controller, (talk, Turns(), attachments, Rows()), directory=tmp_path
    )
    server = LocalServer(controller, path=tmp_path / "tagalong.sock")
    server.start()
    bridge = McpBridge.connect(server.path)
    try:
        names = {tool["name"] for tool in bridge.list_tools()}
        assert tool_name("session.quit") not in names
        assert tool_name("message.send") in names
        assert frozenset(bridge.scopes) == frozenset(
            scope.value for scope in SOCKET_AGENT_SCOPES
        )
        denied = bridge.call_tool("tagalong_session_quit", {})
        assert denied["type"] == "rejected"
        assert denied["reason"] == "forbidden"
    finally:
        bridge.close()
        server.stop()


def test_mcp_bridge_dispatches_through_the_socket(tmp_path: Path) -> None:
    controller = Controller()
    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    server = LocalServer(controller, path=tmp_path / "tagalong.sock")
    server.start()
    bridge = McpBridge.connect(server.path)
    try:
        assert bridge.actor_id.startswith("mcp-")
        assert bridge.list_tools()[0]["name"] == tool_name(CATALOG[0].id)
        outcome = bridge.call_tool("tagalong_tts_set_enabled", {"enabled": False})
        assert outcome["type"] == "applied"
        assert controller.state.tts_enabled is False
        with pytest.raises(ValueError, match="unknown tool"):
            bridge.call_tool("not_a_tool", {})
        with pytest.raises(ValueError, match="unknown tool"):
            bridge.call_tool("tagalong_missing_action", {})
    finally:
        bridge.close()
        server.stop()


def test_mcp_bridge_connects_to_the_default_runtime_socket(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    controller = Controller()
    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    server = LocalServer(controller, path=socket_path())
    server.start()
    try:
        bridge = McpBridge.connect()
        try:
            assert bridge.actor_id.startswith("mcp-")
        finally:
            bridge.close()
    finally:
        server.stop()
