"""MCP adapter: catalog-derived tool schemas over the same local transport.

Input schemas are generated from the static :data:`~.control.actions.CATALOG`
so a protocol version cannot drift between human and agent surfaces. Which
tools a connected session *lists* is capability policy: only actions the
actor is authorized to invoke. Transient applicability (device missing, stale
generation) stays a per-call answer — that is what the RFC kept dynamic.
Advertising a tool the policy will FORBIDDEN is the defect this adapter
refuses to repeat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .control.actions import CATALOG, ActionSpec, Kind, Parameter
from .transport import LocalClient, socket_path


def tool_name(action_id: str) -> str:
    """``tts.set_enabled`` → ``tagalong_tts_set_enabled``."""
    return "tagalong_" + action_id.replace(".", "_")


def _schema_type(parameter: Parameter) -> dict[str, Any]:
    if parameter.kind is Kind.FLAG:
        types: object = "boolean"
    elif parameter.kind is Kind.NUMBER:
        types = "number"
    elif parameter.kind is Kind.IDS:
        return {"type": "array", "items": {"type": "string"}}
    elif parameter.kind is Kind.DATA:
        return {"type": "string", "contentEncoding": "base64"}
    else:
        types = "string"
    if parameter.nullable:
        return {"type": [types, "null"]}
    return {"type": types}


def tool_schema(spec: ActionSpec) -> dict[str, Any]:
    properties = {
        parameter.name: _schema_type(parameter) for parameter in spec.parameters
    }
    required = [parameter.name for parameter in spec.parameters if parameter.required]
    return {
        "name": tool_name(spec.id),
        "description": spec.summary,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def mcp_tools(
    catalog: tuple[ActionSpec, ...] = CATALOG,
    *,
    allowed_ids: frozenset[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Tool schemas for *catalog*, optionally limited to *allowed_ids*.

    Omit *allowed_ids* for the full static protocol surface (tests, docs).
    A live session passes the ids capability policy marks allowed for its
    actor so an agent is never offered a tool it cannot invoke.
    """
    specs = (
        catalog
        if allowed_ids is None
        else tuple(spec for spec in catalog if spec.id in allowed_ids)
    )
    return [tool_schema(spec) for spec in specs]


class McpBridge:
    """One MCP client session talking to a live TagAlong runtime."""

    def __init__(self, client: LocalClient) -> None:
        self._client = client
        hello = self._client.call("initialize", {"client": "mcp"})
        self.actor_id = hello["actor_id"]
        self.scopes = tuple(hello["scopes"])

    @classmethod
    def connect(cls, path=None) -> McpBridge:
        return cls(LocalClient(path if path is not None else socket_path()))

    def list_tools(self) -> list[dict[str, Any]]:
        capabilities = self._client.call("capabilities")
        allowed = {entry["id"] for entry in capabilities["actions"] if entry["allowed"]}
        return mcp_tools(allowed_ids=allowed)

    def call_tool(self, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        action_id = _action_id_from_tool(name)
        return self._client.call(
            "dispatch", {"action": action_id, "payload": dict(arguments)}
        )

    def close(self) -> None:
        self._client.close()


def _action_id_from_tool(name: str) -> str:
    if not name.startswith("tagalong_"):
        raise ValueError(f"unknown tool: {name}")
    for spec in CATALOG:
        if tool_name(spec.id) == name:
            return spec.id
    raise ValueError(f"unknown tool: {name}")
