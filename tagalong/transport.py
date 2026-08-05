"""Secure local JSON-RPC over a Unix socket.

The in-process controller is the session. This module is how a second writer
— Electron, an MCP adapter, a CLI — reaches it without inventing a parallel
API. Framing is NDJSON, the directory is mode ``0700``, the socket is ``0600``,
and every accepted connection is checked with ``SO_PEERCRED`` so a peer from
another uid never speaks to the session.

There is no ``/tmp`` fallback. If ``XDG_RUNTIME_DIR`` is unset the server
refuses to start, rather than listen somewhere any local user can connect.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .control import (
    Accepted,
    Actor,
    Applied,
    Controller,
    Failed,
    Outcome,
    Rejected,
    Snapshot,
    Superseded,
    agent,
)
from .control.actions import PROTOCOL_VERSION
from .discovery import list_commands

SO_PEERCRED = getattr(socket, "SO_PEERCRED", 17)
_CRED_FORMAT = "3i"
# Base64 expands by 4/3; 20 MiB of image bytes become ~28 MiB on the wire.
# Leave room for the JSON-RPC envelope around attachment.upload. Same-uid
# only, so a larger frame is not a cross-user DoS surface.
MAX_FRAME = 30 * 1024 * 1024
ACCEPT_TIMEOUT = 0.25
JOIN_TIMEOUT = 2.0
JSONRPC = "2.0"
_FRAME_LIMIT_LABEL = f"{MAX_FRAME // (1024 * 1024)} MiB"


class TransportError(Exception):
    """The socket path, peer, or frame is not acceptable."""


@dataclass(frozen=True)
class Peer:
    """Kernel-derived identity of one connected client."""

    pid: int
    uid: int
    gid: int


@dataclass
class _Session:
    """Per-connection RPC state."""

    peer: Peer
    actor: Actor | None = None
    subscribed: Any = None


def runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``$XDG_RUNTIME_DIR/tagalong`` — missing XDG is a hard error, not ``/tmp``."""
    env = os.environ if environ is None else environ
    root = env.get("XDG_RUNTIME_DIR")
    if not root:
        raise TransportError("XDG_RUNTIME_DIR is unset; refusing a /tmp socket")
    return Path(root) / "tagalong"


def socket_path(environ: Mapping[str, str] | None = None) -> Path:
    return runtime_dir(environ) / "tagalong.sock"


def prepare_runtime_dir(path: Path) -> None:
    """Create the runtime directory and force ``0700`` even if it already existed."""
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def read_peer(connection: socket.socket) -> Peer:
    raw = connection.getsockopt(
        socket.SOL_SOCKET, SO_PEERCRED, struct.calcsize(_CRED_FORMAT)
    )
    pid, uid, gid = struct.unpack(_CRED_FORMAT, raw)
    return Peer(pid=pid, uid=uid, gid=gid)


def encode_frame(payload: Mapping[str, object]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"))
    frame = body.encode("utf-8") + b"\n"
    if len(frame) > MAX_FRAME:
        raise TransportError(f"frame exceeds the {_FRAME_LIMIT_LABEL} limit")
    return frame


def decode_frame(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_FRAME:
        raise TransportError(f"frame exceeds the {_FRAME_LIMIT_LABEL} limit")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransportError("frame is not JSON") from error
    if not isinstance(payload, dict):
        raise TransportError("frame must be a JSON object")
    return payload


def snapshot_payload(snapshot: Snapshot) -> dict[str, object]:
    return {
        "instance": snapshot.instance,
        "sequence": snapshot.sequence,
        "protocol_version": snapshot.protocol_version,
        "state": asdict(snapshot.state),
    }


def outcome_payload(outcome: Outcome) -> dict[str, object]:
    if isinstance(outcome, Applied):
        return {
            "type": "applied",
            "request_id": outcome.request_id,
            "effective": outcome.effective,
        }
    if isinstance(outcome, Accepted):
        return {"type": "accepted", "request_id": outcome.request_id}
    if isinstance(outcome, Failed):
        return {
            "type": "failed",
            "request_id": outcome.request_id,
            "detail": outcome.detail,
        }
    if isinstance(outcome, Superseded):
        return {"type": "superseded", "request_id": outcome.request_id}
    if isinstance(outcome, Rejected):
        return {
            "type": "rejected",
            "request_id": outcome.request_id,
            "reason": outcome.reason.value,
            "detail": outcome.detail,
        }
    raise TransportError(f"unserialisable outcome: {type(outcome)!r}")


def actor_for_client(client: str, peer: Peer) -> Actor:
    """Socket peers are agents. The client name is a label, not a capability.

    The transport authenticates uid, not which program connected. A self-asserted
    ``client`` string therefore cannot mint :class:`ActorKind.HUMAN` — that
    would let any same-uid process pose as a person in the room. Scopes come
    from :func:`~.control.policy.scopes_for_socket_client`, not from the
    request. The in-process TUI is still :func:`local_user`.
    """
    from .control.policy import scopes_for_socket_client

    label = client.strip() or "local"
    return agent(f"{label}-{peer.uid}", scopes_for_socket_client(label))


class LocalServer:
    """Accept same-uid JSON-RPC connections and dispatch them on *controller*."""

    def __init__(
        self,
        controller: Controller,
        *,
        path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._controller = controller
        self._path = path if path is not None else socket_path(environ)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._connections: list[socket.socket] = []
        self._connections_lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def start(self) -> None:
        prepare_runtime_dir(self._path.parent)
        if self._path.exists():
            self._path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self._path))
        os.chmod(self._path, 0o600)
        listener.listen(16)
        listener.settimeout(ACCEPT_TIMEOUT)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve, name="tagalong-transport", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        with self._connections_lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
        if self._thread is not None:
            self._thread.join(timeout=JOIN_TIMEOUT)
        if self._path.exists():
            self._path.unlink()
        self._listener = None
        self._thread = None

    def _serve(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                connection, _ = self._accept(listener)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            worker = threading.Thread(
                target=self._handle, args=(connection,), daemon=True
            )
            worker.start()

    def _accept(self, listener: socket.socket):
        return listener.accept()

    def _recv(self, connection: socket.socket) -> bytes:
        return connection.recv(4096)

    def _handle(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.append(connection)
        try:
            peer = read_peer(connection)
            if peer.uid != os.getuid():
                return
            connection.settimeout(ACCEPT_TIMEOUT)
            buffer = b""
            session = _Session(peer=peer)
            while not self._stop.is_set():
                try:
                    chunk = self._recv(connection)
                except TimeoutError:
                    continue
                if not chunk:
                    return
                buffer += chunk
                if len(buffer) > MAX_FRAME and b"\n" not in buffer:
                    return
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    response = self._dispatch(decode_frame(line), session)
                    connection.sendall(encode_frame(response))
        except (TransportError, OSError):
            return
        finally:
            connection.close()
            with self._connections_lock:
                if connection in self._connections:
                    self._connections.remove(connection)

    def _dispatch(
        self, request: Mapping[str, Any], session: _Session
    ) -> dict[str, object]:
        rpc_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(rpc_id, -32602, "params must be an object")
        try:
            return self._call_method(method, rpc_id, params, session)
        except (TypeError, ValueError, KeyError) as error:
            return _error(rpc_id, -32002, str(error))

    def _call_method(
        self,
        method: object,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
    ) -> dict[str, object]:
        if method == "initialize":
            return self._rpc_initialize(rpc_id, params, session)
        actor = session.actor
        if actor is None:
            return _error(rpc_id, -32000, "initialize first")
        handlers = {
            "snapshot": self._rpc_snapshot,
            "capabilities": self._rpc_capabilities,
            "commands.list": self._rpc_commands,
            "subscribe": self._rpc_subscribe,
            "poll": self._rpc_poll,
            "dispatch": self._rpc_dispatch,
        }
        handler = handlers.get(method) if isinstance(method, str) else None
        if handler is None:
            return _error(rpc_id, -32601, f"unknown method: {method}")
        return handler(rpc_id, params, session, actor)

    def _rpc_initialize(
        self, rpc_id: object, params: Mapping[str, Any], session: _Session
    ) -> dict[str, object]:
        if session.actor is not None:
            return _error(rpc_id, -32003, "already initialized")
        session.actor = actor_for_client(
            str(params.get("client") or "local"), session.peer
        )
        return _result(
            rpc_id,
            {
                "protocol_version": PROTOCOL_VERSION,
                "actor_id": session.actor.id,
                "actor_kind": session.actor.kind.value,
                "scopes": sorted(scope.value for scope in session.actor.scopes),
            },
        )

    def _rpc_snapshot(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del params, session, actor
        return _result(rpc_id, snapshot_payload(self._controller.snapshot()))

    def _rpc_capabilities(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del params, session
        entries = [
            {"id": entry.action.id, "allowed": entry.allowed}
            for entry in self._controller.capabilities(actor)
        ]
        return _result(rpc_id, {"actions": entries})

    def _rpc_commands(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del params, session, actor
        listing = [
            {
                "name": entry.name,
                "summary": entry.summary,
                "aliases": list(entry.aliases),
                "action_id": entry.action_id,
            }
            for entry in list_commands()
        ]
        return _result(rpc_id, {"commands": listing})

    def _rpc_subscribe(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del params, actor
        snapshot, session.subscribed = self._controller.subscribe()
        return _result(rpc_id, snapshot_payload(snapshot))

    def _rpc_poll(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del params, actor
        if session.subscribed is None:
            return _error(rpc_id, -32001, "not subscribed")
        events = [
            {
                "sequence": event.sequence,
                "name": event.name,
                "payload": json_ready(dict(event.payload)),
            }
            for event in session.subscribed.drain()
        ]
        return _result(rpc_id, {"events": events})

    def _rpc_dispatch(
        self,
        rpc_id: object,
        params: Mapping[str, Any],
        session: _Session,
        actor: Actor,
    ) -> dict[str, object]:
        del session
        payload = params.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return _error(rpc_id, -32602, "payload must be an object")
        key = params.get("idempotency_key")
        outcome = self._controller.dispatch(
            str(params.get("action") or ""),
            payload,
            actor=actor,
            idempotency_key=None if key is None else str(key),
        )
        return _result(rpc_id, outcome_payload(outcome))


def _result(rpc_id: object, result: object) -> dict[str, object]:
    return {"jsonrpc": JSONRPC, "id": rpc_id, "result": result}


def _error(rpc_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": JSONRPC,
        "id": rpc_id,
        "error": {"code": code, "message": message},
    }


class LocalClient:
    """In-process test / MCP helper that speaks the same frames as Electron."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(path))
        self._buffer = b""

    def close(self) -> None:
        self._sock.close()

    def call(self, method: str, params: Mapping[str, object] | None = None) -> Any:
        request = {
            "jsonrpc": JSONRPC,
            "id": method,
            "method": method,
            "params": dict(params or {}),
        }
        try:
            self._sock.sendall(encode_frame(request))
            return self._read_result()
        except OSError as error:
            raise TransportError("connection closed") from error

    def _read_result(self) -> Any:
        while True:
            line = self._readline()
            payload = decode_frame(line)
            if payload.get("method") == "event":
                continue
            if "error" in payload:
                message = payload["error"]["message"]
                raise TransportError(message)
            return payload.get("result")

    def _readline(self) -> bytes:
        try:
            while b"\n" not in self._buffer:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise TransportError("connection closed")
                self._buffer += chunk
        except OSError as error:
            raise TransportError("connection closed") from error
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line


def json_ready(value: object) -> object:
    """Turn in-process event values into JSON-encodable ones."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


class EventPump:
    """Drain a controller subscription onto a display, off the writer lock."""

    def __init__(
        self,
        drain: Callable[[], Any],
        apply: Callable[[Mapping[str, object]], None],
    ) -> None:
        self._drain = drain
        self._apply = apply
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="tagalong-event-pump", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=JOIN_TIMEOUT)

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.pump()

    def pump(self) -> None:
        for event in self._drain():
            if event.name == "state.changed":
                self._apply(dict(event.payload))
