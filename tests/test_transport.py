"""Local JSON-RPC transport: framing, peer checks, and dispatch."""

from __future__ import annotations

import os
import socket
import stat
import time
from pathlib import Path
from typing import cast

import pytest

from tagalong.application import bind_first_slice
from tagalong.control import (
    Accepted,
    Controller,
    Failed,
    Outcome,
    Superseded,
    local_user,
)
from tagalong.control.actions import PROTOCOL_VERSION
from tagalong.control.outcomes import Applied, Rejected, Rejection
from tagalong.transport import (
    EventPump,
    LocalClient,
    LocalServer,
    Peer,
    TransportError,
    actor_for_client,
    apply_state_fragment,
    decode_frame,
    encode_frame,
    outcome_payload,
    prepare_runtime_dir,
    read_peer,
    runtime_dir,
    snapshot_payload,
    socket_path,
)


class Speech:
    def __init__(self) -> None:
        self.enabled = True

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = enabled
        return True


class Conversation:
    generation = 1

    def ingest(self, speaker, text, respond, images=()):
        del speaker, text, respond, images

    def interrupt(self) -> None:
        return None

    def start_fresh_thread(self):
        return None

    def adopt_fresh_thread(self, started) -> None:
        del started


def wired(tmp_path: Path) -> tuple[Controller, LocalServer, LocalClient]:
    controller = Controller()
    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    server = LocalServer(controller, path=tmp_path / "tagalong.sock")
    server.start()
    client = LocalClient(server.path)
    return controller, server, client


def test_runtime_dir_refuses_a_tmp_fallback() -> None:
    with pytest.raises(TransportError, match="XDG_RUNTIME_DIR"):
        runtime_dir({})


def test_runtime_dir_and_socket_path_live_under_xdg(tmp_path: Path) -> None:
    environ = {"XDG_RUNTIME_DIR": str(tmp_path)}

    assert runtime_dir(environ) == tmp_path / "tagalong"
    assert socket_path(environ) == tmp_path / "tagalong" / "tagalong.sock"


def test_prepare_runtime_dir_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "tagalong"
    path.mkdir()
    os.chmod(path, 0o777)
    prepare_runtime_dir(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_frames_are_ndjson_and_capped() -> None:
    frame = encode_frame({"ok": True})

    assert frame.endswith(b"\n")
    assert decode_frame(frame[:-1]) == {"ok": True}
    with pytest.raises(TransportError, match="JSON"):
        decode_frame(b"not-json")
    with pytest.raises(TransportError, match="JSON"):
        decode_frame(b"\xff")
    with pytest.raises(TransportError, match="JSON object"):
        decode_frame(b"[]")
    with pytest.raises(TransportError, match="1 MiB"):
        decode_frame(b"x" * (1_048_576 + 1))
    with pytest.raises(TransportError, match="1 MiB"):
        encode_frame({"blob": "x" * 1_048_576})


def test_outcomes_round_trip_to_json() -> None:
    assert outcome_payload(Applied("req-1", False)) == {
        "type": "applied",
        "request_id": "req-1",
        "effective": False,
    }
    assert outcome_payload(Accepted("req-3")) == {
        "type": "accepted",
        "request_id": "req-3",
    }
    assert outcome_payload(Failed("req-4", "broke")) == {
        "type": "failed",
        "request_id": "req-4",
        "detail": "broke",
    }
    assert outcome_payload(Superseded("req-5")) == {
        "type": "superseded",
        "request_id": "req-5",
    }
    assert outcome_payload(Rejected("req-2", Rejection.FORBIDDEN, "no"))["reason"] == (
        "forbidden"
    )

    class UnknownOutcome:
        request_id = "req-x"

    with pytest.raises(TransportError, match="unserialisable"):
        outcome_payload(cast(Outcome, UnknownOutcome()))


def test_a_same_uid_client_can_dispatch_tts(tmp_path: Path) -> None:
    controller, server, client = wired(tmp_path)
    try:
        hello = client.call("initialize", {"client": "electron"})
        assert hello["protocol_version"] == PROTOCOL_VERSION
        assert hello["actor_kind"] == "human"

        snapped = client.call("snapshot")
        assert snapped["state"]["tts_enabled"] is True

        outcome = client.call(
            "dispatch",
            {
                "action": "tts.set_enabled",
                "payload": {"enabled": False},
                "idempotency_key": "tts-off",
            },
        )
        assert outcome == {
            "type": "applied",
            "request_id": "req-1",
            "effective": False,
        }
        replay = client.call(
            "dispatch",
            {
                "action": "tts.set_enabled",
                "payload": {"enabled": False},
                "idempotency_key": "tts-off",
            },
        )
        assert replay["request_id"] == "req-1"
        assert controller.state.tts_enabled is False
        assert (server.path.stat().st_mode & 0o777) == 0o600
    finally:
        client.close()
        server.stop()
    assert not server.path.exists()


def test_commands_list_is_available_over_the_socket(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "mcp"})
        listing = client.call("commands.list")
        assert listing["commands"][0]["action_id"] == "session.new"
        caps = client.call("capabilities")
        assert any(entry["id"] == "session.new" for entry in caps["actions"])
    finally:
        client.close()
        server.stop()


def test_subscribe_then_poll_sees_state_changed(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        client.call(
            "dispatch", {"action": "tts.set_enabled", "payload": {"enabled": False}}
        )
        polled = client.call("poll")
        names = [event["name"] for event in polled["events"]]
        assert "state.changed" in names
        assert "action.applied" in names
    finally:
        client.close()
        server.stop()


def test_poll_without_subscribe_is_an_error(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        with pytest.raises(TransportError, match="not subscribed"):
            client.call("poll")
    finally:
        client.close()
        server.stop()


def test_unknown_method_is_a_json_rpc_error(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        hello = client.call("initialize")
        assert hello["actor_id"].startswith("local-")
        with pytest.raises(TransportError, match="unknown method"):
            client.call("explode")
    finally:
        client.close()
        server.stop()


def test_dispatch_before_initialize_is_rejected(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        with pytest.raises(TransportError, match="initialize first"):
            client.call("snapshot")
    finally:
        client.close()
        server.stop()


def test_malformed_params_and_payloads_are_rejected(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        raw.connect(str(server.path))
        raw.sendall(b"\n")
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {"client": "electron"},
                }
            )
        )
        assert b"protocol_version" in raw.recv(4096)
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": "bad-params",
                    "method": "snapshot",
                    "params": [],
                }
            )
        )
        assert b"params must be an object" in raw.recv(4096)
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": "null-params",
                    "method": "snapshot",
                    "params": None,
                }
            )
        )
        assert b"protocol_version" in raw.recv(4096)
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": "bad-payload",
                    "method": "dispatch",
                    "params": {"action": "tts.set_enabled", "payload": []},
                }
            )
        )
        assert b"payload must be an object" in raw.recv(4096)
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": "null-payload",
                    "method": "dispatch",
                    "params": {"action": "session.interrupt", "payload": None},
                }
            )
        )
        assert b'"type":"applied"' in raw.recv(4096)
        raw.sendall(
            encode_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": 12,
                    "params": {},
                }
            )
        )
        assert b"unknown method" in raw.recv(4096)
        raw.sendall(b"not-json\n")
        assert raw.recv(4096) == b""
    finally:
        raw.close()
        client.close()
        server.stop()


def test_handler_exceptions_become_rpc_errors(tmp_path: Path, monkeypatch) -> None:
    controller, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        monkeypatch.setattr(
            controller,
            "snapshot",
            lambda: (_ for _ in ()).throw(ValueError("boom")),
        )
        with pytest.raises(TransportError, match="boom"):
            client.call("snapshot")
    finally:
        client.close()
        server.stop()


def test_a_stale_socket_is_replaced_on_start(tmp_path: Path) -> None:
    path = tmp_path / "tagalong.sock"
    path.write_text("stale", encoding="utf-8")
    server = LocalServer(Controller(), path=path)
    server.start()
    try:
        assert path.is_socket()
    finally:
        server.stop()


def test_a_foreign_uid_is_dropped(tmp_path: Path, monkeypatch) -> None:
    controller = Controller()
    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    server = LocalServer(controller, path=tmp_path / "tagalong.sock")
    server.start()
    real_uid = os.getuid()
    monkeypatch.setattr("tagalong.transport.os.getuid", lambda: real_uid + 1)
    client = LocalClient(server.path)
    try:
        with pytest.raises(TransportError, match="connection closed"):
            client.call("initialize", {"client": "electron"})
    finally:
        client.close()
        server.stop()


def test_serve_without_a_listener_returns() -> None:
    server = LocalServer(Controller(), path=Path("/unused.sock"))

    assert server._listener is None
    assert server._serve() is None


def test_serve_exits_immediately_when_already_stopped(tmp_path: Path) -> None:
    server = LocalServer(Controller(), path=tmp_path / "tagalong.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server._listener = listener
    server._stop.set()
    try:
        assert server._serve() is None
        assert listener.fileno() >= 0
    finally:
        listener.close()


def test_accept_timeout_and_transient_oserror_are_ignored(
    tmp_path: Path, monkeypatch
) -> None:
    server = LocalServer(Controller(), path=tmp_path / "tagalong.sock")
    server.start()
    calls = {"n": 0}
    original = server._accept

    def flaky_accept(listener):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return original(listener)

    monkeypatch.setattr(server, "_accept", flaky_accept)
    time.sleep(0.4)
    server.stop()
    assert calls["n"] >= 1


def test_peer_credentials_match_this_process() -> None:
    left, right = socket.socketpair(socket.AF_UNIX)
    try:
        peer = read_peer(left)
        assert peer.uid == os.getuid()
        assert peer.pid == os.getpid() or peer.pid > 0
    finally:
        left.close()
        right.close()


def test_mcp_actor_is_an_agent_and_electron_is_human() -> None:
    peer = Peer(pid=1, uid=7, gid=7)

    assert actor_for_client("mcp", peer).kind.value == "agent"
    assert actor_for_client("electron", peer).kind.value == "human"
    assert actor_for_client("electron", peer).id == "electron-7"
    assert actor_for_client("local", peer).id == "local-7"


def test_event_pump_applies_tts_fragments() -> None:
    class State:
        tts_enabled = True

    class Event:
        def __init__(self, name, payload):
            self.name = name
            self.payload = payload

    pending = [
        Event("action.applied", {"request_id": "req-1"}),
        Event("state.changed", {"tts_enabled": False}),
    ]
    state = State()
    pump = EventPump(
        lambda: pending, lambda changed: apply_state_fragment(state, changed)
    )
    pump.pump()

    assert state.tts_enabled is False
    apply_state_fragment(state, {"other": True})
    assert state.tts_enabled is False


def test_event_pump_thread_applies_then_stops() -> None:
    class State:
        tts_enabled = True

    class Event:
        def __init__(self) -> None:
            self.name = "state.changed"
            self.payload = {"tts_enabled": False}

    state = State()
    pending: list[Event] = [Event()]

    def drain():
        events = tuple(pending)
        pending.clear()
        return events

    pump = EventPump(drain, lambda changed: apply_state_fragment(state, changed))
    pump.start()
    deadline = time.time() + 1.0
    while state.tts_enabled is not False and time.time() < deadline:
        time.sleep(0.05)
    pump.stop()
    assert state.tts_enabled is False


def test_a_second_writer_updates_tui_state_through_the_event_pump() -> None:
    class Display:
        def __init__(self) -> None:
            self.tts_enabled = True

    controller = Controller()
    bind_first_slice(controller, conversation=Conversation(), tts=Speech())
    display = Display()
    _snapshot, subscription = controller.subscribe()
    pump = EventPump(
        subscription.drain, lambda changed: apply_state_fragment(display, changed)
    )

    controller.dispatch(
        "tts.set_enabled", {"enabled": False}, actor=local_user("agent-ui")
    )
    pump.pump()

    assert display.tts_enabled is False


def test_snapshot_payload_includes_protocol_cursor() -> None:
    controller = Controller()
    payload = snapshot_payload(controller.snapshot())

    state = cast(dict[str, object], payload["state"])
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["sequence"] == 0
    assert state["tts_enabled"] is True
    assert payload["instance"]


def test_client_skips_event_notifications_and_reports_a_closed_socket(
    tmp_path: Path,
) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client._buffer = encode_frame(
            {"jsonrpc": "2.0", "method": "event", "params": {"name": "state.changed"}}
        ) + encode_frame({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        assert client._read_result() == {"ok": True}
        server.stop()
        with pytest.raises(TransportError, match="connection closed"):
            client._readline()
    finally:
        client.close()
        server.stop()


def test_client_call_wraps_a_closed_file_descriptor(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client._sock.close()
        with pytest.raises(TransportError, match="connection closed"):
            client.call("snapshot")
    finally:
        client.close()
        server.stop()


def test_client_readline_wraps_a_closed_file_descriptor(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client._sock.close()
        with pytest.raises(TransportError, match="connection closed"):
            client._readline()
    finally:
        client.close()
        server.stop()


def test_an_idle_connection_notices_stop_after_recv_timeout(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        server._stop.set()
        deadline = time.time() + 1.0
        while server._connections and time.time() < deadline:
            time.sleep(0.05)
        assert server._connections == []
    finally:
        client.close()
        server.stop()


def test_a_client_disconnect_ends_the_handler(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client._sock.shutdown(socket.SHUT_RDWR)
        deadline = time.time() + 1.0
        while server._connections and time.time() < deadline:
            time.sleep(0.05)
        assert server._connections == []
    finally:
        client.close()
        server.stop()
