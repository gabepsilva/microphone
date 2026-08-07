"""Local JSON-RPC transport: framing, peer checks, and dispatch."""

from __future__ import annotations

import os
import socket
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tagalong import transport
from tagalong.application import bind_first_slice
from tagalong.control import (
    Accepted,
    Controller,
    Failed,
    Outcome,
    Selection,
    Superseded,
    local_user,
)
from tagalong.control.actions import PROTOCOL_VERSION, Scope
from tagalong.control.outcomes import Applied, Rejected, Rejection
from tagalong.transport import (
    MAX_FRAME,
    EventPump,
    LocalClient,
    LocalServer,
    Peer,
    TransportError,
    actor_for_client,
    decode_frame,
    encode_frame,
    json_ready,
    outcome_payload,
    prepare_runtime_dir,
    read_peer,
    runtime_dir,
    snapshot_payload,
    socket_path,
)
from tagalong.tui import SessionState, apply_state_fragment


class Speech:
    def __init__(self) -> None:
        self.enabled = True

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = enabled
        return True


class Conversation:
    generation = 1

    def __init__(self) -> None:
        self.ingested: list[tuple] = []

    def ingest(self, speaker, text, respond, timestamp=None, images=()):
        del timestamp
        self.ingested.append((speaker, text, respond, images))

    def interrupt(self) -> None:
        return None

    def start_fresh_thread(self):
        return None

    def adopt_fresh_thread(self, started) -> None:
        del started


def wired(
    tmp_path: Path,
    *,
    conversation: Conversation | None = None,
    capacity: int | None = None,
) -> tuple[Controller, LocalServer, LocalClient]:
    talk = conversation if conversation is not None else Conversation()
    controller = Controller() if capacity is None else Controller(capacity=capacity)
    bind_first_slice(controller, conversation=talk, tts=Speech())
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


def test_frames_are_ndjson_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    import tagalong.transport as transport

    frame = encode_frame({"ok": True})

    assert frame.endswith(b"\n")
    assert decode_frame(frame[:-1]) == {"ok": True}
    with pytest.raises(TransportError, match="JSON"):
        decode_frame(b"not-json")
    with pytest.raises(TransportError, match="JSON"):
        decode_frame(b"\xff")
    with pytest.raises(TransportError, match="JSON object"):
        decode_frame(b"[]")
    # Keep the oversize probe cheap; production MAX_FRAME is 30 MiB for uploads.
    monkeypatch.setattr(transport, "MAX_FRAME", 64)
    monkeypatch.setattr(transport, "_FRAME_LIMIT_LABEL", "0 MiB")
    with pytest.raises(TransportError, match="0 MiB"):
        transport.decode_frame(b"x" * 65)
    with pytest.raises(TransportError, match="0 MiB"):
        transport.encode_frame({"blob": "x" * 64})
    assert MAX_FRAME == 30 * 1024 * 1024


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
        assert hello["actor_kind"] == "agent"

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
        hello = client.call("initialize", {"client": "mcp"})
        assert hello["actor_kind"] == "agent"
        assert set(hello["scopes"]) == {scope.value for scope in Scope}
        listing = client.call("commands.list")
        assert listing["commands"][0]["action_id"] == "session.new"
        caps = client.call("capabilities")
        assert any(entry["id"] == "session.new" for entry in caps["actions"])
    finally:
        client.close()
        server.stop()


def test_codex_catalog_offers_models_and_their_efforts(
    tmp_path: Path, monkeypatch
) -> None:
    from tagalong.catalog import CodexModelOption

    monkeypatch.setattr(
        transport,
        "probe_codex_models",
        lambda: [CodexModelOption("gpt-5.6-luna", "Luna", ("low", "high"), "high")],
    )
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        catalog = client.call("codex.catalog")
        assert catalog["models"] == [
            {
                "slug": "gpt-5.6-luna",
                "label": "Luna",
                "efforts": ["low", "high"],
                "default_effort": "high",
            }
        ]
    finally:
        client.close()
        server.stop()


def test_codex_catalog_answers_empty_when_the_cli_offers_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    # An empty catalog is an answer: the client keeps what the session runs
    # rather than being told the query failed.
    monkeypatch.setattr(transport, "probe_codex_models", list)
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        assert client.call("codex.catalog") == {"models": []}
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
        assert polled["lost"] is False
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


def test_poll_reports_lost_after_overflow_and_resubscribe_clears_it(
    tmp_path: Path,
) -> None:
    """lost is terminal per subscription; recovery is subscribe again (#96)."""
    _, server, client = wired(tmp_path, capacity=2)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        # Each toggle publishes state.changed + action.applied (≥2 events).
        for enabled in (False, True, False):
            client.call(
                "dispatch",
                {"action": "tts.set_enabled", "payload": {"enabled": enabled}},
            )
        polled = client.call("poll")
        assert polled["lost"] is True
        assert len(polled["events"]) == 2

        client.call("subscribe")
        recovered = client.call("poll")
        assert recovered["lost"] is False
        assert recovered["events"] == []
    finally:
        client.close()
        server.stop()


def test_poll_with_timeout_returns_empty_when_nothing_arrives(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        started = time.monotonic()
        polled = client.call("poll", {"timeout_ms": 150})
        elapsed = time.monotonic() - started
        assert polled == {"events": [], "lost": False}
        assert elapsed >= 0.12
        assert elapsed < 1.0
    finally:
        client.close()
        server.stop()


def test_poll_rejects_invalid_timeout_ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tagalong.transport as transport

    monkeypatch.setattr(transport, "MAX_POLL_MS", 100)
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        with pytest.raises(TransportError, match="timeout_ms must be a number"):
            client.call("poll", {"timeout_ms": "soon"})
        with pytest.raises(TransportError, match="timeout_ms must be a number"):
            client.call("poll", {"timeout_ms": True})
        with pytest.raises(TransportError, match="timeout_ms must be >= 0"):
            client.call("poll", {"timeout_ms": -1})
        with pytest.raises(TransportError, match="timeout_ms must be finite"):
            client.call("poll", {"timeout_ms": float("inf")})
        with pytest.raises(TransportError, match="timeout_ms must be finite"):
            client.call("poll", {"timeout_ms": float("nan")})
        # Above the ceiling: clamp and return empty rather than kill the thread.
        started = time.monotonic()
        polled = client.call("poll", {"timeout_ms": 1_000_000})
        elapsed = time.monotonic() - started
        assert polled == {"events": [], "lost": False}
        assert elapsed >= 0.05
        assert elapsed < 1.0
    finally:
        client.close()
        server.stop()


def test_poll_with_timeout_returns_when_an_event_arrives(tmp_path: Path) -> None:
    """Long-poll parks one connection; a second connection must deliver work."""
    _, server, events = wired(tmp_path)
    commands = LocalClient(server.path)
    try:
        events.call("initialize", {"client": "electron"})
        events.call("subscribe")
        commands.call("initialize", {"client": "electron"})

        result: dict[str, object] = {}

        def park() -> None:
            result["polled"] = events.call("poll", {"timeout_ms": 2000})

        waiter = threading.Thread(target=park)
        waiter.start()
        time.sleep(0.05)
        commands.call(
            "dispatch", {"action": "tts.set_enabled", "payload": {"enabled": False}}
        )
        waiter.join(timeout=2.0)
        assert not waiter.is_alive()
        polled = cast(dict[str, object], result["polled"])
        names = [event["name"] for event in cast(list, polled["events"])]
        assert "state.changed" in names
        assert polled["lost"] is False
    finally:
        events.close()
        commands.close()
        server.stop()


def test_devices_list_returns_inputs_and_applications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tagalong.transport as transport

    monkeypatch.setattr(
        transport,
        "input_devices",
        lambda: [
            (0, {"name": "Built-in Mic", "max_input_channels": 2}),
            (3, {"name": "USB Mic", "max_input_channels": 1}),
        ],
    )
    monkeypatch.setattr(transport, "graph", lambda: [{"fake": True}])
    monkeypatch.setattr(
        transport,
        "offered_applications",
        lambda objects: [("Firefox (playing)", "Firefox")],
    )
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        listed = client.call("devices.list")
        assert listed == {
            "inputs": [
                {"index": 0, "name": "Built-in Mic", "channels": 2},
                {"index": 3, "name": "USB Mic", "channels": 1},
            ],
            "applications": [{"label": "Firefox (playing)", "name": "Firefox"}],
        }
    finally:
        client.close()
        server.stop()


def test_devices_list_maps_portaudio_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tagalong.transport as transport

    def boom():
        raise RuntimeError(
            "Audio device discovery requires the PortAudio system library."
        )

    monkeypatch.setattr(transport, "input_devices", boom)
    _, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        with pytest.raises(TransportError, match="PortAudio"):
            client.call("devices.list")
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


def test_a_closed_then_reopened_connection_must_initialize_again(
    tmp_path: Path,
) -> None:
    """The pattern a per-call Electron client used: initialize, hang up, retry."""
    _, server, first = wired(tmp_path)
    try:
        first.call("initialize", {"client": "electron"})
        first.close()
        second = LocalClient(server.path)
        try:
            with pytest.raises(TransportError, match="initialize first"):
                second.call("snapshot")
            hello = second.call("initialize", {"client": "electron"})
            assert hello["actor_kind"] == "agent"
            assert second.call("snapshot")["state"]["tts_enabled"] is True
        finally:
            second.close()
    finally:
        server.stop()


def test_initialize_cannot_be_replayed_to_change_actor(tmp_path: Path) -> None:
    _, server, client = wired(tmp_path)
    try:
        first = client.call("initialize", {"client": "electron"})
        with pytest.raises(TransportError, match="already initialized"):
            client.call("initialize", {"client": "mcp"})
        assert client.call("snapshot")["state"]["tts_enabled"] is True
        assert first["actor_id"].startswith("electron-")
    finally:
        client.close()
        server.stop()


def test_socket_message_send_ingests_as_agent_not_text(tmp_path: Path) -> None:
    talk = Conversation()
    _, server, client = wired(tmp_path, conversation=talk)
    try:
        client.call("initialize", {"client": "electron"})
        outcome = client.call(
            "dispatch", {"action": "message.send", "payload": {"text": "hi"}}
        )
        assert outcome["type"] == "applied"
        assert talk.ingested == [("Agent", "hi", True, ())]
    finally:
        client.close()
        server.stop()


def test_frame_cap_covers_a_max_size_image_upload() -> None:
    from tagalong.attachments import MAX_IMAGE_BYTES

    # Base64 expands by 4/3; leave 64 KiB for the JSON-RPC envelope.
    wire_bytes = -(-MAX_IMAGE_BYTES * 4 // 3) + 65_536
    assert wire_bytes <= MAX_FRAME
    assert MAX_FRAME > 1_048_576


def test_a_frame_without_a_newline_is_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tagalong.transport as transport

    # Avoid pushing 30 MiB across the socket just to prove the cap.
    monkeypatch.setattr(transport, "MAX_FRAME", 64)
    _, server, _client = wired(tmp_path)
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        raw.connect(str(server.path))
        raw.sendall(b"x" * 65)
        assert raw.recv(4096) == b""
    finally:
        raw.close()
        _client.close()
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


def test_socket_client_labels_do_not_mint_a_human_actor() -> None:
    peer = Peer(pid=1, uid=7, gid=7)

    assert actor_for_client("mcp", peer).kind.value == "agent"
    assert actor_for_client("electron", peer).kind.value == "agent"
    assert actor_for_client("electron", peer).id == "electron-7"
    assert actor_for_client("local", peer).id == "local-7"
    assert actor_for_client("  ", peer).id == "local-7"


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


def test_apply_state_fragment_copies_every_controller_owned_field() -> None:
    state = SessionState(
        microphone="Yeti",
        audio_stream=None,
        policy="both",
        tts_provider="piper",
        codex_model="gpt-5.6-luna",
        codex_effort="low",
        turn_silence=3.0,
        codex_efforts_by_model={"gpt-5.6-sol": ["low", "high"]},
        codex_default_effort_by_model={"gpt-5.6-sol": "low"},
    )

    apply_state_fragment(
        state,
        {
            "tts_enabled": False,
            "tts_provider": "edge",
            "response_policy": "voice",
            "microphone_muted": True,
            "audio_stream_muted": True,
            "microphone": Selection(desired="Webcam", effective="Webcam"),
            "audio_stream": {"desired": "Zoom", "effective": None},
            "codex_model": "gpt-5.6-sol",
            "codex_reasoning": "high",
            "turn_silence": 1.25,
        },
    )

    assert state.tts_enabled is False
    assert state.tts_provider == "edge"
    assert state.tts_voice == "en-US-AndrewNeural"
    assert state.policy == "voice"
    assert state.mic.muted is True
    assert state.audio.muted is True
    assert state.microphone == "Webcam"
    assert state.audio_stream == "Zoom"
    assert state.codex_model == "gpt-5.6-sol"
    assert state.codex_effort == "high"
    assert state.turn_silence == 1.25
    assert state.codex_efforts == ["low", "high"]


def test_applying_a_provider_change_does_not_unmute() -> None:
    """A remote provider event is not the sidebar's unmute composition."""
    state = SessionState(tts_provider="piper", tts_enabled=False)

    apply_state_fragment(state, {"tts_provider": "edge"})

    assert state.tts_enabled is False
    assert state.tts_provider == "edge"
    assert state.tts_voice == "en-US-AndrewNeural"


def test_json_ready_turns_selection_values_into_dicts() -> None:
    payload = json_ready(
        {
            "microphone": Selection(desired="Yeti", effective="Yeti"),
            "flag": True,
            "pair": ("a", Selection(desired="Webcam")),
        }
    )

    assert payload == {
        "microphone": {"desired": "Yeti", "effective": "Yeti"},
        "flag": True,
        "pair": ["a", {"desired": "Webcam", "effective": None}],
    }


def test_apply_state_fragment_selection_helpers_accept_wire_shapes() -> None:
    state = SessionState(codex_model="gpt-5.6-luna", codex_effort="medium")

    apply_state_fragment(state, {"microphone": None, "audio_stream": "Zoom"})
    assert state.microphone is None
    assert state.audio_stream == "Zoom"

    apply_state_fragment(state, {"codex_model": "missing-model"})
    assert state.codex_model == "missing-model"
    assert state.codex_effort == "medium"

    state.codex_efforts_by_model = {"gpt-5.6-sol": ["low", "high"]}
    apply_state_fragment(state, {"codex_model": "gpt-5.6-sol"})
    assert state.codex_effort == "low"

    broken = SimpleNamespace(
        codex_model="gpt-5.6-sol",
        codex_effort="medium",
        codex_efforts_by_model={"gpt-5.6-sol": ["low", "high"]},
        codex_default_effort_by_model="not-a-mapping",
    )
    apply_state_fragment(broken, {"codex_model": "gpt-5.6-sol"})
    assert broken.codex_effort == "low"

    class Display:
        tts_enabled = True
        tts_provider = "piper"
        microphone = None
        policy = "both"
        codex_model = "gpt-5.6-luna"
        codex_effort = "low"
        turn_silence = 3.0
        codex_efforts_by_model = 1

    display = Display()
    apply_state_fragment(
        display,
        {
            "tts_provider": "edge",
            "microphone_muted": True,
            "audio_stream_muted": True,
            "codex_model": "gpt-5.6-sol",
        },
    )
    assert display.tts_provider == "edge"
    assert display.codex_model == "gpt-5.6-sol"
    assert not hasattr(display, "tts_voice")


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
    assert state["partial_source"] == ""
    assert state["partial_text"] == ""
    assert payload["transcript"] == []
    assert payload["instance"]


def test_subscribe_snapshot_includes_accepted_transcript_rows(tmp_path: Path) -> None:
    from tagalong.presentation import Entry

    controller, server, client = wired(tmp_path)
    try:
        store = controller.transcript
        store.append(Entry(kind="note", text="kept", stamp="12:00"))
        store.append(
            Entry(kind="speech", source="Voice", text="pending", stamp="12:01"),
            provisional=True,
        )
        client.call("initialize", {"client": "electron"})
        subscribed = client.call("subscribe")
        rows = cast(list[object], subscribed["transcript"])
        assert len(rows) == 1
        row = cast(dict[str, object], rows[0])
        entry = cast(dict[str, object], row["entry"])
        assert row["id"] == 1
        assert entry["text"] == "kept"
    finally:
        client.close()
        server.stop()


def test_poll_delivers_transcript_events_in_order(tmp_path: Path) -> None:
    from tagalong.presentation import Entry

    controller, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        entry = Entry(kind="speech", source="Taga", text="", streaming=True)
        controller.transcript.append(entry)
        controller.transcript.append_text(entry, "Hi")
        controller.transcript.append_text(entry, "!")
        controller.transcript.flush_updates()
        controller.transcript.clear()

        polled = client.call("poll")
        names = [event["name"] for event in polled["events"]]
        assert names == [
            "transcript.entry_added",
            "transcript.entry_updated",
            "transcript.cleared",
        ]
        updated = next(
            event
            for event in polled["events"]
            if event["name"] == "transcript.entry_updated"
        )
        payload = cast(dict[str, object], updated["payload"])
        entry_payload = cast(dict[str, object], payload["entry"])
        assert entry_payload["text"] == "Hi!"
        assert polled["lost"] is False
    finally:
        client.close()
        server.stop()


def test_lost_resubscribe_restores_transcript_via_snapshot(tmp_path: Path) -> None:
    from tagalong.presentation import Entry

    controller, server, client = wired(tmp_path, capacity=2)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        for enabled in (False, True, False):
            client.call(
                "dispatch",
                {"action": "tts.set_enabled", "payload": {"enabled": enabled}},
            )
        assert client.call("poll")["lost"] is True

        controller.transcript.append(
            Entry(kind="note", text="after overflow", stamp="1")
        )
        recovered = client.call("subscribe")
        rows = cast(list[object], recovered["transcript"])
        assert len(rows) == 1
        entry = cast(dict[str, object], cast(dict[str, object], rows[0])["entry"])
        assert entry["text"] == "after overflow"
        assert client.call("poll")["lost"] is False
    finally:
        client.close()
        server.stop()


def test_partials_arrive_as_state_changed_on_the_wire(tmp_path: Path) -> None:
    controller, server, client = wired(tmp_path)
    try:
        client.call("initialize", {"client": "electron"})
        client.call("subscribe")
        controller.set_partial("Voice", "hello there")
        polled = client.call("poll")
        changed = [
            event for event in polled["events"] if event["name"] == "state.changed"
        ]
        assert changed
        payload = cast(dict[str, object], changed[0]["payload"])
        assert payload["partial_source"] == "Voice"
        assert payload["partial_text"] == "hello there"
        snap = client.call("snapshot")
        state = cast(dict[str, object], snap["state"])
        assert state["partial_source"] == "Voice"
        assert state["partial_text"] == "hello there"
    finally:
        client.close()
        server.stop()


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


def test_an_idle_connection_notices_stop_after_recv_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    _, server, client = wired(tmp_path)
    calls = {"n": 0}
    original = server._recv

    def flaky_recv(connection):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError
        return original(connection)

    monkeypatch.setattr(server, "_recv", flaky_recv)
    try:
        hello = client.call("initialize", {"client": "electron"})
        assert hello["protocol_version"] == PROTOCOL_VERSION
        assert calls["n"] >= 1
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
