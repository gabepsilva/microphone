"""Headless start mode (#102 D9) — session without Textual."""

from __future__ import annotations

import signal
from types import SimpleNamespace
from typing import cast

from tagalong.control import Controller
from tagalong.domain import TAGA, VOICE
from tagalong.headless import HeadlessSession
from tagalong.presentation import Entry
from tagalong.startup import build_parser
from tagalong.tui import SessionState, apply_state_fragment


def _texts(host: HeadlessSession) -> list[str]:
    return [entry.text for entry in host.transcript.entries()]


def _kinds(host: HeadlessSession) -> list[str]:
    return [entry.kind for entry in host.transcript.entries()]


def test_parser_accepts_headless_flag() -> None:
    args = build_parser().parse_args(["--headless"])
    assert args.headless is True
    assert build_parser().parse_args([]).headless is False


def test_headless_session_writes_accepted_speech_to_the_store() -> None:
    host = HeadlessSession(SessionState())
    controller = Controller(transcript=host.transcript)
    host.bind_partial_publisher(controller.set_partial)
    recorded: list[Entry] = []
    host.hooks.on_entry = recorded.append

    host.update("Voice", "partial")
    assert controller.state.partial_text == "partial"
    host.commit("Voice", "hello")
    host.finish_turn("Voice", accepted=True)

    assert _texts(host) == ["hello"]
    assert recorded
    assert recorded[0].text == "hello"


def test_headless_session_rejects_provisional_speech() -> None:
    host = HeadlessSession()
    controller = Controller(transcript=host.transcript)
    host.bind_session_state_publisher(controller.set_session_state)
    host.commit("Voice", "echo")
    host.finish_turn("Voice", accepted=False)
    assert _texts(host) == []
    assert host.state.echoes_cut == 1
    assert controller.state.echoes_cut == 1


def test_headless_speech_activity_reaches_attached_clients() -> None:
    class Speaking:
        value = True

        def is_speaking(self) -> bool:
            return self.value

    speech = Speaking()
    host = HeadlessSession(speech=speech)
    controller = Controller(transcript=host.transcript)
    host.bind_session_state_publisher(controller.set_session_state)

    host._tick_speaking()
    assert controller.state.codex_speaking is True
    host._tick_speaking()
    speech.value = False
    host._tick_speaking()

    assert controller.state.codex_speaking is False


def test_headless_close_speaker_accepts_like_finish_turn() -> None:
    host = HeadlessSession()
    host.commit("Voice", "kept")
    host.close_speaker("Voice")
    assert _texts(host) == ["kept"]


def test_headless_run_polls_speech_until_stopped(monkeypatch) -> None:
    class StopAfterPoll:
        def is_speaking(self) -> bool:
            host.stop()
            return True

    host = HeadlessSession(speech=StopAfterPoll())
    waits = iter([False, True])
    monkeypatch.setattr(host._stop, "wait", lambda _timeout: next(waits))
    host.run()

    assert host.state.codex_speaking is True


def test_headless_session_streams_taga_onto_the_store() -> None:
    host = HeadlessSession()
    host.begin_codex()
    host.codex_message_open("Voice")
    host.codex_delta("hi ")
    host.codex_delta("there")
    host.codex_message_close()
    host.end_codex()
    rows = list(host.transcript.entries())
    assert len(rows) == 1
    assert rows[0].text == "hi there"
    assert rows[0].streaming is False


def test_headless_codex_delta_opens_streaming_row_when_missing() -> None:
    host = HeadlessSession()
    host.codex_delta("orphan")
    host.end_codex()
    rows = list(host.transcript.entries())
    assert rows[0].source == TAGA
    assert rows[0].reply_to == VOICE
    assert rows[0].text == "orphan"


def test_headless_empty_codex_delta_is_a_no_op() -> None:
    host = HeadlessSession()
    host.codex_message_open("Voice")
    host.codex_delta("")
    assert next(iter(host.transcript.entries())).text == ""


def test_headless_reasoning_and_command_stream() -> None:
    recorded: list[Entry] = []
    host = HeadlessSession()
    host.hooks.on_entry = recorded.append

    host.reasoning_started()
    host.reasoning_delta("think ")
    host.reasoning_delta("hard")
    host.reasoning_completed()

    before_command = len(recorded)
    host.command_started("ls")
    host.command_output("a\n")
    # Must stay unrecorded until command_completed (parity with VoiceCodexApp).
    assert len(recorded) == before_command
    host.command_completed(0)

    assert _kinds(host) == ["reasoning", "command"]
    assert any(entry.kind == "reasoning" and entry.recorded for entry in recorded)
    commands = [entry for entry in recorded if entry.kind == "command"]
    assert len(commands) == 1
    assert commands[0].output == ["a\n"]
    assert commands[0].exit_code == 0
    assert commands[0].recorded is True


def test_headless_reasoning_delta_starts_row_when_missing() -> None:
    host = HeadlessSession()
    host.reasoning_delta("late")
    host.reasoning_completed()
    assert _texts(host) == ["late"]


def test_headless_command_output_without_start_is_ignored() -> None:
    host = HeadlessSession()
    host.command_output("orphan")
    host.command_completed(None)
    assert _texts(host) == []


def test_headless_tools_tokens_errors_and_panels() -> None:
    host = HeadlessSession(SessionState(codex_model="gpt", codex_effort="low"))
    controller = Controller(transcript=host.transcript)
    host.bind_session_state_publisher(controller.set_session_state)
    host.tool_called("server", "tool")
    host.tool_completed("ok")
    host.token_usage(42)
    host.error("boom")
    host.set_audio("mic", active=True)
    host.set_audio("audio", active=False)
    host.set_audio("unknown", active=True)
    host.set_codex(
        model="gpt",
        effort="high",
        state="thinking",
        thread="thread-9",
        speaking=True,
        unknown=True,
    )
    host.set_codex_catalog(
        [("gpt", "GPT")],
        {"gpt": ["low", "high"]},
        {"gpt": "high"},
    )
    host.set_session(status="ready", missing_key=True)
    host.set_status("live", live=False)

    assert host.state.tokens == 42
    assert controller.state.tokens == 42
    assert controller.state.codex_thread == "thread-9"
    assert controller.state.codex_state == "thinking"
    assert controller.state.codex_speaking is True
    assert host.state.mic.active is True
    assert host.state.audio.active is False
    assert host.state.codex_effort == "high"
    assert host.state.codex_efforts == ["low", "high"]
    assert host.state.status == "live"
    assert host.state.live is False
    texts = _texts(host)
    assert "tool server.tool" in texts
    assert "tool status: ok" in texts
    assert "boom" in texts


def test_headless_reset_and_finish_recording_flush_streams() -> None:
    host = HeadlessSession()
    host.commit("Voice", "gone")
    host.codex_message_open("Voice")
    host.codex_delta("partial answer")
    host.reasoning_started()
    host.reasoning_delta("partial think")
    host.reset_transcript()
    assert _texts(host) == []
    assert host.transcript_entries() == []
    assert host._streaming is None
    assert host._reasoning is None


def test_headless_finish_recording_records_unrecorded_entries() -> None:
    calls: list[Entry] = []
    host = HeadlessSession()
    host.hooks.on_entry = lambda entry: calls.append(entry) or False
    host.note("skip-record")
    assert calls
    assert calls[0].recorded is False
    host.hooks.on_entry = lambda entry: True
    host.finish_recording()
    assert any(
        entry.text == "skip-record" and entry.recorded for entry in host._entries
    )


def test_headless_hooks_kwargs_bind_on_construct() -> None:
    seen: list[Entry] = []
    host = HeadlessSession(on_entry=seen.append)
    host.note("bound")
    assert seen[0].text == "bound"


def test_headless_run_unblocks_on_stop() -> None:
    host = HeadlessSession()
    host.stop()
    host.run()  # must return immediately
    assert host._stop.is_set()


def test_headless_signal_stop_sets_the_stop_event() -> None:
    host = HeadlessSession()
    host._signal_stop(signal.SIGINT, None)
    assert host._stop.is_set()


def test_headless_delta_while_flush_pending_skips_second_schedule() -> None:
    host = HeadlessSession()
    host.codex_message_open("Voice")
    assert host._answer_deltas.append("held") is True
    host.codex_delta("dropped-from-schedule")
    # First append still owns the batch; flush drains both chunks.
    host._flush_answer_deltas()
    assert "held" in next(iter(host.transcript.entries())).text

    host.reasoning_started()
    assert host._reasoning_deltas.append("held-r") is True
    host.reasoning_delta("also-held")
    host._flush_reasoning_deltas()
    assert "held-r" in list(host.transcript.entries())[-1].text


def test_headless_end_codex_and_reasoning_completed_without_open_rows() -> None:
    host = HeadlessSession()
    host.begin_codex()
    host.end_codex()
    host.reasoning_completed()
    assert host.state.codex_state == "idle"
    assert _texts(host) == []


def test_headless_catalog_defaults_effort_when_current_invalid() -> None:
    host = HeadlessSession(SessionState(codex_model="gpt", codex_effort="missing"))
    host.set_codex_catalog(
        [("gpt", "GPT")],
        {"gpt": ["low", "medium"]},
        {},
    )
    assert host.state.codex_effort == "low"


def test_headless_catalog_without_efforts_leaves_state_alone() -> None:
    host = HeadlessSession(SessionState(codex_model="gpt", codex_effort="low"))
    host.set_codex_catalog([("other", "Other")], {}, {})

    assert host.state.codex_effort == "low"


def test_headless_finish_turn_only_clears_matching_partial() -> None:
    host = HeadlessSession()
    host.update("Voice", "v")
    host.finish_turn("Audio", accepted=True)
    assert host.state.partial_source == "Voice"
    host.finish_turn("Voice", accepted=True)
    assert host.state.partial_source == ""


def test_headless_device_lists_and_notes_do_not_need_textual() -> None:
    host = HeadlessSession()
    host.set_microphones([("Yeti", "Yeti")])
    host.set_audio_streams([("Zoom", "zoom")])
    host.note("hello")
    assert host.state.microphones == [("Yeti", "Yeti")]
    assert host.state.audio_streams == [("Zoom", "zoom")]
    assert _texts(host) == ["hello"]


def test_headless_show_message_draws_accepted_speech() -> None:
    # Socket peers have no prompt; show_message is what puts their line on
    # the transcript. Unlike commit(), the text is final when it arrives.
    host = HeadlessSession()
    recorded: list[Entry] = []
    host.hooks.on_entry = recorded.append

    host.show_message("Agent", "hello from a client")

    assert [(e.kind, e.source, e.text) for e in host.transcript.entries()] == [
        ("speech", "Agent", "hello from a client")
    ]
    assert [e.text for e in recorded] == ["hello from a client"]
    assert recorded[0].recorded is True


def test_event_pump_apply_updates_headless_session_state() -> None:
    host = HeadlessSession(SessionState(tts_enabled=True))
    apply_state_fragment(host.state, {"tts_enabled": False})
    assert host.state.tts_enabled is False


def test_build_session_host_picks_headless_or_tui(monkeypatch) -> None:
    from tagalong import cli
    from tagalong.headless import HeadlessSession

    monkeypatch.setattr(
        cli,
        "build_session_state",
        lambda *_a, **_k: SessionState(),
    )
    headless = cli.build_session_host(
        SimpleNamespace(headless=True), SimpleNamespace(), [], None, None
    )
    assert isinstance(headless, HeadlessSession)

    monkeypatch.setattr(
        "tagalong.tui.VoiceCodexTUI",
        lambda *a, **k: SimpleNamespace(kind="tui"),
    )
    tui_host = cli.build_session_host(
        SimpleNamespace(headless=False), SimpleNamespace(), [], None, None
    )
    assert tui_host.kind == "tui"


def _recorded_view(entries: list[Entry]) -> list[tuple[object, ...]]:
    """Comparable recorder view — stamps excluded (clock-dependent)."""
    return [
        (
            entry.kind,
            entry.source,
            entry.text,
            list(entry.output),
            entry.exit_code,
            entry.streaming,
            entry.interrupted,
        )
        for entry in entries
    ]


def _capture_at_record_time(sink: list[tuple[object, ...]]):
    """on_entry hook that freezes fields at the moment of recording."""

    def capture(entry: Entry) -> None:
        sink.append(_recorded_view([entry])[0])

    return capture


def _command_and_stream_script(host: HeadlessSession | object) -> None:
    """Shared presentation script for headless/TUI recorder parity."""
    cast_host = cast(HeadlessSession, host)
    cast_host.begin_codex()
    cast_host.codex_message_open("Voice")
    cast_host.codex_delta("hi ")
    cast_host.codex_delta("there")
    cast_host.end_codex()
    cast_host.command_started("ls -la")
    cast_host.command_output("total 0\n")
    cast_host.command_completed(0)


def test_headless_and_tui_record_the_same_command_and_stream_script() -> None:
    """Parity pin: same script → same *record-time* payloads (#102).

    Must snapshot at ``on_entry`` time. Appending live ``Entry`` references
    and reading them later is blind to early-vs-late recording drift — the
    objects converge after ``command_completed`` either way.
    """
    from tagalong import tui as tui_mod
    from tests.test_tui_facade import drive

    headless_recorded: list[tuple[object, ...]] = []
    tui_recorded: list[tuple[object, ...]] = []
    headless = HeadlessSession(on_entry=_capture_at_record_time(headless_recorded))
    facade = tui_mod.VoiceCodexTUI(on_entry=_capture_at_record_time(tui_recorded))

    _command_and_stream_script(headless)

    async def body(pilot):
        _command_and_stream_script(facade)
        await pilot.pause()

    drive(facade, body)

    assert headless_recorded == tui_recorded
    command = headless_recorded[-1]
    assert command[0] == "command"
    assert command[3] == ["total 0\n"]
    assert command[4] == 0
