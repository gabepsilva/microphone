"""The thread-marshalling façade and the keyboard actions.

The façade only forwards to the app once the UI is mounted, so these run the
real app under Textual's test pilot and exercise the production ``_call``
path rather than the ``_*_impl`` methods directly.
"""

from __future__ import annotations

import asyncio
import threading

from textual.widgets import Input

from voice_codex.tui import EntryRow


def drive(facade, body):
    """Run ``body`` against a mounted app with the façade marked ready."""

    async def exercise() -> None:
        async with facade.app.run_test() as pilot:
            facade._app_thread = threading.get_ident()
            facade._ready.set()
            await body(pilot)

    asyncio.run(exercise())


def entry_texts(facade):
    return [entry.text for entry in facade.app.entries]


def test_a_committed_turn_becomes_a_transcript_entry(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.partial(tui.USER_VOICE, "half a sen")
        facade.commit(tui.USER_VOICE, "half a sentence")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["half a sentence"]
    assert facade.state.partial_text == ""


def test_a_note_becomes_a_dim_entry(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.note("ignored likely echo")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["ignored likely echo"]
    assert facade.app.entries[0].kind == "note"


def test_a_codex_turn_streams_into_one_row_and_closes(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_message_open(tui.THEM)
        await pilot.pause()
        facade.codex_delta("Hello ")
        facade.codex_delta("there.")
        await pilot.pause()
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["Hello there."]
    assert facade.app.entries[0].source == tui.CODEX
    assert facade.app.entries[0].streaming is False
    assert facade.app._streaming is None
    assert facade.state.codex_state == "idle"


def test_a_delta_with_no_open_row_opens_one(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_delta("Unannounced.")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["Unannounced."]


def test_an_interrupted_turn_is_marked_cut_off(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_message_open(tui.THEM)
        await pilot.pause()
        facade.codex_delta("Half a th")
        await pilot.pause()
        facade.codex_end(interrupted=True)
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries[0].interrupted is True
    assert "cut off" in tui.render_entry_body(facade.app.entries[0]).plain


def test_a_command_row_collects_output_and_its_exit_code(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.command_started("ls -la")
        await pilot.pause()
        facade.command_output("total 0")
        facade.command_output("drwx .")
        await pilot.pause()
        facade.command_completed(2)
        await pilot.pause()

    drive(facade, body)

    entry = facade.app.entries[0]

    assert (entry.kind, entry.text) == ("command", "ls -la")
    assert entry.output == ["total 0", "drwx ."]
    assert entry.exit_code == 2
    assert facade.app._command_row is None


def test_a_command_without_an_exit_code_records_minus_one(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.command_started("sleep 1")
        await pilot.pause()
        facade.command_completed(None)
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries[0].exit_code == -1


def test_command_output_with_no_command_row_is_dropped(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.command_output("orphaned line")
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries == []


def test_tool_activity_is_noted(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.tool_called("files", "read")
        facade.tool_completed("completed")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["tool files.read", "tool status: completed"]


def test_a_codex_error_is_shown_in_the_transcript(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.error("Codex error: stream broke")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["Codex error: stream broke"]


def test_panel_updates_reach_the_running_sidebar(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.set_audio("mic", device="Blue Yeti")
        facade.set_audio("mic", level=0.5)
        facade.set_audio("nonexistent", device="ignored")
        facade.set_output("Speakers")
        facade.set_codex(model="gpt-5.6-nova", thread="thread-9")
        facade.set_session(tokens=42)
        facade.set_status("listening", live=True)
        facade.set_policy("quiet")
        facade.set_tts_queue(["one", "two"])
        await pilot.pause()

    drive(facade, body)

    assert facade.state.mic.device == "Blue Yeti"
    assert facade.state.mic.level == 0.5
    assert facade.state.out_device == "Speakers"
    assert facade.state.codex_model == "gpt-5.6-nova"
    assert facade.state.codex_thread == "thread-9"
    assert facade.state.tokens == 42
    assert facade.state.policy == "quiet"
    assert facade.state.tts_queue == ["one", "two"]


def test_an_unknown_policy_is_refused(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.set_policy("nonsense")

    assert facade.state.policy != "nonsense"


def test_a_discovered_catalog_adopts_the_models_efforts(tui) -> None:
    facade = tui.VoiceCodexTUI(tui.SessionState(codex_model="m1", codex_effort="low"))

    facade.set_codex_catalog(
        [("Model One", "m1")], {"m1": ["medium", "high"]}, {"m1": "high"}
    )

    assert facade.state.codex_efforts == ["medium", "high"]
    assert facade.state.codex_effort == "high"


def test_a_catalog_keeps_an_effort_the_model_still_supports(tui) -> None:
    facade = tui.VoiceCodexTUI(tui.SessionState(codex_model="m1", codex_effort="low"))

    facade.set_codex_catalog(
        [("Model One", "m1")], {"m1": ["low", "high"]}, {"m1": "high"}
    )

    assert facade.state.codex_effort == "low"


def test_a_catalog_that_omits_the_active_model_changes_nothing(tui) -> None:
    facade = tui.VoiceCodexTUI(tui.SessionState(codex_model="m9", codex_effort="low"))

    facade.set_codex_catalog([("Model One", "m1")], {"m1": ["high"]}, {"m1": "high"})

    assert facade.state.codex_effort == "low"


def test_stopping_before_the_app_is_ready_is_harmless(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.stop()
    facade.note("dropped")

    assert facade.app.entries == []


def test_the_facade_reports_when_the_app_is_ready(tui) -> None:
    facade = tui.VoiceCodexTUI()

    assert facade.wait_ready(timeout=0) is False

    async def body(pilot):
        assert facade.wait_ready(timeout=0) is True
        await pilot.pause()

    drive(facade, body)


def test_finishing_a_turn_clears_only_that_speakers_partial(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.update(tui.THEM, "mid sentence")
    facade.finish_turn(tui.USER_VOICE)

    assert facade.state.partial_text == "mid sentence"

    facade.close_speaker(tui.THEM)

    assert facade.state.partial_text == ""


def test_cycling_the_policy_notes_it_and_calls_back(tui) -> None:
    chosen: list[str] = []
    facade = tui.VoiceCodexTUI(on_policy=chosen.append)

    async def body(pilot):
        await pilot.press("ctrl+p")
        await pilot.pause()

    drive(facade, body)

    assert chosen == [facade.state.policy]
    assert any("response policy" in text for text in entry_texts(facade))


def test_muting_toggles_the_channel_and_calls_back(tui) -> None:
    muted: list[bool] = []
    facade = tui.VoiceCodexTUI(on_mute=muted.append)

    async def body(pilot):
        await pilot.press("ctrl+k")
        await pilot.pause()
        await pilot.press("ctrl+k")
        await pilot.pause()

    drive(facade, body)

    assert muted == [True, False]
    assert facade.state.mic.muted is False


def test_toggling_tts_off_clears_the_queue(tui) -> None:
    facade = tui.VoiceCodexTUI(
        tui.SessionState(tts_enabled=True, tts_queue=["pending"]),
        on_tts=lambda enabled: True,
    )

    async def body(pilot):
        await pilot.press("ctrl+t")
        await pilot.pause()

    drive(facade, body)

    assert facade.state.tts_enabled is False
    assert facade.state.tts_queue == []


def test_tts_stays_off_when_the_session_has_no_speech(tui) -> None:
    facade = tui.VoiceCodexTUI(
        tui.SessionState(tts_enabled=False), on_tts=lambda enabled: False
    )

    async def body(pilot):
        await pilot.press("ctrl+t")
        await pilot.pause()

    drive(facade, body)

    assert facade.state.tts_enabled is False
    assert any("tts unavailable" in text for text in entry_texts(facade))


def test_interrupting_marks_the_streaming_row_and_calls_back(tui) -> None:
    interrupts: list[bool] = []
    facade = tui.VoiceCodexTUI(on_interrupt=lambda: interrupts.append(True))

    async def body(pilot):
        facade.codex_message_open(tui.THEM)
        await pilot.pause()
        facade.codex_delta("Talking when")
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()

    drive(facade, body)

    assert interrupts == [True]
    assert facade.app.entries[0].interrupted is True
    assert facade.state.codex_state == "idle"


def test_saving_hands_over_the_entries_and_notes_the_count(tui) -> None:
    saved: list[list] = []
    facade = tui.VoiceCodexTUI(on_save=saved.append)

    async def body(pilot):
        facade.commit(tui.USER_VOICE, "one")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

    drive(facade, body)

    assert len(saved) == 1
    assert [entry.text for entry in saved[0]] == ["one"]
    assert any("saved transcript" in text for text in entry_texts(facade))


def test_a_slash_command_is_routed_to_the_command_hook(tui) -> None:
    commands: list[str] = []
    facade = tui.VoiceCodexTUI(on_command=commands.append)

    async def body(pilot):
        facade.app.query_one("#input", Input).value = "/save"
        await pilot.press("enter")
        await pilot.pause()

    drive(facade, body)

    assert commands == ["/save"]


def test_blank_input_submits_nothing(tui) -> None:
    typed: list[str] = []
    facade = tui.VoiceCodexTUI(on_user_text=typed.append)

    async def body(pilot):
        facade.app.query_one("#input", Input).value = "   "
        await pilot.press("enter")
        await pilot.pause()

    drive(facade, body)

    assert typed == []
    assert facade.app.entries == []


def test_clear_removes_typed_text_before_it_would_quit(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.app.query_one("#input", Input).value = "half typed"
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert facade.app.query_one("#input", Input).value == ""

    drive(facade, body)


def test_a_command_entry_renders_its_output_and_exit_code(tui) -> None:
    entry = tui.Entry(kind="command", text="ls -la", output=["total 0"], exit_code=127)

    body = tui.render_entry_body(entry).plain

    assert body.startswith("$ ls -la")
    assert "total 0" in body
    assert "[command exit: 127]" in body


def test_a_streaming_entry_shows_a_cursor(tui) -> None:
    entry = tui.Entry(kind="speech", source=tui.CODEX, text="typing", streaming=True)

    assert "▌" in tui.render_entry_body(entry).plain


def test_a_note_entry_renders_as_plain_text(tui) -> None:
    assert tui.render_entry_body(tui.Entry(kind="note", text="a note")).plain == (
        "a note"
    )


def test_the_transcript_shows_committed_entries(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.commit(tui.USER_VOICE, "visible text")
        await pilot.pause()
        rows = facade.app.query(EntryRow).results()

        assert [row.entry.text for row in rows] == ["visible text"]

    drive(facade, body)
