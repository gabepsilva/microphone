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
        facade.update(tui.USER_VOICE, "half a sen")
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


def test_an_interrupted_turn_reads_as_cut_off(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_message_open(tui.THEM)
        await pilot.pause()
        facade.codex_delta("Half a th")
        await pilot.pause()
        await pilot.press("ctrl+x")
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
        facade.set_audio("mic", active=True)
        facade.set_audio("nonexistent", device="ignored")
        facade.set_output("Speakers")
        facade.set_codex(model="gpt-5.6-nova", thread="thread-9")
        facade.set_session(tokens=42)
        facade.set_status("listening", live=True)
        facade.set_tts_queue(["one", "two"])
        await pilot.pause()

    drive(facade, body)

    assert facade.state.mic.device == "Blue Yeti"
    assert facade.state.mic.active is True
    assert facade.state.out_device == "Speakers"
    assert facade.state.codex_model == "gpt-5.6-nova"
    assert facade.state.codex_thread == "thread-9"
    assert facade.state.tokens == 42
    assert facade.state.tts_queue == ["one", "two"]


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


# --------------------------------------------------------------------------
# The facade has one naming convention
# --------------------------------------------------------------------------

# Methods that are legitimately not part of the presentation protocols.
# Adding a name here is a deliberate decision, which is the point: the fifteen
# extras this list replaced grew one alias at a time, each reasonable on its
# own, until the facade had two names for most of what it did.
HOST_ONLY_METHODS = frozenset(
    {
        # Lifecycle, owned by whoever runs the interface.
        "run",
        "wait_ready",
        "stop",
        # Sidebar panels the host fills in. These are not part of a Codex
        # turn, so no presentation protocol describes them.
        "set_audio",
        "set_output",
        "set_session",
        "set_status",
        "set_tts_queue",
    }
)


def protocol_methods():
    from voice_codex import presentation

    names: set[str] = set()
    for protocol in (
        presentation.TranscriptSink,
        presentation.SessionStatusSink,
        presentation.CodexStreamSink,
    ):
        names |= {name for name in vars(protocol) if not name.startswith("_")}
    return names


def test_the_facade_offers_no_second_name_for_a_presentation_call(tui) -> None:
    """Every public method is a protocol method or a listed host control.

    A second spelling of an existing call is how the runtime and the interface
    drift into two vocabularies, and an alias is invisible to coverage because
    its own tests keep it green.
    """
    public = {name for name in vars(tui.VoiceCodexTUI) if not name.startswith("_")}

    assert public - protocol_methods() - HOST_ONLY_METHODS == set()


def test_the_facade_implements_every_presentation_call(tui) -> None:
    public = {name for name in vars(tui.VoiceCodexTUI) if not name.startswith("_")}

    assert protocol_methods() - public == set()


# --------------------------------------------------------------------------
# Keeping the interface fast as the transcript grows
#
# Textual lays out every widget in the application on each layout pass, so an
# unbounded transcript makes every repaint anywhere in the interface slower —
# the sidebar included. These hold the two properties that prevent it.
# --------------------------------------------------------------------------


def mounted_rows(facade):
    return list(facade.app.query_one("#transcript").query(EntryRow))


def test_the_transcript_stops_mounting_rows_once_it_is_full(tui) -> None:
    facade = tui.VoiceCodexTUI()
    facade.app.MAX_MOUNTED_ROWS = 5
    counted: list[int] = []

    async def body(pilot):
        for index in range(12):
            facade.note(f"line {index}")
        await pilot.pause()
        counted.append(len(mounted_rows(facade)))

    drive(facade, body)

    assert counted == [5]


def test_every_entry_is_kept_however_few_rows_stay_mounted(tui) -> None:
    """Scrollback is what the cap spends. The record is not."""
    facade = tui.VoiceCodexTUI()
    facade.app.MAX_MOUNTED_ROWS = 3
    saved: list[list] = []
    facade.hooks.on_save = saved.append

    async def body(pilot):
        for index in range(10):
            facade.note(f"line {index}")
        await pilot.pause()
        facade.app.action_save()
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade)[:10] == [f"line {index}" for index in range(10)]
    assert [entry.text for entry in saved[0]] == [
        f"line {index}" for index in range(10)
    ]


def test_the_streaming_row_is_never_unmounted_under_it(tui) -> None:
    """A row still being written to must survive the cap however old it is."""
    facade = tui.VoiceCodexTUI()
    facade.app.MAX_MOUNTED_ROWS = 2

    async def body(pilot):
        facade.codex_message_open(tui.USER_VOICE)
        streaming = facade.app._streaming
        for index in range(8):
            facade.note(f"line {index}")
        await pilot.pause()
        assert streaming in mounted_rows(facade)
        facade.codex_delta("answer")
        await pilot.pause()
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    codex_entries = [
        entry.text for entry in facade.app.entries if entry.source == tui.CODEX
    ]
    assert codex_entries == ["answer"]


def test_the_running_command_row_is_never_unmounted_under_it(tui) -> None:
    facade = tui.VoiceCodexTUI()
    facade.app.MAX_MOUNTED_ROWS = 2

    async def body(pilot):
        facade.command_started("ls -la")
        command_row = facade.app._command_row
        for index in range(8):
            facade.note(f"line {index}")
        await pilot.pause()
        assert command_row in mounted_rows(facade)
        facade.command_output("total 0")
        facade.command_completed(0)
        await pilot.pause()

    drive(facade, body)

    command = next(entry for entry in facade.app.entries if entry.kind == "command")
    assert command.output == ["total 0"]
    assert command.exit_code == 0


# --------------------------------------------------------------------------
# Coalescing the stream
# --------------------------------------------------------------------------


def test_streamed_deltas_do_not_repaint_once_per_token(tui) -> None:
    """Codex streams faster than a terminal can usefully redraw."""
    facade = tui.VoiceCodexTUI()
    repaints: list[int] = []

    async def body(pilot):
        facade.codex_message_open(tui.USER_VOICE)
        row = facade.app._streaming
        original = row.sync
        row.sync = lambda: (repaints.append(1), original())[1]
        for word in ("one ", "two ", "three ", "four "):
            facade.codex_delta(word)
        await pilot.pause()

    drive(facade, body)

    assert len(repaints) < 4


def test_the_finished_answer_is_drawn_in_full_however_it_was_coalesced(tui) -> None:
    """Coalescing may skip repaints, but never the last one."""
    facade = tui.VoiceCodexTUI()
    drawn: list[str] = []

    async def body(pilot):
        facade.codex_message_open(tui.USER_VOICE)
        for word in ("one ", "two ", "three"):
            facade.codex_delta(word)
        facade.end_codex()
        await pilot.pause()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.CODEX)
        drawn.append(next(iter(row.query(".entry-body"))).content.plain)

    drive(facade, body)

    assert entry_texts(facade) == ["one two three"]
    assert "one two three" in drawn[0]
    assert facade.app._dirty == []


def test_an_interrupted_answer_shows_the_text_that_arrived_before_the_cut(tui) -> None:
    """An interrupt must not strand text the flush timer had not drawn yet."""
    facade = tui.VoiceCodexTUI()
    drawn: list[str] = []

    async def body(pilot):
        facade.codex_message_open(tui.USER_VOICE)
        facade.codex_delta("half a th")
        facade.app.action_interrupt()
        await pilot.pause()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.CODEX)
        drawn.append(next(iter(row.query(".entry-body"))).content.plain)

    drive(facade, body)

    assert "half a th" in drawn[0]
    assert "cut off" in drawn[0]
    assert facade.app.entries[0].interrupted is True


def test_an_idle_session_flushes_nothing(tui) -> None:
    facade = tui.VoiceCodexTUI()
    facade.app.flush_stream()

    assert facade.app._dirty == []


# --------------------------------------------------------------------------
# Repainting only what changed
# --------------------------------------------------------------------------


def test_token_usage_repaints_the_counters_and_not_the_pickers(tui) -> None:
    """It arrives all through a streamed answer, so it must stay cheap."""
    facade = tui.VoiceCodexTUI()
    calls: list[str] = []

    async def body(pilot):
        sidebar = facade.app.query_one("#sidebar", tui.Sidebar)
        sidebar.sync = lambda: calls.append("whole sidebar")
        sidebar.sync_session = lambda: calls.append("session panel")
        facade.token_usage(1234)
        await pilot.pause()

    drive(facade, body)

    assert calls == ["session panel"]
    assert facade.state.tokens == 1234


def test_a_sound_report_never_waits_on_the_application_thread(tui) -> None:
    """These come from an audio callback; waiting there stalls capture."""
    facade = tui.VoiceCodexTUI()
    waited: list[str] = []

    async def body(pilot):
        facade._app_thread = -1  # pretend we are a capture thread
        facade.app.call_from_thread = lambda *a, **k: waited.append("blocked")
        facade.set_audio("mic", active=True)
        await pilot.pause()

    drive(facade, body)

    assert waited == []
    assert facade.state.mic.active is True


def test_naming_a_capture_device_still_refreshes_the_whole_sidebar(tui) -> None:
    """A device name is set once, from the main thread, and moves the layout."""
    facade = tui.VoiceCodexTUI()
    calls: list[str] = []

    async def body(pilot):
        sidebar = facade.app.query_one("#sidebar", tui.Sidebar)
        sidebar.sync = lambda: calls.append("whole sidebar")
        facade.set_audio("mic", device="Blue Yeti")
        await pilot.pause()

    drive(facade, body)

    assert calls == ["whole sidebar"]
