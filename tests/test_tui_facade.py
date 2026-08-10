"""The thread-marshalling façade and the keyboard actions.

The façade only forwards to the app once the UI is mounted, so these run the
real app under Textual's test pilot and exercise the production ``_call``
path rather than the ``_*_impl`` methods directly.
"""

from __future__ import annotations

import asyncio
import threading

from rich.text import Text
from textual import events
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.widgets import Markdown, Static

from tagalong.tui import EntryRow, OrderedDeltaBuffer, PromptInput


def test_an_ordered_delta_buffer_schedules_one_flush_per_batch() -> None:
    buffer = OrderedDeltaBuffer()

    assert buffer.append("one ") is True
    assert buffer.append("two") is False
    assert buffer.take() == "one two"
    assert buffer.append("three") is True
    assert buffer.take() == "three"


def test_abandoning_a_schedule_keeps_chunks_and_allows_another_flush() -> None:
    buffer = OrderedDeltaBuffer()

    assert buffer.append("early ") is True
    buffer.abandon_schedule()
    assert buffer.buffered() is True
    assert buffer.append("late") is True
    assert buffer.take() == "early late"


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


async def wait_markdown(row: EntryRow, pilot, text: str) -> Markdown:
    """Wait until the row hosts Textual Markdown whose source matches ``text``."""
    for _ in range(20):
        for widget in row.query(Markdown):
            if widget.source == text:
                return widget
        await pilot.pause()
    raise AssertionError(f"markdown body with source {text!r} did not mount")


def test_a_committed_turn_becomes_a_transcript_entry(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.update(tui.VOICE, "half a sen")
        facade.commit(tui.VOICE, "half a sentence")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["half a sentence"]
    assert facade.state.partial_text == ""


def test_an_accepted_turn_is_recorded_after_echo_filtering(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.commit(tui.VOICE, "a real question")
        await pilot.pause()
        assert recorded == []
        facade.finish_turn(tui.VOICE, accepted=True)
        await pilot.pause()

    drive(facade, body)

    assert [entry.text for entry in recorded] == ["a real question"]
    assert entry_texts(facade) == ["a real question"]


def test_transcript_save_view_excludes_provisional_rows(tui) -> None:
    """F5 / #102: export matches the recorded view, not every painted row.

    Before the store owned save, ``transcript_entries`` returned ``app.entries``
    including commits still awaiting ``finish_turn``. A save in that window
    exported speech the session file would never contain on echo reject.
    """
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.commit(tui.VOICE, "maybe echo")
        await pilot.pause()
        assert entry_texts(facade) == ["maybe echo"]
        assert facade.transcript_entries() == []
        facade.finish_turn(tui.VOICE, accepted=True)
        await pilot.pause()

    drive(facade, body)

    assert [entry.text for entry in facade.transcript_entries()] == ["maybe echo"]


def test_controller_adopts_tui_store_and_publishes_commit(tui) -> None:
    from tagalong.control import Controller

    facade = tui.VoiceCodexTUI()
    controller = Controller(transcript=facade.transcript)
    _snapshot, subscription = controller.subscribe()

    async def body(pilot):
        facade.commit(tui.VOICE, "hi")
        await pilot.pause()
        assert [event.name for event in subscription.drain()] == [
            "transcript.entry_added"
        ]
        facade.finish_turn(tui.VOICE, accepted=True)
        await pilot.pause()

    drive(facade, body)

    names = [event.name for event in subscription.drain()]
    assert names == ["transcript.entry_updated"]
    assert controller.transcript.rows()[0].provisional is False


def test_partials_mirror_onto_controller_app_state(tui) -> None:
    from tagalong.control import Controller
    from tagalong.tui import SessionState, apply_state_fragment

    facade = tui.VoiceCodexTUI()
    controller = Controller(transcript=facade.transcript)
    facade.bind_partial_publisher(controller.set_partial)
    _snapshot, subscription = controller.subscribe()

    async def body(pilot):
        facade.update(tui.VOICE, "hello partial")
        await pilot.pause()

    drive(facade, body)

    assert controller.state.partial_source == tui.VOICE
    assert controller.state.partial_text == "hello partial"
    changed = [event for event in subscription.drain() if event.name == "state.changed"]
    assert changed
    assert changed[-1].payload["partial_text"] == "hello partial"

    # EventPump must not echo partials onto SessionState (TUI is the writer).
    echoed = SessionState()
    apply_state_fragment(echoed, dict(changed[-1].payload))
    assert echoed.partial_source == ""
    assert echoed.partial_text == ""


def test_partial_publish_stays_leaf_under_controller_lock(tui) -> None:
    """Recognition and end-turn/commit must not deadlock on lock order (#102).

    ``voice.end_turn`` holds ``Controller._lock`` into ``commit`` →
    ``_show_partial``. Recognition takes ``_partial_lock`` then publishes.
    Publishing outside ``_partial_lock`` keeps that lock a leaf.
    """
    import threading
    import time

    from tagalong.control import Controller

    facade = tui.VoiceCodexTUI()
    controller = Controller(transcript=facade.transcript)
    facade.bind_partial_publisher(controller.set_partial)
    ready = threading.Event()
    done = threading.Event()

    def dispatcher() -> None:
        with controller._lock:
            ready.set()
            time.sleep(0.05)
            facade._show_partial("", "")
        done.set()

    def recognizer() -> None:
        ready.wait(timeout=1.0)
        facade._show_partial(tui.VOICE, "racing")

    threads = [
        threading.Thread(target=dispatcher, name="dispatcher"),
        threading.Thread(target=recognizer, name="recognizer"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert done.is_set()
    assert all(not thread.is_alive() for thread in threads)
    assert controller.state.partial_text in {"", "racing"}


def test_a_rejected_echo_is_removed_instead_of_recorded(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.update(tui.AUDIO, "Taga speaking")
        facade.commit(tui.AUDIO, "Taga speaking")
        await pilot.pause()
        assert entry_texts(facade) == ["Taga speaking"]
        facade.finish_turn(tui.AUDIO, accepted=False)
        await pilot.pause()

    drive(facade, body)

    assert recorded == []
    assert facade.app.entries == []
    assert facade.state.partial_text == ""
    assert facade.state.echoes_cut == 1


def test_rejecting_echo_removes_only_that_channels_interleaved_rows(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.commit(tui.VOICE, "possible echo one")
        facade.commit(tui.AUDIO, "other channel")
        facade.commit(tui.VOICE, "possible echo two")
        await pilot.pause()
        facade.finish_turn(tui.VOICE, accepted=False)
        facade.finish_turn(tui.AUDIO, accepted=True)
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["other channel"]


def test_on_entry_fires_once_for_a_finished_note_and_not_for_an_open_taga_row(
    tui,
) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.note("policy → voice")
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()

    drive(facade, body)

    assert [entry.kind for entry in recorded] == ["note"]
    assert recorded[0].text == "policy → voice"
    assert facade.app._streaming is not None
    assert facade.app._streaming.entry.recorded is False


def test_a_streamed_taga_answer_is_recorded_when_the_turn_ends(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("Hello ")
        facade.codex_delta("there.")
        await pilot.pause()
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].source == tui.TAGA
    assert recorded[0].text == "Hello there."
    assert recorded[0].recorded is True
    assert recorded[0].streaming is False


def test_an_interrupted_answer_is_recorded_once(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("half an answ")
        await pilot.pause()
        facade.app.action_interrupt()
        await pilot.pause()
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].interrupted is True
    assert recorded[0].text == "half an answ"


def test_a_command_is_recorded_at_completion_with_its_output(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.command_started("ls")
        facade.command_output("file.txt")
        await pilot.pause()
        assert recorded == []
        facade.command_completed(0)
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].kind == "command"
    assert recorded[0].text == "ls"
    assert recorded[0].output == ["file.txt"]
    assert recorded[0].exit_code == 0


def test_record_open_entries_sweeps_what_a_cut_off_turn_left_behind(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.reasoning_started()
        facade.reasoning_delta("still thinking")
        await pilot.pause()
        assert recorded == []
        facade.finish_recording()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].kind == "reasoning"
    assert recorded[0].text == "still thinking"


def test_recording_sweeps_buffered_stream_text_before_the_entry(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        facade._answer_deltas.append("buffered answer")
        facade.finish_recording()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].text == "buffered answer"


def test_reset_transcript_records_buffered_stream_text_before_clearing(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        facade._answer_deltas.append("buffered before reset")
        facade.reset_transcript()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].text == "buffered before reset"
    assert facade.app.entries == []


def test_a_failed_recording_is_retried_on_a_later_sweep(tui) -> None:
    attempts: list = []

    def record(entry):
        attempts.append(entry)
        return len(attempts) > 1

    facade = tui.VoiceCodexTUI(on_entry=record)

    async def body(pilot):
        facade.note("retry me")
        await pilot.pause()
        assert attempts[0].recorded is False
        facade.finish_recording()
        await pilot.pause()

    drive(facade, body)

    assert len(attempts) == 2
    assert attempts[0] is attempts[1]
    assert attempts[0].recorded is True


def test_clearing_the_transcript_records_open_entries_first(tui) -> None:
    recorded: list = []
    facade = tui.VoiceCodexTUI(on_entry=recorded.append)

    async def body(pilot):
        facade.command_started("pwd")
        await pilot.pause()
        facade.reset_transcript()
        await pilot.pause()

    drive(facade, body)

    assert len(recorded) == 1
    assert recorded[0].kind == "command"
    assert recorded[0].text == "pwd"
    assert facade.app.entries == []


def test_a_note_becomes_a_dim_entry(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.note("ignored likely echo")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["ignored likely echo"]
    assert facade.app.entries[0].kind == "note"


def test_a_remote_message_becomes_a_speech_entry(tui) -> None:
    # Socket peers have no prompt of their own; this path draws for them.
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.show_message(tui.AGENT, "hello from a client")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["hello from a client"]
    assert facade.app.entries[0].kind == "speech"
    assert facade.app.entries[0].source == tui.AGENT


def test_a_remote_message_is_scheduled_not_waited_on(tui) -> None:
    """``show_message`` hands the draw over instead of waiting for it.

    It runs inside a control handler, under the controller's writer lock —
    and ``cli.attach_conversation_hooks`` gives the controller this façade's
    transcript store, so that lock is the store's lock. Waiting on the
    application thread would wait on the thread that has to take that lock to
    append the row, while the caller still holds it: both stop for good.

    The test above cannot see this. It calls from the application thread,
    where ``_call`` takes its same-thread fast path and the marshalling never
    happens. Here ``_app_thread`` is set to another ident, so the façade takes
    the cross-thread path a socket peer would.
    """
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade._app_thread = -1
        facade.show_message(tui.AGENT, "hello from a client")
        # Handed over, not drawn: the caller is free to release the lock the
        # application thread is about to want.
        assert entry_texts(facade) == []
        await pilot.pause()
        assert entry_texts(facade) == ["hello from a client"]

    drive(facade, body)

    # Stamped when the session was told, not whenever the row happened to mount.
    assert facade.app.entries[0].stamp != ""


def test_a_codex_turn_streams_into_one_row_and_closes(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_message_open(tui.AUDIO)
        await pilot.pause()
        facade.codex_delta("Hello ")
        facade.codex_delta("there.")
        await pilot.pause()
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["Hello there."]
    assert facade.app.entries[0].source == tui.TAGA
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
        facade.codex_message_open(tui.AUDIO)
        await pilot.pause()
        facade.codex_delta("Half a th")
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries[0].interrupted is True
    assert facade.app.entries[0].text == "Half a th"


def test_thinking_is_a_codex_row_that_closes_with_what_it_cost(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.reasoning_started()
        await pilot.pause()
        facade.reasoning_delta("Weighing the riddle.")
        await pilot.pause()

        assert facade.state.codex_state == "thinking"
        assert facade.app.entries[0].streaming is True

        facade.reasoning_completed()
        await pilot.pause()

    drive(facade, body)

    entry = facade.app.entries[0]

    assert (entry.kind, entry.source) == ("reasoning", tui.TAGA)
    assert entry.stamp != ""
    assert entry.text == "Weighing the riddle."
    assert entry.streaming is False
    assert entry.seconds is not None
    assert facade.app._reasoning_row is None


def test_a_turn_cut_off_mid_thought_still_closes_its_thinking(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.reasoning_started()
        await pilot.pause()
        facade.reasoning_delta("Half a thought")
        # No reasoning_completed: the item never finished.
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries[0].streaming is False
    assert facade.app.entries[0].seconds is not None
    assert facade.app._reasoning_row is None


def test_closing_thinking_that_never_started_is_harmless(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries == []
    assert facade._thinking_started is None


def test_summary_text_with_no_open_section_opens_one(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.reasoning_delta("Unannounced thought.")
        await pilot.pause()

    drive(facade, body)

    assert entry_texts(facade) == ["Unannounced thought."]
    assert facade.app.entries[0].kind == "reasoning"


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
    from tagalong.control import Controller

    controller = Controller(transcript=facade.transcript)
    facade.bind_session_state_publisher(controller.set_session_state)
    _snapshot, subscription = controller.subscribe()

    async def body(pilot):
        facade.set_audio("mic", active=True)
        facade.set_audio("nonexistent", active=True)
        facade.set_codex(model="gpt-5.6-nova", thread="thread-9", state="thinking")
        facade.set_session(tokens=42)
        facade.set_status("listening", live=True)
        await pilot.pause()

    drive(facade, body)

    assert facade.state.mic.active is True
    assert facade.state.codex_model == "gpt-5.6-nova"
    assert facade.state.codex_thread == "thread-9"
    assert controller.state.codex_thread == "thread-9"
    assert controller.state.codex_state == "thinking"
    assert any(
        dict(event.payload)
        == {
            "codex_thread": "thread-9",
            "codex_state": "thinking",
        }
        for event in subscription.drain()
        if event.name == "state.changed"
    )
    assert facade.state.tokens == 42


def test_speech_activity_reaches_attached_clients(tui) -> None:
    class Speaking:
        def is_speaking(self) -> bool:
            return True

    facade = tui.VoiceCodexTUI(speech=Speaking())
    from tagalong.control import Controller

    controller = Controller(transcript=facade.transcript)
    facade.bind_session_state_publisher(controller.set_session_state)

    async def body(pilot):
        facade.app._tick_speaking()
        await pilot.pause()

    drive(facade, body)

    assert facade.state.codex_speaking is True
    assert controller.state.codex_speaking is True


def test_codex_state_transitions_reach_attached_clients(tui) -> None:
    facade = tui.VoiceCodexTUI()
    from tagalong.control import Controller

    controller = Controller(transcript=facade.transcript)
    facade.bind_session_state_publisher(controller.set_session_state)

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        assert controller.state.codex_state == "replying to Voice"
        facade.reasoning_started()
        await pilot.pause()
        assert controller.state.codex_state == "thinking"
        facade.command_started("ls")
        await pilot.pause()
        assert controller.state.codex_state == "running command"
        facade.end_codex()
        await pilot.pause()
        assert controller.state.codex_state == "idle"
        facade.token_usage(1234)
        await pilot.pause()
        assert controller.state.tokens == 1234
        facade.commit(tui.VOICE, "echo")
        await pilot.pause()
        facade.finish_turn(tui.VOICE, accepted=False)
        await pilot.pause()
        assert controller.state.echoes_cut == 1

    drive(facade, body)

    assert controller.state.codex_state == "idle"


def test_a_discovered_catalog_adopts_the_models_efforts(tui) -> None:
    facade = tui.VoiceCodexTUI(tui.SessionState(codex_model="m1", codex_effort="low"))

    facade.set_codex_catalog(
        [("Model One", "m1")], {"m1": ["medium", "high"]}, {"m1": "high"}
    )

    assert facade.state.codex_efforts == ["medium", "high"]
    assert facade.state.codex_effort == "high"


def test_the_applications_on_offer_are_installed_from_the_refresher(tui) -> None:
    facade = tui.VoiceCodexTUI(tui.SessionState())

    facade.set_audio_streams([("Brave (playing)", "Brave")])

    assert facade.state.audio_streams == [("Brave (playing)", "Brave")]


def test_the_application_being_listened_to_survives_leaving_the_graph(tui) -> None:
    """An application that stops playing must not read as a changed choice."""
    facade = tui.VoiceCodexTUI(tui.SessionState(audio_stream="Brave"))

    facade.set_audio_streams([])

    assert facade.state.audio_stream == "Brave"


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

    facade.update(tui.AUDIO, "mid sentence")
    facade.finish_turn(tui.VOICE)

    assert facade.state.partial_text == "mid sentence"

    facade.close_speaker(tui.AUDIO)

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


def test_toggling_tts_off_updates_the_session(tui) -> None:
    facade = tui.VoiceCodexTUI(
        tui.SessionState(tts_enabled=True),
        on_tts=lambda enabled: True,
    )

    async def body(pilot):
        await pilot.press("ctrl+t")
        await pilot.pause()

    drive(facade, body)

    assert facade.state.tts_enabled is False


def test_tts_stays_off_when_the_toggle_is_refused(tui) -> None:
    facade = tui.VoiceCodexTUI(
        tui.SessionState(tts_enabled=False), on_tts=lambda enabled: False
    )

    async def body(pilot):
        await pilot.press("ctrl+t")
        await pilot.pause()

    drive(facade, body)

    assert facade.state.tts_enabled is False
    assert any("tts could not be changed" in text for text in entry_texts(facade))


def test_interrupting_marks_the_streaming_row_and_calls_back(tui) -> None:
    interrupts: list[bool] = []
    facade = tui.VoiceCodexTUI(on_interrupt=lambda: interrupts.append(True))

    async def body(pilot):
        facade.codex_message_open(tui.AUDIO)
        await pilot.pause()
        facade.codex_delta("Talking when")
        await pilot.pause()
        await pilot.press("ctrl+x")
        await pilot.pause()

    drive(facade, body)

    assert interrupts == [True]
    assert facade.app.entries[0].interrupted is True
    assert facade.state.codex_state == "idle"


def test_ending_a_voice_turn_calls_the_flush_hook(tui) -> None:
    ended: list[bool] = []
    facade = tui.VoiceCodexTUI(on_end_turn=lambda: ended.append(True))

    async def body(pilot):
        await pilot.press("ctrl+d")
        await pilot.pause()

    drive(facade, body)

    assert ended == [True]


def test_saving_hands_over_the_entries_and_notes_the_count(tui) -> None:
    saved: list[list] = []
    facade = tui.VoiceCodexTUI(on_save=saved.append)

    async def body(pilot):
        facade.commit(tui.VOICE, "one")
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
        facade.app.query_one("#input", PromptInput).value = "/save"
        await pilot.press("enter")
        await pilot.pause()

    drive(facade, body)

    assert commands == ["/save"]


def _catalog(*names_and_descriptions: tuple[str, str]):
    from tagalong.commands import CommandSpec

    specs = tuple(
        CommandSpec(name=name, description=description)
        for name, description in names_and_descriptions
    )
    return lambda: specs


def test_typing_a_slash_opens_the_command_palette(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh session"), ("help", "List commands"))
    )
    seen: list[tuple[bool, list[str]]] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.value = "/"
        facade.app._sync_command_palette(prompt.value)
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        seen.append((palette.is_open, [spec.name for spec in palette.items]))

    drive(facade, body)

    assert seen == [(True, ["new", "help"])]


def test_the_palette_drops_down_under_the_prompt(tui) -> None:
    """Open menu sits under the prompt; nothing else lives below it."""
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"))
    )
    layout: list[tuple[str, ...]] = []

    async def body(pilot):
        left = facade.app.query_one("#left")
        layout.append(tuple(child.id for child in left.children if child.id))
        facade.app._sync_command_palette("/")
        await pilot.pause()

    drive(facade, body)

    assert layout[0].index("promptbar") < layout[0].index("command-palette")
    assert layout[0][-1] == "command-palette"


def test_the_palette_filters_as_the_prompt_changes(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"), ("note", "Write"))
    )
    names: list[list[str]] = []

    async def body(pilot):
        facade.app._sync_command_palette("/ne")
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        names.append([spec.name for spec in palette.items])

    drive(facade, body)

    assert names == [["new", "note"]]


def test_enter_on_a_partial_slash_runs_the_highlighted_command(tui) -> None:
    ran: list[str] = []
    closed: list[bool] = []
    facade = tui.VoiceCodexTUI(
        on_command=ran.append,
        list_commands=_catalog(("new", "Fresh session"), ("help", "List commands")),
    )

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        prompt.value = "/ne"
        facade.app._sync_command_palette(prompt.value)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        closed.append(
            not facade.app.query_one("#command-palette", tui.CommandPalette).is_open
        )

    drive(facade, body)

    assert ran == ["/new"]
    assert closed == [True]


def test_arrow_keys_move_the_palette_highlight(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"))
    )
    selected: list[str] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        prompt.value = "/"
        facade.app._sync_command_palette("/")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        assert palette.selected() is not None
        selected.append(palette.selected().name)

    drive(facade, body)

    assert selected == ["help"]


def test_tab_completes_the_highlighted_command_name(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"))
    )
    completed: list[str] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        prompt.value = "/h"
        facade.app._sync_command_palette("/h")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        completed.append(facade.app.query_one("#input", PromptInput).value)

    drive(facade, body)

    assert completed == ["/help"]


def test_escape_closes_the_palette_without_submitting(tui) -> None:
    ran: list[str] = []
    facade = tui.VoiceCodexTUI(
        on_command=ran.append,
        list_commands=_catalog(
            ("new", "Fresh"),
        ),
    )
    snapshot: list[tuple[bool, str]] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        prompt.value = "/"
        facade.app._sync_command_palette("/")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        snapshot.append((palette.is_open, prompt.value))

    drive(facade, body)

    assert snapshot == [(False, "/")]
    assert ran == []


def test_the_palette_stays_closed_without_a_catalog(tui) -> None:
    facade = tui.VoiceCodexTUI()
    open_states: list[bool] = []

    async def body(pilot):
        facade.app._sync_command_palette("/")
        await pilot.pause()
        open_states.append(
            facade.app.query_one("#command-palette", tui.CommandPalette).is_open
        )

    drive(facade, body)

    assert open_states == [False]


def test_no_matches_shows_an_empty_state_and_submits_the_typed_command(tui) -> None:
    ran: list[str] = []
    empty: list[str] = []
    facade = tui.VoiceCodexTUI(
        on_command=ran.append,
        list_commands=_catalog(
            ("new", "Fresh"),
        ),
    )

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        prompt.value = "/zzzz"
        facade.app._sync_command_palette(prompt.value)
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        empty.append(str(palette.render()))
        await pilot.press("enter")
        await pilot.pause()

    drive(facade, body)

    assert "no matching commands" in empty[0]
    assert ran == ["/zzzz"]


def test_palette_keeps_the_highlighted_row_when_filtering(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"), ("halt", "Stop"))
    )
    selected: list[str] = []

    async def body(pilot):
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        facade.app._sync_command_palette("/")
        await pilot.pause()
        palette.move(1)  # help
        facade.app._sync_command_palette("/h")
        await pilot.pause()
        assert palette.selected() is not None
        selected.append(palette.selected().name)
        selected.append(str(palette.index))

    drive(facade, body)

    assert selected[0] == "help"
    assert selected[1] == "0"  # help is first among /h matches (help, halt)


def test_palette_window_scrolls_when_the_catalog_is_long(tui) -> None:
    specs = tuple((f"cmd{index:02d}", f"Command {index}") for index in range(12))
    facade = tui.VoiceCodexTUI(list_commands=_catalog(*specs))
    window: list[tuple[int, int]] = []

    async def body(pilot):
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        facade.app._sync_command_palette("/")
        await pilot.pause()
        for _ in range(10):
            palette.move(1)
        window.append(palette._window())
        # Empty list is a no-op for move; exercise the early return.
        palette.show(())
        palette.move(1)
        assert palette.selected() is None

    drive(facade, body)

    start, end = window[0]
    assert end - start == tui.CommandPalette.MAX_VISIBLE
    assert start > 0


def test_render_command_palette_is_pure(tui) -> None:
    from tagalong.commands import CommandSpec

    empty = tui.render_command_palette((), 0)
    filled = tui.render_command_palette(
        (CommandSpec("new", "Fresh"), CommandSpec("help", "List")),
        1,
    )

    assert "no matching commands" in str(empty)
    assert "▸ /help" in str(filled)
    assert "  /new" in str(filled)


def test_palette_actions_skip_when_the_menu_is_closed(tui) -> None:
    from textual.actions import SkipAction

    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(
            ("new", "Fresh"),
        )
    )
    skipped: list[str] = []

    async def body(_pilot):
        try:
            facade.app.action_palette_move(1)
        except SkipAction:
            skipped.append("move")
        try:
            facade.app.action_palette_complete()
        except SkipAction:
            skipped.append("complete")

    drive(facade, body)

    assert skipped == ["move", "complete"]


def test_prompt_changes_sync_the_palette_through_textual(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(("new", "Fresh"), ("help", "List"))
    )
    opened: list[bool] = []
    filtered: list[list[str]] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        await pilot.press("/")
        await pilot.pause()
        opened.append(
            facade.app.query_one("#command-palette", tui.CommandPalette).is_open
        )
        await pilot.press("n")
        await pilot.pause()
        filtered.append(
            [
                spec.name
                for spec in facade.app.query_one(
                    "#command-palette", tui.CommandPalette
                ).items
            ]
        )

    drive(facade, body)

    assert opened == [True]
    assert filtered == [["new"]]


def test_palette_paints_commands_that_have_no_description(tui) -> None:
    from tagalong.commands import CommandSpec

    facade = tui.VoiceCodexTUI(list_commands=lambda: (CommandSpec("quiet", ""),))
    painted: list[str] = []

    async def body(pilot):
        facade.app._sync_command_palette("/")
        await pilot.pause()
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        painted.append(str(palette.render()))

    drive(facade, body)

    assert "/quiet" in painted[0]
    assert "no matching" not in painted[0]


def test_palette_prefer_unknown_falls_back_to_the_first_row(tui) -> None:
    from tagalong.commands import CommandSpec

    facade = tui.VoiceCodexTUI(
        list_commands=lambda: (
            CommandSpec("new", "Fresh"),
            CommandSpec("help", "List"),
        )
    )
    selected: list[str] = []

    async def body(pilot):
        palette = facade.app.query_one("#command-palette", tui.CommandPalette)
        palette.show(
            (
                CommandSpec("new", "Fresh"),
                CommandSpec("help", "List"),
            ),
            prefer="missing",
        )
        await pilot.pause()
        assert palette.selected() is not None
        selected.append(palette.selected().name)

    drive(facade, body)

    assert selected == ["new"]


def test_tab_on_empty_matches_is_skipped(tui) -> None:
    from textual.actions import SkipAction

    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(
            ("new", "Fresh"),
        )
    )
    skipped: list[bool] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.focus()
        facade.app._sync_command_palette("/zzzz")
        await pilot.pause()
        try:
            facade.app.action_palette_complete()
        except SkipAction:
            skipped.append(True)

    drive(facade, body)

    assert skipped == [True]


def test_non_prompt_text_area_changes_do_not_touch_the_palette(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(
            ("new", "Fresh"),
        )
    )
    open_states: list[bool] = []

    async def body(pilot):
        class _Area:
            id = "other"
            text = "/"

        class _Event:
            text_area = _Area()

        facade.app.on_text_area_changed(_Event())  # type: ignore[arg-type]
        await pilot.pause()
        open_states.append(
            facade.app.query_one("#command-palette", tui.CommandPalette).is_open
        )

    drive(facade, body)

    assert open_states == [False]


def test_clearing_the_prompt_closes_the_palette(tui) -> None:
    facade = tui.VoiceCodexTUI(
        list_commands=_catalog(
            ("new", "Fresh"),
        )
    )
    open_states: list[bool] = []

    async def body(pilot):
        prompt = facade.app.query_one("#input", PromptInput)
        prompt.value = "/"
        facade.app._sync_command_palette("/")
        await pilot.pause()
        facade.app.action_clear_input_or_quit()
        await pilot.pause()
        open_states.append(
            facade.app.query_one("#command-palette", tui.CommandPalette).is_open
        )

    drive(facade, body)

    assert open_states == [False]


def test_blank_input_submits_nothing(tui) -> None:
    typed: list[str] = []
    facade = tui.VoiceCodexTUI(on_user_text=lambda message: typed.append(message.text))

    async def body(pilot):
        facade.app.query_one("#input", PromptInput).value = "   "
        await pilot.press("enter")
        await pilot.pause()

    drive(facade, body)

    assert typed == []
    assert facade.app.entries == []


def test_clear_removes_typed_text_before_it_would_quit(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.app.query_one("#input", PromptInput).value = "half typed"
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert facade.app.query_one("#input", PromptInput).value == ""

    drive(facade, body)


def test_a_command_entry_renders_its_output_and_exit_code(tui) -> None:
    entry = tui.Entry(kind="command", text="ls -la", output=["total 0"], exit_code=127)

    body = tui.render_entry_body(entry).plain

    assert body.startswith("$ ls -la")
    assert "total 0" in body
    assert "[command exit: 127]" in body


def test_a_streaming_entry_shows_a_cursor(tui) -> None:
    entry = tui.Entry(kind="speech", source=tui.TAGA, text="typing", streaming=True)

    assert "▌" in tui.render_entry_body(entry).plain
    assert tui.uses_markdown_body(entry) is False


def test_only_finished_taga_answers_use_markdown_bodies(tui) -> None:
    """The host widget switch is the product rule; assert it directly."""
    finished = tui.Entry(kind="speech", source=tui.TAGA, text="**done**")
    streaming = tui.Entry(
        kind="speech", source=tui.TAGA, text="**live**", streaming=True
    )
    empty = tui.Entry(kind="speech", source=tui.TAGA, text="")
    voice = tui.Entry(kind="speech", source=tui.VOICE, text="**noise**")

    assert tui.uses_markdown_body(finished) is True
    assert tui.uses_markdown_body(streaming) is False
    assert tui.uses_markdown_body(empty) is False
    assert tui.uses_markdown_body(voice) is False
    # Static path keeps markers literal; markdown parsing is the widget's job.
    assert tui.render_entry_body(streaming).plain == "**live** ▌"
    assert tui.render_entry_body(voice).plain == "**noise**"


def test_a_streaming_taga_answer_stays_plain_until_the_turn_closes(tui) -> None:
    """Live tokens stay on Static; the closed turn swaps in Markdown."""
    facade = tui.VoiceCodexTUI()
    kinds: list[str] = []

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("Hello **bold**")
        for _ in range(20):
            await pilot.pause()
            if facade.app.entries and facade.app.entries[0].text == "Hello **bold**":
                break
        else:
            raise AssertionError("streamed text did not land on the entry")
        facade.app.flush_stream()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.TAGA)
        streaming_body = next(iter(row.query(".entry-body")))
        kinds.append(type(streaming_body).__name__)
        assert isinstance(streaming_body, Static)
        assert streaming_body.content.plain == "Hello **bold** ▌"
        facade.end_codex()
        md = await wait_markdown(row, pilot, "Hello **bold**")
        kinds.append(type(md).__name__)

    drive(facade, body)

    assert kinds == ["Static", "Markdown"]


def test_a_finished_taga_row_hosts_clickable_markdown(tui) -> None:
    """Production path: mounted Markdown opens hrefs through the app."""
    facade = tui.VoiceCodexTUI()
    opened: list[str] = []
    source: list[str] = []

    async def body(pilot):
        text = "see [docs](https://example.com/x)"
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta(text)
        facade.end_codex()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.TAGA)
        md = await wait_markdown(row, pilot, text)
        source.append(md.source)
        pilot.app.open_url = lambda url, **_kw: opened.append(url)
        md.post_message(Markdown.LinkClicked(md, "https://example.com/x"))
        await pilot.pause()

    drive(facade, body)

    assert source == ["see [docs](https://example.com/x)"]
    assert opened == ["https://example.com/x"]


def test_a_finished_taga_markdown_row_updates_in_place(tui) -> None:
    """A second sync rewrites Markdown source rather than remounting the widget."""
    facade = tui.VoiceCodexTUI()
    result: list[object] = []

    async def body(pilot):
        first = "first [a](https://a.test)"
        second = "second [b](https://b.test)"
        row = facade.app.add_entry(
            tui.Entry(kind="speech", source=tui.TAGA, text=first)
        )
        md = await wait_markdown(row, pilot, first)
        first_id = id(md)
        row.entry.text = second
        row.sync()
        md = await wait_markdown(row, pilot, second)
        result.extend([md.source, id(md) == first_id])

    drive(facade, body)

    assert result == ["second [b](https://b.test)", True]


def test_an_interrupted_taga_row_keeps_answer_text_and_cut_off_chrome(tui) -> None:
    """Cut-off is sibling UI chrome, not rewritten into the model answer."""
    facade = tui.VoiceCodexTUI()
    answer_source = ""
    cutoff_plain = ""
    stored_text = ""

    async def body(pilot):
        nonlocal answer_source, cutoff_plain, stored_text
        text = "Half **a** thought"
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta(text)
        facade.app.action_interrupt()
        await pilot.pause()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.TAGA)
        md = await wait_markdown(row, pilot, text)
        cutoff = row.query_one(".entry-cutoff", Static)
        answer_source = md.source
        cutoff_plain = cutoff.content.plain
        stored_text = row.entry.text

    drive(facade, body)

    assert answer_source == "Half **a** thought"
    assert "cut off" in cutoff_plain
    assert "⊥" in cutoff_plain
    assert stored_text == "Half **a** thought"


def test_static_rows_carry_cut_off_when_they_are_not_markdown(tui) -> None:
    """Voice has no Markdown host, so interrupt chrome lives in the Static body."""
    cut = tui.render_entry_body(
        tui.Entry(kind="speech", source=tui.VOICE, text="mid", interrupted=True)
    )

    assert isinstance(cut, Text)
    assert "mid" in cut.plain
    assert "cut off" in cut.plain


def test_cut_off_chrome_can_be_refreshed_and_removed(tui) -> None:
    """Interrupt chrome is synced as its own widget, not only at compose time."""
    facade = tui.VoiceCodexTUI()
    had_cutoff: list[bool] = []

    async def body(pilot):
        row = facade.app.add_entry(
            tui.Entry(
                kind="speech",
                source=tui.TAGA,
                text="answer",
                interrupted=True,
            )
        )
        await wait_markdown(row, pilot, "answer")
        assert row.query(".entry-cutoff")
        # A second sync while still interrupted refreshes the existing chrome.
        row.sync()
        await pilot.pause()
        had_cutoff.append(bool(list(row.query(".entry-cutoff"))))
        row.entry.interrupted = False
        row.sync()
        await pilot.pause()
        had_cutoff.append(bool(list(row.query(".entry-cutoff"))))

    drive(facade, body)

    assert had_cutoff == [True, False]


def test_emptying_a_markdown_answer_swaps_back_to_static(tui) -> None:
    """Losing the text that justified Markdown remounts a Static host cleanly."""
    facade = tui.VoiceCodexTUI()
    host_types: list[str] = []

    async def body(pilot):
        row = facade.app.add_entry(
            tui.Entry(
                kind="speech",
                source=tui.TAGA,
                text="temporary",
                interrupted=True,
            )
        )
        await wait_markdown(row, pilot, "temporary")
        host_types.append(type(next(iter(row.query(".entry-body")))).__name__)
        row.entry.text = ""
        row.sync()
        await pilot.pause()
        body = next(iter(row.query(".entry-body")))
        host_types.append(type(body).__name__)
        assert isinstance(body, Static)
        # Cut-off remains; body swap must mount before it, not drop it.
        assert list(row.query(".entry-cutoff"))

    drive(facade, body)

    assert host_types == ["Markdown", "Static"]


def test_a_thinking_entry_hides_what_it_is_thinking(tui) -> None:
    entry = tui.Entry(
        kind="reasoning", source=tui.TAGA, text="half a thought", streaming=True
    )

    body = tui.render_entry_body(entry).plain

    assert body == "thinking ▌"
    assert "half a thought" not in body


def test_a_finished_thinking_entry_shows_its_cost_and_then_its_content(tui) -> None:
    entry = tui.Entry(
        kind="reasoning", source=tui.TAGA, text="the whole thought", seconds=1.4
    )

    assert tui.render_entry_body(entry).plain == "thinking · 1.4s\nthe whole thought"


def test_thinking_never_wears_the_same_style_as_the_answer(tui) -> None:
    """The point of the section: it must not read as the reply.

    Nothing else keeps the two apart. Both are Codex rows, both carry prose,
    and they sit next to each other — so if the thinking ever renders in the
    body style, a summary becomes indistinguishable from an answer.
    """
    thinking = tui.render_entry_body(
        tui.Entry(kind="reasoning", source=tui.TAGA, text="a thought", seconds=1.4)
    )
    answer = tui.Entry(kind="speech", source=tui.TAGA, text="an answer")

    styles = [str(thinking.style)] + [str(span.style) for span in thinking.spans]

    # Finished answers leave Static entirely; reasoning stays italic Text.
    assert tui.uses_markdown_body(answer) is True
    assert isinstance(thinking, Text)
    assert all(style != tui.BODY_STYLE for style in styles)
    # The prose itself is italic; only the duration label is allowed to be a
    # bare dim colour.
    assert "italic" in str(thinking.style)
    assert "italic" in styles[-1]


def test_thinking_that_was_never_timed_says_only_that_it_happened(tui) -> None:
    """A section closed without a duration still renders rather than raising."""
    entry = tui.Entry(kind="reasoning", source=tui.TAGA, text="a thought")

    assert tui.render_entry_body(entry).plain == "thinking\na thought"


def test_a_note_entry_renders_as_plain_text(tui) -> None:
    assert tui.render_entry_body(tui.Entry(kind="note", text="a note")).plain == (
        "a note"
    )


def test_the_transcript_shows_committed_entries(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.commit(tui.VOICE, "visible text")
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
        # Flush unfinished rows into the session file before shutdown.
        "finish_recording",
        # Snapshot of live rows for transcript.save — application, not paint.
        "transcript_entries",
        # Wire SessionState partials onto controller AppState (#102 Q3a).
        "bind_partial_publisher",
        # Wire host-owned session state onto controller AppState.
        "bind_session_state_publisher",
        # Sidebar panels the host fills in. These are not part of a Codex
        # turn, so no presentation protocol describes them.
        "set_audio",
        "set_microphones",
        "set_session",
        "set_status",
    }
)


def protocol_methods():
    from tagalong import presentation

    names: set[str] = set()
    for protocol in (
        presentation.TranscriptSink,
        presentation.MessageSink,
        presentation.SessionStatusSink,
        presentation.CodexStreamSink,
        presentation.ApplicationListSink,
        presentation.NewSessionSink,
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
        facade.codex_message_open(tui.VOICE)
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
        entry.text for entry in facade.app.entries if entry.source == tui.TAGA
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


def test_the_open_thinking_row_is_never_unmounted_under_it(tui) -> None:
    """A long thought is written to while the transcript moves past it."""
    facade = tui.VoiceCodexTUI()
    facade.app.MAX_MOUNTED_ROWS = 2

    async def body(pilot):
        facade.reasoning_started()
        thinking_row = facade.app._reasoning_row
        for index in range(8):
            facade.note(f"line {index}")
        await pilot.pause()
        assert thinking_row in mounted_rows(facade)
        facade.reasoning_delta("a late thought")
        facade.reasoning_completed()
        await pilot.pause()

    drive(facade, body)

    thinking = next(entry for entry in facade.app.entries if entry.kind == "reasoning")
    assert thinking.text == "a late thought"
    assert thinking.seconds is not None


# --------------------------------------------------------------------------
# Scrolling back through the history
#
# The window is what is mounted, not what is kept. These hold the property
# that makes the cap above affordable: every entry can still be reached by
# scrolling, and the run of mounted rows stays bounded while it is reached.
# --------------------------------------------------------------------------


def wheel(transcript, delta: int):
    kind = events.MouseScrollUp if delta < 0 else events.MouseScrollDown
    return kind(
        widget=transcript,
        x=0,
        y=0,
        delta_x=0,
        delta_y=delta,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


async def scroll_back(pilot, facade):
    """Take the view to the top and turn the wheel, as a reader does."""
    transcript = facade.app.query_one("#transcript")
    transcript.scroll_home(animate=False)
    await pilot.pause()
    transcript.post_message(wheel(transcript, -1))
    await pilot.pause()
    await pilot.pause()


async def scroll_forward(pilot, facade):
    transcript = facade.app.query_one("#transcript")
    transcript.scroll_end(animate=False)
    await pilot.pause()
    transcript.post_message(wheel(transcript, 1))
    await pilot.pause()
    await pilot.pause()


def windowed(facade, mounted_limit=20, page=20, ceiling=60):
    facade.app.MAX_MOUNTED_ROWS = mounted_limit
    facade.app.SCROLLBACK_PAGE_ROWS = page
    facade.app.MAX_SCROLLBACK_ROWS = ceiling


def mounted_texts(facade):
    return [row.entry.text for row in mounted_rows(facade)]


def test_scrolling_back_reaches_entries_the_window_left_behind(tui) -> None:
    """The point of the whole mechanism: no entry is out of reach."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    reached: list[list[str]] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        assert "line 0" not in mounted_texts(facade)
        for _ in range(12):
            await scroll_back(pilot, facade)
        reached.append(mounted_texts(facade))

    drive(facade, body)

    assert "line 0" in reached[0]
    assert reached[0][:3] == ["line 0", "line 1", "line 2"]


def test_paging_older_rows_in_leaves_the_view_where_it_was(tui) -> None:
    """Mounting above the view must not slide it down under the reader."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    seen: list[str] = []
    reachable: list[list[str]] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        transcript = facade.app.query_one("#transcript")
        transcript.scroll_home(animate=False)
        await pilot.pause()
        seen.append(facade.app._top_row(transcript).entry.text)
        transcript.post_message(wheel(transcript, -1))
        await pilot.pause()
        await pilot.pause()
        seen.append(facade.app._top_row(transcript).entry.text)
        reachable.append(mounted_texts(facade))

    drive(facade, body)

    assert seen == ["line 180", "line 180"]
    assert reachable[0][0] == "line 160"


def test_scrolling_back_keeps_the_mounted_run_bounded(tui) -> None:
    """Reachable history must not cost an unbounded number of widgets."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    counted: list[int] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        for _ in range(12):
            await scroll_back(pilot, facade)
            counted.append(len(mounted_rows(facade)))

    drive(facade, body)

    assert max(counted) <= facade.app.MAX_SCROLLBACK_ROWS


def test_a_held_back_view_still_records_what_arrives(tui) -> None:
    """New entries land in the record without moving what is being read."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    read: list[list[str]] = []

    async def body(pilot):
        for index in range(100):
            facade.note(f"line {index}")
        await pilot.pause()
        await scroll_back(pilot, facade)
        before = mounted_texts(facade)
        for index in range(100, 130):
            facade.note(f"line {index}")
        await pilot.pause()
        read.append(before)
        read.append(mounted_texts(facade))

    drive(facade, body)

    assert read[0] == read[1]
    assert entry_texts(facade)[-1] == "line 129"
    assert "line 129" not in read[1]


def test_returning_to_the_bottom_follows_the_live_end_again(tui) -> None:
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    shown: list[list[str]] = []

    async def body(pilot):
        for index in range(100):
            facade.note(f"line {index}")
        await pilot.pause()
        await scroll_back(pilot, facade)
        assert not facade.app._tailing
        for index in range(100, 120):
            facade.note(f"line {index}")
        await pilot.pause()
        for _ in range(12):
            await scroll_forward(pilot, facade)
            if facade.app._tailing:
                break
        await pilot.pause()
        shown.append(mounted_texts(facade))

    drive(facade, body)

    assert facade.app._tailing
    assert shown[0][-1] == "line 119"


def test_a_stream_scrolled_away_from_keeps_the_row_it_is_writing_to(tui) -> None:
    """The row created out of view is the row that is mounted, not a copy."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    shown: list[list[str]] = []

    async def body(pilot):
        for index in range(100):
            facade.note(f"line {index}")
        await pilot.pause()
        await scroll_back(pilot, facade)
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("an answer written out of view")
        await pilot.pause()
        for _ in range(12):
            await scroll_forward(pilot, facade)
            if facade.app._tailing:
                break
        facade.end_codex()
        await pilot.pause()
        shown.append(mounted_texts(facade))

    drive(facade, body)

    assert shown[0][-1] == "an answer written out of view"
    assert shown[0].count("an answer written out of view") == 1


def test_arriving_at_the_bottom_returns_to_the_live_end_at_once(tui) -> None:
    """However far back the reader went, the bottom is the live end."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    arrived: list[tuple[bool, str, int]] = []

    async def body(pilot):
        for index in range(300):
            facade.note(f"line {index}")
        await pilot.pause()
        for _ in range(12):
            await scroll_back(pilot, facade)
        assert facade.app._window_start < 100
        await scroll_forward(pilot, facade)
        arrived.append(
            (
                facade.app._tailing,
                mounted_texts(facade)[-1],
                len(facade.app._window),
            )
        )

    drive(facade, body)

    assert arrived[0][0] is True
    assert arrived[0][1] == "line 299"
    assert arrived[0][2] <= facade.app.MAX_MOUNTED_ROWS


def test_scrolling_forward_off_the_bottom_edge_pages_nothing(tui) -> None:
    """Only arriving at the bottom asks for the entries below it."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    windows: list[int] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        await scroll_back(pilot, facade)
        await scroll_back(pilot, facade)
        transcript = facade.app.query_one("#transcript")
        transcript.scroll_home(animate=False)
        await pilot.pause()
        windows.append(facade.app._window_end)
        transcript.post_message(wheel(transcript, 1))
        await pilot.pause()
        await pilot.pause()
        windows.append(facade.app._window_end)

    drive(facade, body)

    assert windows[0] == windows[1]
    assert not facade.app._tailing


def test_dragging_the_scrollbar_reads_as_scrolling_too(tui) -> None:
    """A drag never reaches the wheel handler, and must still be heard."""
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    walked: list[tuple[bool, int]] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        transcript = facade.app.query_one("#transcript")

        async def drag_to(y):
            transcript.post_message(ScrollTo(y=y, animate=False))
            await pilot.pause()
            await pilot.pause()

        await drag_to(0)
        walked.append((facade.app._tailing, facade.app._window_start))
        await drag_to(transcript.max_scroll_y)
        walked.append((facade.app._tailing, facade.app._window_start))

    drive(facade, body)

    assert walked[0] == (False, 160)
    assert walked[1][0] is True


def test_clicking_the_scrollbar_gutter_reads_as_scrolling_too(tui) -> None:
    facade = tui.VoiceCodexTUI()
    windowed(facade)
    walked: list[bool] = []

    async def body(pilot):
        for index in range(200):
            facade.note(f"line {index}")
        await pilot.pause()
        transcript = facade.app.query_one("#transcript")
        transcript.post_message(ScrollUp())
        await pilot.pause()
        walked.append(facade.app._tailing)
        transcript.scroll_end(animate=False)
        await pilot.pause()
        transcript.post_message(ScrollDown())
        await pilot.pause()
        await pilot.pause()
        walked.append(facade.app._tailing)

    drive(facade, body)

    assert walked == [False, True]


def test_an_empty_transcript_has_no_row_at_the_top(tui) -> None:
    facade = tui.VoiceCodexTUI()
    top: list[object] = []

    async def body(pilot):
        transcript = facade.app.query_one("#transcript")
        top.append(facade.app._top_row(transcript))
        await pilot.pause()

    drive(facade, body)

    assert top == [None]


def test_a_stream_at_the_far_end_survives_scrolling_back_past_it(tui) -> None:
    """The ceiling gives rows back from the live end — but never that one."""
    facade = tui.VoiceCodexTUI()
    windowed(facade, mounted_limit=5, page=5, ceiling=10)

    async def body(pilot):
        for index in range(60):
            facade.note(f"line {index}")
        await pilot.pause()
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("still open")
        await pilot.pause()
        streaming = facade.app._streaming
        for _ in range(4):
            await scroll_back(pilot, facade)
        assert streaming in facade.app._window
        assert len(facade.app._window) > facade.app.MAX_SCROLLBACK_ROWS
        facade.end_codex()
        await pilot.pause()

    drive(facade, body)

    assert facade.app.entries[-1].text == "still open"


# --------------------------------------------------------------------------
# Coalescing the stream
# --------------------------------------------------------------------------


def test_streamed_deltas_do_not_repaint_once_per_token(tui) -> None:
    """Codex streams faster than a terminal can usefully redraw."""
    facade = tui.VoiceCodexTUI()
    repaints: list[int] = []

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
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
        facade.codex_message_open(tui.VOICE)
        for word in ("one ", "two ", "three"):
            facade.codex_delta(word)
        facade.end_codex()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.TAGA)
        md = await wait_markdown(row, pilot, "one two three")
        drawn.append(md.source)

    drive(facade, body)

    assert entry_texts(facade) == ["one two three"]
    assert drawn == ["one two three"]
    assert facade.app._dirty == []


def test_an_interrupted_answer_shows_the_text_that_arrived_before_the_cut(tui) -> None:
    """An interrupt must not strand text the flush timer had not drawn yet."""
    facade = tui.VoiceCodexTUI()
    drawn: list[str] = []

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        facade.codex_delta("half a th")
        facade.app.action_interrupt()
        await pilot.pause()
        row = next(r for r in mounted_rows(facade) if r.entry.source == tui.TAGA)
        md = await wait_markdown(row, pilot, "half a th")
        cutoff = row.query_one(".entry-cutoff", Static)
        drawn.extend([md.source, cutoff.content.plain])

    drive(facade, body)

    assert drawn[0] == "half a th"
    assert "cut off" in drawn[1]
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
    from tagalong.control import Controller

    controller = Controller(transcript=facade.transcript)
    facade.bind_session_state_publisher(controller.set_session_state)
    _snapshot, subscription = controller.subscribe()
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
    assert controller.state.tokens == 1234
    changed = [event for event in subscription.drain() if event.name == "state.changed"]
    assert dict(changed[-1].payload) == {"tokens": 1234}


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


def test_a_partial_update_never_waits_on_the_application_thread(tui) -> None:
    """Partials come from recognition; waiting there stalls transcription."""
    facade = tui.VoiceCodexTUI()
    waited: list[str] = []

    async def body(pilot):
        facade._app_thread = -1
        facade.app.call_from_thread = lambda *a, **k: waited.append("blocked")
        facade.update(tui.VOICE, "half a sen")
        await pilot.pause()

    drive(facade, body)

    assert waited == []
    assert facade.state.partial_text == "half a sen"


def test_partials_before_ready_still_paint_when_the_app_starts(tui) -> None:
    """A sticky pending bit before mount would leave the live line blank."""
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade._ready.clear()
        facade.update(tui.VOICE, "early")
        facade.update(tui.VOICE, "early more")
        assert facade._partial_pending is False
        assert facade.state.partial_text == "early more"
        facade._on_app_ready()
        await pilot.pause()
        assert "early more" in facade.app.query_one("#partial", Static).content.plain

    drive(facade, body)


def test_deltas_before_ready_still_land_when_the_app_starts(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        facade._ready.clear()
        facade.codex_delta("buffered ")
        facade.codex_delta("answer")
        assert facade._answer_deltas.buffered() is True
        facade._on_app_ready()
        await pilot.pause()
        assert entry_texts(facade) == ["buffered answer"]

    drive(facade, body)


def test_begin_codex_discards_orphaned_stream_buffers(tui) -> None:
    facade = tui.VoiceCodexTUI()
    facade._answer_deltas.append("stale answer")
    facade._reasoning_deltas.append("stale thought")

    facade.begin_codex()

    assert facade._answer_deltas.buffered() is False
    assert facade._reasoning_deltas.buffered() is False


def test_interrupt_drains_buffered_reasoning_as_well_as_answers(tui) -> None:
    facade = tui.VoiceCodexTUI()

    async def body(pilot):
        facade.reasoning_started()
        await pilot.pause()
        facade._app_thread = -1
        facade.app.call_later = lambda fn, *args: None
        facade.reasoning_delta("still thinking")
        facade._app_thread = threading.get_ident()
        assert facade._reasoning_deltas.buffered() is True
        await pilot.press("ctrl+x")
        await pilot.pause()
        assert facade._reasoning_deltas.buffered() is False
        assert any(entry.text == "still thinking" for entry in facade.app.entries)

    drive(facade, body)


def test_rapid_partials_coalesce_into_one_repaint(tui) -> None:
    """Only the newest text matters; queueing every revision wastes layout."""
    facade = tui.VoiceCodexTUI()
    scheduled: list[tuple] = []

    async def body(pilot):
        facade._app_thread = -1
        facade.app.call_later = lambda fn, *args: scheduled.append((fn, args))
        facade.update(tui.VOICE, "a")
        facade.update(tui.VOICE, "ab")
        facade.update(tui.VOICE, "abc")
        assert len(scheduled) == 1
        assert facade.state.partial_text == "abc"
        assert facade._partial_pending is True
        fn, args = scheduled[0]
        fn(*args)
        assert facade._partial_pending is False
        await pilot.pause()

    drive(facade, body)


def test_a_codex_delta_never_waits_on_the_application_thread(tui) -> None:
    """Token deltas come from the stream worker; waiting there stalls replies."""
    facade = tui.VoiceCodexTUI()
    waited: list[str] = []

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        facade._app_thread = -1
        facade.app.call_from_thread = lambda *a, **k: waited.append("blocked")
        facade.codex_delta("Hello")
        await pilot.pause()

    drive(facade, body)

    assert waited == []
    assert entry_texts(facade) == ["Hello"]


def test_rapid_codex_deltas_coalesce_into_one_append(tui) -> None:
    """Tokens stay ordered; joining them once beats ``text +=`` per token."""
    facade = tui.VoiceCodexTUI()
    scheduled: list[tuple] = []

    async def body(pilot):
        facade.codex_message_open(tui.VOICE)
        await pilot.pause()
        facade._app_thread = -1
        facade.app.call_later = lambda fn, *args: scheduled.append((fn, args))
        facade.codex_delta("one ")
        facade.codex_delta("two ")
        facade.codex_delta("three")
        assert len(scheduled) == 1
        fn, args = scheduled[0]
        fn(*args)
        assert entry_texts(facade) == ["one two three"]
        await pilot.pause()

    drive(facade, body)


def test_rapid_reasoning_deltas_coalesce_into_one_append(tui) -> None:
    facade = tui.VoiceCodexTUI()
    scheduled: list[tuple] = []

    async def body(pilot):
        facade.reasoning_started()
        await pilot.pause()
        facade._app_thread = -1
        facade.app.call_later = lambda fn, *args: scheduled.append((fn, args))
        facade.reasoning_delta("one ")
        facade.reasoning_delta("two")
        assert len(scheduled) == 1
        fn, args = scheduled[0]
        fn(*args)
        assert entry_texts(facade) == ["one two"]
        await pilot.pause()

    drive(facade, body)
