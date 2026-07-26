from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input, Select, Static


def test_meter_clamps_audio_levels(tui) -> None:
    assert tui.meter(-1, width=4).plain == "□□□□"
    assert tui.meter(2, width=4).plain == "■■■■"


def test_quiet_response_policy_is_labeled_stay_silent(tui) -> None:
    assert tui.POLICIES["quiet"] == "stay silent"


def test_facade_updates_state_before_the_app_starts(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.partial(tui.USER_VOICE, "testing")
    facade.set_audio("mic", device="USB mic", level=3)
    facade.set_policy("quiet")

    assert facade.state.partial_text == "testing"
    assert facade.state.mic.device == "USB mic"
    assert facade.state.mic.level == 1.0
    assert facade.state.policy == "quiet"


def test_facade_implements_runtime_display_events_before_the_app_starts(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.update(tui.USER_VOICE, "testing")
    facade.begin_codex()
    facade.codex_message_open(tui.THEM)
    facade.token_usage(123)
    facade.end_codex()

    assert facade.state.partial_text == ""
    assert facade.state.codex_state == "idle"
    assert facade.state.tokens == 123


def test_textual_app_accepts_typed_input_and_records_a_transcript_entry(tui) -> None:
    received: list[str] = []
    app = tui.VoiceCodexApp(
        tui.SessionState(),
        tui.TuiHooks(on_user_text=received.append),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", Input)
            input_widget.value = "hello from test"
            await pilot.press("enter")

    asyncio.run(exercise())
    app._tick()

    assert received == ["hello from test"]
    assert [(entry.source, entry.text) for entry in app.entries] == [
        (tui.USER_TEXT, "hello from test")
    ]


def test_response_picker_has_a_descriptive_label_above_it(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test():
            label = app.query_one("#policy-label", Static)
            picker = app.query_one("#policy-select", Select)

            assert str(label.render()) == "AI agent responds to:"
            assert label.parent is picker.parent

    asyncio.run(exercise())


def test_codex_selectors_change_model_and_reasoning_effort(tui) -> None:
    received: list[tuple[str, str]] = []
    state = tui.SessionState(
        codex_model="gpt-5.6-luna",
        codex_effort="low",
        codex_models=[("Luna", "gpt-5.6-luna"), ("Sol", "gpt-5.6-sol")],
        codex_efforts=["low"],
        codex_efforts_by_model={
            "gpt-5.6-luna": ["low"],
            "gpt-5.6-sol": ["low", "high"],
        },
        codex_default_effort_by_model={
            "gpt-5.6-luna": "low",
            "gpt-5.6-sol": "high",
        },
    )
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_codex_model=lambda model: received.append(("model", model)),
            on_codex_effort=lambda effort: received.append(("effort", effort)),
        ),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#model-select", Select).value = "gpt-5.6-sol"
            await pilot.pause()
            app.query_one("#reasoning-select", Select).value = "high"
            await pilot.pause()

    asyncio.run(exercise())

    assert state.codex_model == "gpt-5.6-sol"
    assert state.codex_effort == "high"
    assert received == [("model", "gpt-5.6-sol"), ("effort", "high")]


def test_transcript_rows_can_be_selected_for_copying(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            row = app.add_entry(
                tui.Entry(kind="speech", source=tui.USER_VOICE, text="copy this")
            )
            await pilot.pause()
            body = row.query_one(".entry-body", Static)
            assert body.allow_select is True
            app.screen._select_all_in_widget(row)

            assert "copy this" in (app.screen.get_selected_text() or "")

    asyncio.run(exercise())


def test_transcript_timestamp_column_fits_a_full_timestamp(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            row = app.add_entry(
                tui.Entry(
                    kind="speech",
                    source=tui.USER_VOICE,
                    text="test",
                    stamp="13:40:46",
                )
            )
            await pilot.pause()

            stamp = row.query_one(".entry-stamp", Static)
            assert stamp.styles.width.value == 9

    asyncio.run(exercise())


@pytest.mark.parametrize("key", ["ctrl+c", "ctrl+w"])
def test_control_c_and_control_w_quit_the_app(tui, key: str) -> None:
    quit_calls: list[None] = []
    app = tui.VoiceCodexApp(
        tui.SessionState(),
        tui.TuiHooks(on_quit=lambda: quit_calls.append(None)),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press(key)

    asyncio.run(exercise())

    assert quit_calls == [None]


def test_textual_tts_control_stays_off_when_the_runtime_refuses_it(tui) -> None:
    app = tui.VoiceCodexApp(
        tui.SessionState(tts_enabled=False),
        tui.TuiHooks(on_tts=lambda enabled: False),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")

    asyncio.run(exercise())

    assert app.state.tts_enabled is False
    assert app.entries[-1].text == "tts unavailable for this session"
