from __future__ import annotations

import asyncio

from textual.widgets import Input


def test_meter_clamps_audio_levels(tui) -> None:
    assert tui.meter(-1, width=4).plain == "□□□□"
    assert tui.meter(2, width=4).plain == "■■■■"


def test_facade_updates_state_before_the_app_starts(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.partial(tui.USER_VOICE, "testing")
    facade.set_audio("mic", device="USB mic", level=3)
    facade.set_models(["gpt-5.6-codex"])
    facade.set_policy("quiet")

    assert facade.state.partial_text == "testing"
    assert facade.state.mic.device == "USB mic"
    assert facade.state.mic.level == 1.0
    assert facade.state.codex_models == [
        facade.state.codex_model,
        "gpt-5.6-codex",
    ]
    assert facade.state.policy == "quiet"


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
