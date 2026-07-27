from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading

from rich.console import Console
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


def _catalog_state(tui, **overrides):
    """A session that knows two models with disjoint reasoning-effort lists."""
    return tui.SessionState(
        **{
            "codex_model": "luna",
            "codex_effort": "high",
            "codex_models": [("Luna", "luna"), ("Sol", "sol")],
            "codex_efforts": ["low", "medium", "high"],
            "codex_efforts_by_model": {
                "luna": ["low", "medium", "high"],
                "sol": ["minimal"],
            },
            "codex_default_effort_by_model": {"luna": "high", "sol": "minimal"},
            **overrides,
        }
    )


def _sidebar_snapshot(app, act=None) -> dict[str, str]:
    """Mount ``app``, run ``act``, let the pickers settle, then read them back.

    The picker writes bounce back through the message queue, so the snapshot is
    taken after several frames and before ``run_test`` tears the DOM down.
    """
    snapshot: dict[str, str] = {}

    async def exercise() -> None:
        async with app.run_test() as pilot:
            if act is not None:
                act()
            for _ in range(4):
                await pilot.pause()
            console = Console(width=40)
            with console.capture() as capture:
                console.print(app.query_one("#panel-codex", Static).content)
            snapshot["model"] = app.query_one("#model-select", Select).value
            snapshot["effort"] = app.query_one("#reasoning-select", Select).value
            snapshot["codex-panel"] = capture.get()

    asyncio.run(exercise())
    return snapshot


def _set_picker(app, selector: str, value: str):
    return lambda: setattr(app.query_one(selector, Select), "value", value)


def test_pickers_accept_a_startup_model_and_effort_the_catalog_omits(tui) -> None:
    app = tui.VoiceCodexApp(
        tui.SessionState(codex_model="gpt-5.6-nebula", codex_effort="high"),
        tui.TuiHooks(),
    )

    snapshot = _sidebar_snapshot(app)

    assert snapshot["model"] == "gpt-5.6-nebula"
    assert snapshot["effort"] == "high"


def test_installing_the_codex_catalog_does_not_fire_the_pickers_hooks(tui) -> None:
    received: list[tuple[str, str]] = []
    facade = tui.VoiceCodexTUI(
        tui.SessionState(codex_model="luna", codex_effort="high"),
        on_codex_model=lambda model: received.append(("model", model)),
        on_codex_effort=lambda effort: received.append(("effort", effort)),
    )

    def install() -> None:
        # Stand in for run(), which normally marks the app thread as ready.
        facade._app_thread = threading.get_ident()
        facade._ready.set()
        facade.set_codex_catalog(
            [("Sol", "sol"), ("Luna", "luna")],
            {"luna": ["low", "high"], "sol": ["minimal"]},
            {"luna": "high", "sol": "minimal"},
        )

    snapshot = _sidebar_snapshot(facade.app, install)

    assert received == []
    assert (facade.state.codex_model, facade.state.codex_effort) == ("luna", "high")
    assert (snapshot["model"], snapshot["effort"]) == ("luna", "high")


def test_changing_the_model_only_reports_the_effort_it_settles_on(tui) -> None:
    received: list[tuple[str, str]] = []
    state = _catalog_state(tui)
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_codex_model=lambda model: received.append(("model", model)),
            on_codex_effort=lambda effort: received.append(("effort", effort)),
        ),
    )

    snapshot = _sidebar_snapshot(app, _set_picker(app, "#model-select", "sol"))

    assert received == [("model", "sol"), ("effort", "minimal")]
    assert (state.codex_model, state.codex_effort) == ("sol", "minimal")
    assert (snapshot["model"], snapshot["effort"]) == ("sol", "minimal")


def test_a_refused_effort_switch_restores_the_whole_model_choice(tui) -> None:
    state = _catalog_state(tui)
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_codex_model=lambda model: True,
            on_codex_effort=lambda effort: False,
        ),
    )

    snapshot = _sidebar_snapshot(app, _set_picker(app, "#model-select", "sol"))

    assert (state.codex_model, state.codex_effort) == ("luna", "high")
    assert state.codex_efforts == ["low", "medium", "high"]
    assert (snapshot["model"], snapshot["effort"]) == ("luna", "high")


def test_choosing_an_effort_repaints_the_codex_panel(tui) -> None:
    state = _catalog_state(tui)
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    snapshot = _sidebar_snapshot(app, _set_picker(app, "#reasoning-select", "low"))

    assert state.codex_effort == "low"
    assert "low · standard" in snapshot["codex-panel"]


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


def test_keyboard_shortcut_legend_includes_copy_paste_and_quit(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    assert app._keys_text().plain.endswith(
        "^S save transcript  ^Q quit\n^⇧C copy  ^V paste"
    )


def test_ctrl_c_clears_text_before_quitting_the_application(tui) -> None:
    cleaned_up: list[bool] = []
    app = tui.VoiceCodexApp(
        tui.SessionState(),
        tui.TuiHooks(on_quit=lambda: cleaned_up.append(True)),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", Input)
            input_widget.value = "start over"
            await pilot.press("ctrl+c")

            assert input_widget.value == ""
            assert app._exit is False
            assert cleaned_up == []

            await pilot.press("ctrl+c")
            assert app._exit is True
            assert cleaned_up == [True]

    asyncio.run(exercise())


def test_ctrl_shift_c_copies_selected_transcript_rows_as_tabular_text(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            row = app.add_entry(
                tui.Entry(
                    kind="speech",
                    source=tui.USER_VOICE,
                    text="copy this",
                    stamp="13:40:46",
                )
            )
            await pilot.pause()
            app.screen._select_all_in_widget(row)
            await pilot.press("ctrl+shift+c")
            assert app.clipboard == "13:40:46\tUser Voice\tcopy this"

            await pilot.press("ctrl+v")
            assert app.query_one("#input", Input).value == app.clipboard

    asyncio.run(exercise())


def test_ctrl_v_pastes_clipboard_text_into_the_input(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.copy_to_clipboard("pasted text")
            await pilot.press("ctrl+v")

            assert app.query_one("#input", Input).value == "pasted text"

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


def test_tui_disables_native_interrupts_before_textual_import() -> None:
    environment = os.environ | {
        "TEXTUAL_ALLOW_SIGNALS": "",
        "TEXTUAL_DISABLE_KITTY_KEY": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from voice_codex import tui; "
                "from textual import constants; "
                "print(os.environ.get('TEXTUAL_ALLOW_SIGNALS')); "
                "print(os.environ.get('TEXTUAL_DISABLE_KITTY_KEY')); "
                "print(constants.DISABLE_KITTY_KEY)"
            ),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.stdout.splitlines() == ["None", "None", "False"]


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
