from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console
from textual.widgets import Checkbox, Input, Link, Select, Static


def _rendered(renderable, width: int = 40) -> str:
    """Render a Rich renderable the way the sidebar would draw it."""
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_sidebar_setting_groups_are_divided_by_visible_separators(tui) -> None:
    sidebar = tui.Sidebar(tui.SessionState(), tui.TuiHooks())

    audio = [_rendered(item) for item in sidebar._audio_head()]
    codex = [_rendered(item) for item in sidebar._codex_head()]
    tts = [_rendered(item) for item in sidebar._tts_head()]
    lower_sections = [_rendered(item) for item in sidebar._bottom()]

    separator = "─" * 40 + "\n"
    assert audio[0] == separator
    assert codex[0] == separator
    assert tts[1] == separator
    assert lower_sections[-2] == separator


def test_speech_engine_and_voice_are_grouped_in_the_tts_section(tui) -> None:
    state = tui.SessionState(tts_voice="en_US-amy-medium")
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test():
            sidebar = app.query_one("#sidebar", tui.Sidebar)
            ids = [child.id for child in sidebar.children]
            details = app.query_one("#panel-tts", Static)

            assert ids.index("panel-bottom") < ids.index("speech-row")
            assert ids.index("speech-row") < ids.index("panel-tts")
            assert "voice" in _rendered(details.content)
            assert "en_US-amy-medium" in _rendered(details.content)

    asyncio.run(exercise())


def test_sidebar_links_to_the_github_repository(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test():
            link = app.query_one("#repository-link", Link)
            sidebar = app.query_one("#sidebar", tui.Sidebar)

            assert str(link.render()) == " GitHub ↗"
            assert link.url == "https://github.com/gabepsilva/microphone"
            assert link.region.bottom == sidebar.content_region.bottom

    asyncio.run(exercise())


def test_the_sound_dot_shows_whether_a_channel_hears_anything(tui) -> None:
    assert tui.sound_dot(True).plain == tui.SOUND_ON
    assert tui.sound_dot(False).plain == tui.SOUND_OFF


def test_a_quiet_channel_dims_its_dot_instead_of_colouring_it(tui) -> None:
    """The colour is the signal, so silence must not wear the live one."""
    assert str(tui.sound_dot(True, "#6ba7ff").style) == "#6ba7ff"
    assert str(tui.sound_dot(False, "#6ba7ff").style) != "#6ba7ff"


def test_quiet_response_policy_is_labeled_stay_silent(tui) -> None:
    assert tui.POLICIES["quiet"] == "stay silent"


def test_facade_updates_state_before_the_app_starts(tui) -> None:
    facade = tui.VoiceCodexTUI()

    facade.update(tui.USER_VOICE, "testing")
    facade.set_audio("mic", device="USB mic", active=True)

    assert facade.state.partial_text == "testing"
    assert facade.state.mic.device == "USB mic"
    assert facade.state.mic.active is True


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


def test_session_clock_computes_elapsed_from_timezone_aware_timestamps(tui) -> None:
    """The clock subtracts two datetimes, so both sides must carry a timezone.

    A naive ``datetime.now()`` on either side raises TypeError against an aware
    one, and the session panel is repainted on every tick.
    """
    assert tui.SessionState().started.tzinfo is not None

    started = datetime.now(UTC) - timedelta(hours=1, minutes=2, seconds=3)
    app = tui.VoiceCodexApp(tui.SessionState(started=started), tui.TuiHooks())
    rendered: list[str] = []

    async def exercise() -> None:
        async with app.run_test():
            app.query_one("#sidebar", tui.Sidebar).sync_clock()

            panel = app.query_one("#panel-clock", Static)
            rendered.extend(panel.render_line(y).text for y in range(panel.size.height))

    asyncio.run(exercise())

    assert any("01:02:03" in line for line in rendered), rendered


def test_transcript_stamp_uses_local_wall_clock_time(tui) -> None:
    """Timestamps are read by a person watching the session, not stored as UTC."""
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())

    now = datetime.now(UTC).astimezone()
    # Accept a second of drift so a rollover between the two reads cannot flake.
    acceptable = {
        (now + timedelta(seconds=offset)).strftime("%H:%M:%S") for offset in (-1, 0, 1)
    }

    assert app._stamp() in acceptable


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


def test_the_sidebar_picker_offers_every_defined_policy(tui) -> None:
    from voice_codex.domain import RESPONSE_POLICIES

    # The picker's labels are the domain's, so a policy added there appears
    # here without a second list to remember to update.
    assert {
        name: policy.sidebar_label for name, policy in RESPONSE_POLICIES.items()
    } == tui.POLICIES
    assert set(tui.POLICIES) == set(RESPONSE_POLICIES)


def test_the_speech_picker_offers_every_provider_and_silence(tui) -> None:
    from voice_codex.speech import NO_VOICE, NO_VOICE_LABEL, PROVIDER_LABELS

    # The picker's labels are the speech boundary's, so a provider added there
    # appears here without a second list to remember to update. Silence is
    # last: it is the answer to the same question, not another engine.
    state = tui.SessionState()
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    assert sidebar._speech_options() == [
        *((label, name) for name, label in PROVIDER_LABELS.items()),
        (NO_VOICE_LABEL, NO_VOICE),
    ]


def test_the_far_end_picker_offers_silence_before_the_applications(tui) -> None:
    state = tui.SessionState(them_streams=[("Brave (playing)", "Brave")])
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    assert sidebar._them_options() == [
        (tui.NO_THEM_LABEL, tui.NO_THEM),
        ("Brave (playing)", "Brave"),
    ]


def test_the_microphone_picker_offers_every_input_device(tui) -> None:
    state = tui.SessionState(
        microphone="2",
        microphones=[("Yeti", "0"), ("Webcam", "2")],
    )
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    assert sidebar._microphone_options() == [("Yeti", "0"), ("Webcam", "2")]


def test_choosing_a_microphone_asks_the_host_and_adopts_it(tui) -> None:
    chosen: list[str] = []
    state = tui.SessionState(
        mic=tui.Channel("mic", device="Yeti"),
        microphone="0",
        microphones=[("Yeti", "0"), ("Webcam", "2")],
    )
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(on_microphone=lambda device: chosen.append(device) or True),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#mic-select", Select).value = "2"
            await pilot.pause()

    asyncio.run(exercise())

    assert chosen == ["2"]
    assert state.microphone == "2"
    assert state.mic.device == "Webcam"


def test_a_refused_microphone_leaves_the_picker_where_it_was(tui) -> None:
    shown: list[str] = []
    state = tui.SessionState(
        mic=tui.Channel("mic", device="Yeti"),
        microphone="0",
        microphones=[("Yeti", "0"), ("Webcam", "2")],
    )
    app = tui.VoiceCodexApp(state, tui.TuiHooks(on_microphone=lambda _device: False))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#mic-select", Select).value = "2"
            await pilot.pause()
            shown.append(str(app.query_one("#mic-select", Select).value))

    asyncio.run(exercise())

    assert state.microphone == "0"
    assert state.mic.device == "Yeti"
    assert shown == ["0"]


def test_the_far_end_picker_shows_silence_when_nothing_is_chosen(tui) -> None:
    """None is not a value a Select can hold, so silence is spelled."""
    sidebar = tui.Sidebar(tui.SessionState(them_stream=None), tui.TuiHooks())

    assert sidebar._them_selection() == tui.NO_THEM


def test_choosing_an_application_asks_the_host_and_adopts_it(tui) -> None:
    chosen: list[str | None] = []
    state = tui.SessionState(them_streams=[("Brave (playing)", "Brave")])
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(on_them_stream=lambda name: chosen.append(name) or True),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#them-select", Select).value = "Brave"
            await pilot.pause()

    asyncio.run(exercise())

    assert chosen == ["Brave"]
    assert state.them_stream == "Brave"


def test_choosing_silence_asks_the_host_to_drop_the_far_end(tui) -> None:
    chosen: list[str | None] = []
    state = tui.SessionState(
        them_stream="Brave", them_streams=[("Brave (playing)", "Brave")]
    )
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(on_them_stream=lambda name: chosen.append(name) or True),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#them-select", Select).value = tui.NO_THEM
            await pilot.pause()

    asyncio.run(exercise())

    assert chosen == [None]
    assert state.them_stream is None


def test_a_refused_application_leaves_the_picker_where_it_was(tui) -> None:
    shown: list[str] = []
    state = tui.SessionState(them_streams=[("Brave (playing)", "Brave")])
    app = tui.VoiceCodexApp(state, tui.TuiHooks(on_them_stream=lambda _name: False))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#them-select", Select).value = "Brave"
            await pilot.pause()
            shown.append(str(app.query_one("#them-select", Select).value))

    asyncio.run(exercise())

    assert state.them_stream is None
    assert shown == [tui.NO_THEM]


def test_the_speech_picker_shows_silence_while_the_voice_is_off(tui) -> None:
    from voice_codex.speech import NO_VOICE

    shown: list[str] = []
    state = tui.SessionState(tts_provider="piper", tts_enabled=True)
    app = tui.VoiceCodexApp(state, tui.TuiHooks(on_tts=lambda _enabled: True))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            shown.append(str(app.query_one("#speech-select", Select).value))

    asyncio.run(exercise())

    assert state.tts_enabled is False
    # The engine is remembered underneath, so turning the voice back on has
    # somewhere to return to.
    assert state.tts_provider == "piper"
    assert shown == [NO_VOICE]


def test_choosing_no_voice_reply_silences_the_session(tui) -> None:
    from voice_codex.speech import NO_VOICE

    toggled: list[bool] = []
    state = tui.SessionState(tts_provider="piper")
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(on_tts=lambda enabled: toggled.append(enabled) or True),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = NO_VOICE
            await pilot.pause()

    asyncio.run(exercise())

    assert toggled == [False]
    assert state.tts_enabled is False
    assert state.tts_provider == "piper"


def test_choosing_an_engine_again_gives_the_session_its_voice_back(tui) -> None:
    from voice_codex.speech import default_voice

    toggled: list[bool] = []
    switched: list[str] = []
    state = tui.SessionState(tts_provider="piper", tts_enabled=False)
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_tts=lambda enabled: toggled.append(enabled) or True,
            on_tts_provider=lambda provider: switched.append(provider) or True,
        ),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = "edge"
            await pilot.pause()

    asyncio.run(exercise())

    assert switched == ["edge"]
    assert toggled == [True]
    assert state.tts_enabled is True
    assert state.tts_provider == "edge"
    assert state.tts_voice == default_voice("edge")


def test_a_silent_session_cannot_be_given_a_voice_by_the_picker(tui) -> None:
    from voice_codex.speech import NO_VOICE

    shown: list[str] = []
    # Started with --tts off: there is no engine to switch or unmute, and both
    # hooks say so.
    state = tui.SessionState(tts_provider="piper", tts_enabled=False)
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_tts=lambda _enabled: False,
            on_tts_provider=lambda _provider: False,
        ),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = "piper"
            await pilot.pause()
            shown.append(str(app.query_one("#speech-select", Select).value))

    asyncio.run(exercise())

    assert state.tts_enabled is False
    assert shown == [NO_VOICE]
    assert app.entries[-1].text == "tts unavailable for this session"


def test_the_speech_picker_starts_on_local_synthesis(tui) -> None:
    from voice_codex.speech import DEFAULT_PROVIDER, default_voice

    state = tui.SessionState()

    assert state.tts_provider == DEFAULT_PROVIDER
    assert state.tts_voice == default_voice(DEFAULT_PROVIDER)


def test_choosing_a_speech_provider_switches_the_engine_and_its_voice(tui) -> None:
    from voice_codex.speech import default_voice

    switched: list[str] = []
    state = tui.SessionState(tts_provider="piper")
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(
            on_tts_provider=lambda provider: switched.append(provider) or True
        ),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = "edge"
            await pilot.pause()

    asyncio.run(exercise())

    assert switched == ["edge"]
    assert state.tts_provider == "edge"
    assert state.tts_voice == default_voice("edge")


def test_a_refused_speech_switch_leaves_the_engine_alone(tui) -> None:
    shown: list[str] = []
    state = tui.SessionState(tts_provider="piper")
    app = tui.VoiceCodexApp(
        state,
        tui.TuiHooks(on_tts_provider=lambda _provider: False),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = "edge"
            await pilot.pause()
            # The picker has to snap back: leaving it on the engine the host
            # refused would name a provider the session is not using.
            shown.append(app.query_one("#speech-select", Select).value)

    asyncio.run(exercise())

    assert state.tts_provider == "piper"
    assert shown == ["piper"]


def test_a_session_without_a_speech_hook_cannot_switch_provider(tui) -> None:
    state = tui.SessionState(tts_provider="piper")
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#speech-select", Select).value = "edge"
            await pilot.pause()

    asyncio.run(exercise())

    assert state.tts_provider == "piper"


def test_the_countdown_bar_drains_as_the_silence_runs_out(tui) -> None:
    full = tui.countdown_bar(3.0, 3.0)
    half = tui.countdown_bar(1.5, 3.0)
    gone = tui.countdown_bar(0.0, 3.0)

    assert full.plain == "■■■■■■■■■■ 3.0s"
    assert half.plain == "■■■■■□□□□□ 1.5s"
    assert gone.plain == "□□□□□□□□□□ 0.0s"


def test_the_countdown_turns_amber_as_the_turn_is_about_to_be_sent(tui) -> None:
    # The colour is the warning: speaking now still cancels the turn.
    assert tui.countdown_bar(3.0, 3.0).spans[0].style == "#6cc06c"
    assert tui.countdown_bar(0.6, 3.0).spans[0].style == "#d7b562"


def test_a_countdown_longer_than_its_window_cannot_overfill(tui) -> None:
    assert tui.countdown_bar(9.0, 3.0).plain == "■■■■■■■■■■ 9.0s"


def test_a_zero_window_cannot_divide_by_itself(tui) -> None:
    assert tui.countdown_bar(0.0, 0.0).plain == "□□□□□□□□□□ 0.0s"


def test_the_sidebar_shows_no_countdown_when_no_turn_is_pending(tui) -> None:
    sidebar = tui.Sidebar(tui.SessionState(turn_silence=2.5), tui.TuiHooks())

    assert sidebar._countdown().plain == ""


def test_the_sidebar_shows_the_countdown_while_a_turn_waits(tui) -> None:
    state = tui.SessionState(turn_silence=3.0, turn_countdown=1.8)
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    assert sidebar._countdown().plain == "■■■■■■□□□□ 1.8s"


class FakeCountdown:
    """A silence clock a test drives directly."""

    def __init__(self, remaining=None):
        self.value = remaining
        self.reads = 0

    def remaining(self):
        self.reads += 1
        return self.value


def test_the_countdown_ticks_into_the_session_state(tui) -> None:
    countdown = FakeCountdown(2.4)
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), countdown)

    async def exercise() -> None:
        async with app.run_test():
            app._tick_countdown()

    asyncio.run(exercise())

    assert state.turn_countdown == 2.4


def test_an_idle_session_repaints_nothing_for_the_countdown(tui) -> None:
    """Ten frames a second of nothing is the cost this check exists to avoid."""
    countdown = FakeCountdown(None)
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), countdown)
    painted: list[bool] = []

    async def exercise() -> None:
        async with app.run_test():
            sidebar = app.query_one("#sidebar", tui.Sidebar)
            sidebar.sync_countdown = lambda: painted.append(True)
            app._tick_countdown()

    asyncio.run(exercise())

    # The mounted app runs its own countdown interval, so the poll count is
    # not this test's to predict. That it polls at all, and repaints nothing
    # when the answer is "no turn pending", is the whole property.
    assert countdown.reads >= 1
    assert painted == []
    assert state.turn_countdown is None


def test_the_end_of_a_countdown_is_painted_once_to_clear_it(tui) -> None:
    countdown = FakeCountdown(None)
    state = tui.SessionState(turn_countdown=0.2)
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), countdown)
    painted: list[bool] = []

    async def exercise() -> None:
        async with app.run_test():
            sidebar = app.query_one("#sidebar", tui.Sidebar)
            sidebar.sync_countdown = lambda: painted.append(True)
            # Re-armed here rather than only in the constructor: the mounted
            # app runs its own countdown interval, which may already have
            # cleared the state before this test installed its recorder.
            state.turn_countdown = 0.2
            app._tick_countdown()
            app._tick_countdown()

    asyncio.run(exercise())

    assert state.turn_countdown is None
    assert painted == [True]


def test_a_session_without_a_countdown_clock_never_shows_one(tui) -> None:
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test():
            app._tick_countdown()

    asyncio.run(exercise())

    assert state.turn_countdown is None


ACCEPTS = object()


def silence_app(tui, seconds=3.0, applied=ACCEPTS):
    """An app whose turn-silence field reports to a recording hook.

    ``applied`` is what the host claims to have adopted. ACCEPTS echoes the
    typed value back; an explicit None is a host that refused it, which is a
    different answer and must not be spelled the same way.
    """
    received: list[float] = []

    def on_turn_silence(value):
        received.append(value)
        return value if applied is ACCEPTS else applied

    app = tui.VoiceCodexApp(
        tui.SessionState(turn_silence=seconds),
        tui.TuiHooks(on_turn_silence=on_turn_silence),
    )
    return app, received


def submit_silence(app, typed, then=None):
    """Type a window into the field, submit it, and report what it holds."""
    seen: dict[str, object] = {}

    async def exercise() -> None:
        async with app.run_test() as pilot:
            field = app.query_one("#silence-input", Input)
            field.focus()
            field.value = typed
            await pilot.press("enter")
            await pilot.pause()
            if then is not None:
                await then(app, pilot)
            seen["value"] = app.query_one("#silence-input", Input).value
            seen["invalid"] = app.query_one("#silence-input", Input).has_class(
                "invalid"
            )
            seen["focused"] = app.focused.id if app.focused else None

    asyncio.run(exercise())
    return seen


def test_the_field_starts_showing_the_window_in_force(tui) -> None:
    app, _ = silence_app(tui, 2.5)
    seen: dict[str, str] = {}

    async def exercise() -> None:
        async with app.run_test():
            seen["value"] = app.query_one("#silence-input", Input).value

    asyncio.run(exercise())

    assert seen["value"] == "2.5"


def test_a_typed_window_is_applied_and_normalized(tui) -> None:
    app, received = silence_app(tui, 3.0)

    seen = submit_silence(app, "1.5")

    assert received == [1.5]
    assert app.state.turn_silence == 1.5
    assert seen["value"] == "1.5"
    assert seen["invalid"] is False


def test_applying_a_window_hands_typing_back_to_the_transcript(tui) -> None:
    app, _ = silence_app(tui, 3.0)

    seen = submit_silence(app, "1.5")

    assert seen["focused"] == "input"


def test_a_window_typed_with_its_unit_is_accepted(tui) -> None:
    app, received = silence_app(tui, 3.0)

    submit_silence(app, "2s")

    assert received == [2.0]


@pytest.mark.parametrize("typed", ["abc", "", "0", "99"])
def test_a_value_the_field_cannot_use_is_marked_and_left_alone(tui, typed) -> None:
    """The typist is mid-correction; replacing their text loses the keystrokes."""
    app, received = silence_app(tui, 3.0)

    seen = submit_silence(app, typed)

    assert received == []
    assert app.state.turn_silence == 3.0
    assert seen["value"] == typed
    assert seen["invalid"] is True


def test_a_window_the_host_refuses_leaves_the_field_marked(tui) -> None:
    app, received = silence_app(tui, 3.0, applied=None)

    seen = submit_silence(app, "1.5")

    assert received == [1.5]
    assert app.state.turn_silence == 3.0
    assert seen["invalid"] is True


def test_the_field_shows_the_window_the_host_actually_adopted(tui) -> None:
    """A host that clamps must not leave the field claiming the typed value."""
    app, _ = silence_app(tui, 3.0, applied=0.25)

    seen = submit_silence(app, "1.5")

    assert app.state.turn_silence == 0.25
    assert seen["value"] == "0.25"


def test_escape_puts_the_field_back_to_the_window_in_force(tui) -> None:
    app, received = silence_app(tui, 3.0)

    async def revert(app, pilot):
        field = app.query_one("#silence-input", Input)
        field.focus()
        field.value = "nonsense"
        await pilot.press("escape")
        await pilot.pause()

    seen = submit_silence(app, "1.5", then=revert)

    assert received == [1.5]
    assert seen["value"] == "1.5"
    assert seen["invalid"] is False
    assert seen["focused"] == "input"


def test_a_session_with_no_turn_silence_hook_refuses_the_edit(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(turn_silence=3.0), tui.TuiHooks())
    seen: dict[str, object] = {}

    async def exercise() -> None:
        async with app.run_test() as pilot:
            field = app.query_one("#silence-input", Input)
            field.focus()
            field.value = "1.5"
            await pilot.press("enter")
            await pilot.pause()
            seen["invalid"] = app.query_one("#silence-input", Input).has_class(
                "invalid"
            )

    asyncio.run(exercise())

    assert seen["invalid"] is True
    assert app.state.turn_silence == 3.0


def test_escape_outside_the_field_does_not_steal_the_key(tui) -> None:
    """Escape belongs to whatever has focus when the field does not."""
    app, _ = silence_app(tui, 3.0)
    seen: dict[str, object] = {}

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#input", Input).value = "half typed"
            app.query_one("#input", Input).focus()
            await pilot.press("escape")
            await pilot.pause()
            seen["text"] = app.query_one("#input", Input).value

    asyncio.run(exercise())

    assert seen["text"] == "half typed"


@pytest.mark.parametrize(
    ("stream_state", "speaking", "expected"),
    [
        ("idle", False, "idle"),
        ("idle", True, "speaking"),
        # The stream state is the more specific of the two, and both are true
        # at once while sentences play against an answer still arriving.
        ("replying to User Voice", True, "replying to User Voice"),
        ("replying to User Voice", False, "replying to User Voice"),
        ("running command", True, "running command"),
    ],
)
def test_codex_activity_counts_speech_as_doing_something(
    tui, stream_state, speaking, expected
) -> None:
    assert tui.codex_activity(stream_state, speaking) == expected


def test_the_sidebar_says_speaking_after_the_stream_has_ended(tui) -> None:
    """The tail of a turn: text finished, audio still playing."""
    state = tui.SessionState(codex_state="idle", codex_speaking=True)
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    assert sidebar._activity().plain == "speaking"


def test_the_sidebar_dims_only_a_genuinely_idle_codex(tui) -> None:
    idle = tui.Sidebar(tui.SessionState(), tui.TuiHooks())._activity()
    speaking = tui.Sidebar(
        tui.SessionState(codex_speaking=True), tui.TuiHooks()
    )._activity()

    assert idle.style == "#9aa3ad"
    assert speaking.style == "#6cc06c"


class FakeSpeech:
    """A speech engine a test switches between talking and quiet."""

    def __init__(self, speaking=False):
        self.value = speaking

    def is_speaking(self):
        return self.value


def test_speech_starting_is_picked_up_by_the_tick(tui) -> None:
    speech = FakeSpeech(speaking=True)
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), None, speech)

    async def exercise() -> None:
        async with app.run_test():
            app._tick_speaking()

    asyncio.run(exercise())

    assert state.codex_speaking is True


def test_speech_ending_is_picked_up_by_the_tick(tui) -> None:
    speech = FakeSpeech(speaking=True)
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), None, speech)

    async def exercise() -> None:
        async with app.run_test():
            app._tick_speaking()
            speech.value = False
            app._tick_speaking()

    asyncio.run(exercise())

    assert state.codex_speaking is False


def test_an_unchanged_speech_state_repaints_nothing(tui) -> None:
    """Ten frames a second of an unchanged word is the cost this avoids."""
    speech = FakeSpeech(speaking=True)
    state = tui.SessionState(codex_speaking=True)
    app = tui.VoiceCodexApp(state, tui.TuiHooks(), None, speech)
    painted: list[bool] = []

    async def exercise() -> None:
        async with app.run_test():
            sidebar = app.query_one("#sidebar", tui.Sidebar)
            sidebar.sync_codex = lambda: painted.append(True)
            state.codex_speaking = True
            app._tick_speaking()

    asyncio.run(exercise())

    assert painted == []


def test_a_silent_session_never_claims_to_be_speaking(tui) -> None:
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test():
            app._tick_speaking()

    asyncio.run(exercise())

    assert state.codex_speaking is False


# --------------------------------------------------------------------------
# Muting a capture channel
#
# The checkbox is the control; the hook behind it is what actually stops the
# audio being used. Each test below asserts on the hook, not on the tick mark,
# because a box that ticks without blocking anything is the failure that
# matters.
# --------------------------------------------------------------------------


def _mute_app(tui, **hooks):
    return tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks(**hooks))


def _tick(app, selector: str, value: bool, pauses: int = 3):
    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one(selector, Checkbox).value = value
            for _ in range(pauses):
                await pilot.pause()

    asyncio.run(exercise())


def test_ticking_the_mic_box_blocks_the_microphone(tui) -> None:
    muted: list[bool] = []
    app = _mute_app(tui, on_mute=muted.append)

    _tick(app, "#mic-mute", True)

    assert muted == [True]
    assert app.state.mic.muted is True
    assert app.state.them.muted is False
    assert app.entries[-1].text == "mic muted"


def test_clearing_the_mic_box_lets_the_microphone_through_again(tui) -> None:
    muted: list[bool] = []
    app = tui.VoiceCodexApp(
        tui.SessionState(mic=tui.Channel("mic", muted=True)),
        tui.TuiHooks(on_mute=muted.append),
    )

    _tick(app, "#mic-mute", False)

    assert muted == [False]
    assert app.state.mic.muted is False
    assert app.entries[-1].text == "mic live"


def test_ticking_the_speaker_box_blocks_listening_to_the_speaker(tui) -> None:
    them: list[bool] = []
    mic: list[bool] = []
    app = _mute_app(tui, on_them_mute=them.append, on_mute=mic.append)

    _tick(app, "#them-mute", True)

    assert them == [True]
    # The two channels are muted independently; one box must not move the other.
    assert mic == []
    assert app.state.them.muted is True
    assert app.state.mic.muted is False
    assert app.entries[-1].text == "Audio Stream muted"


def test_a_session_without_a_speaker_channel_still_ticks_its_box(tui) -> None:
    """No Them listener means no hook, and no crash when the box is used."""
    app = _mute_app(tui)

    _tick(app, "#them-mute", True)

    assert app.state.them.muted is True


def test_the_mic_box_follows_the_mute_key(tui) -> None:
    muted: list[bool] = []
    app = _mute_app(tui, on_mute=muted.append)
    shown: list[bool] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+k")
            for _ in range(3):
                await pilot.pause()
            shown.append(app.query_one("#mic-mute", Checkbox).value)

    asyncio.run(exercise())

    # The key and the box are one control: the box shows the mute the key
    # applied, and the display write does not ask for a second mute.
    assert shown == [True]
    assert muted == [True]


def test_a_muted_channel_reads_as_silent(tui) -> None:
    """Nothing a muted channel hears is used, so its dot stays dark."""
    state = tui.SessionState()
    state.mic.active = True
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    hot = _rendered(sidebar._channel(state.mic, "#6ba7ff")[0])
    state.mic.muted = True
    silenced = _rendered(sidebar._channel(state.mic, "#6ba7ff")[0])

    assert tui.SOUND_ON in hot
    assert tui.SOUND_ON not in silenced
    assert tui.SOUND_OFF in silenced


def test_the_channel_heading_leaves_the_mute_state_to_the_box(tui) -> None:
    """One statement of a mute, not two: the box says it, the heading does not."""
    state = tui.SessionState()
    state.mic.device = "Yeti"
    state.mic.muted = True
    sidebar = tui.Sidebar(state, tui.TuiHooks())

    heading = _rendered(sidebar._channel(state.mic, "#6ba7ff")[0])

    assert heading.split() == [tui.SOUND_OFF, "mic", "Yeti"]


def test_the_mute_box_reads_muted_once_the_channel_is(tui) -> None:
    state = tui.SessionState()
    app = tui.VoiceCodexApp(state, tui.TuiHooks())
    labels: list[str] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            labels.append(str(app.query_one("#them-mute", Checkbox).label))
            app.query_one("#them-mute", Checkbox).value = True
            for _ in range(3):
                await pilot.pause()
            labels.append(str(app.query_one("#them-mute", Checkbox).label))

    asyncio.run(exercise())

    assert labels == [tui.MUTE_LABEL, tui.MUTED_LABEL]


def test_the_live_line_names_which_channels_are_muted(tui) -> None:
    app = tui.VoiceCodexApp(tui.SessionState(), tui.TuiHooks())
    lines: list[str] = []

    async def exercise() -> None:
        async with app.run_test():
            for mic, them in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                app.state.mic.muted = mic
                app.state.them.muted = them
                app._sync_partial()
                lines.append(app.query_one("#partial", Static).content.plain)

    asyncio.run(exercise())

    assert lines == [
        "◌ silence — mic hot, nothing pending",
        "◌ mic muted — Them still transcribing",
        "◌ speaker muted — mic still hot",
        "◌ mic and speaker muted — nothing transcribing",
    ]


def test_a_ticked_mute_box_is_visible_and_an_empty_one_is_not(tui) -> None:
    """Textual distinguishes the two by colour alone, so both need styling.

    Restyling only the empty state leaves a ticked box painted in the same
    colour as its own well, and a muted channel then looks exactly like a live
    one.
    """
    state = tui.SessionState()
    state.them.muted = True
    marks: dict[bool, tuple[object, object]] = {}
    app = tui.VoiceCodexApp(state, tui.TuiHooks())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            for box in app.query(Checkbox):
                style = box.get_visual_style("toggle--button")
                marks[box.value] = (style.foreground, style.background)

    asyncio.run(exercise())

    assert set(marks) == {False, True}
    assert marks[False][0] == marks[False][1]  # the mark hides in its well
    assert marks[True][0] != marks[True][1]
