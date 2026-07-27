#!/usr/bin/env python3
"""Textual TUI shell for voice-codex.

This module owns no audio, transcription, or Codex logic. It renders state and
forwards user intent. Everything it displays arrives through
:class:`VoiceCodexTUI`, a thread-safe facade, and everything the user does
leaves through :class:`TuiHooks` callbacks. The runtime uses that facade as its
display boundary when started with ``voice-codex.py``.

``voice-codex-tui.py`` is a compatibility alias for that same command, not a
way to open this module on its own. Both entry points call
``voice_codex.cli.run_entrypoint``, and `tests/test_entrypoints.py` holds them
to it.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, cast

# These must precede Textual imports. Textual needs Ctrl-C as an application
# key so it can clear typed text before closing the app. The Kitty keyboard
# protocol keeps Ctrl-Shift-C distinct for table-copy.
os.environ.pop("TEXTUAL_ALLOW_SIGNALS", None)
os.environ.pop("TEXTUAL_DISABLE_KITTY_KEY", None)

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Checkbox, Input, Select, Static

from .domain import (
    CODEX,
    RESPONSE_POLICIES,
    THEM,
    USER_TEXT,
    USER_VOICE,
    parse_turn_silence,
)
from .speech import (
    DEFAULT_PROVIDER,
    NO_VOICE,
    NO_VOICE_LABEL,
    PROVIDER_LABELS,
    default_voice,
)

# --------------------------------------------------------------------------
# Sources and palette
#
# The speaker names and the response policies are the domain's, not the
# interface's. Restating them here would let the two drift into two
# vocabularies that compare unequal with nothing to catch it.
# --------------------------------------------------------------------------

SOURCE_STYLES = {
    USER_VOICE: "bold #6ba7ff",  # bright blue
    USER_TEXT: "bold #7f9bd1",  # softer blue
    THEM: "bold #d7b562",  # bright yellow — untrusted context
    CODEX: "bold #6cc06c",  # bright green
}
BODY_STYLE = "#cdd6e4"

POLICIES = {name: policy.sidebar_label for name, policy in RESPONSE_POLICIES.items()}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """One rendered row in the transcript."""

    kind: str  # "speech" | "note" | "command"
    source: str = ""
    text: str = ""
    stamp: str = ""
    reply_to: str = ""  # not rendered; carried for the on_save export
    interrupted: bool = False
    output: list[str] = field(default_factory=list)
    exit_code: int | None = None
    streaming: bool = False


@dataclass
class Channel:
    label: str
    device: str = "—"
    level: float = 0.0
    muted: bool = False


@dataclass
class SessionState:
    status: str = "listening"
    live: bool = True
    policy: str = "both"
    started: datetime = field(default_factory=lambda: datetime.now(UTC))

    mic: Channel = field(default_factory=lambda: Channel("mic"))
    them: Channel = field(default_factory=lambda: Channel("speaker"))
    out_device: str = "—"

    codex_model: str = "gpt-5.6-luna"
    codex_effort: str = "low"
    codex_tier: str = "standard"
    codex_sandbox: str = "full-access"
    codex_thread: str = "—"
    codex_state: str = "idle"
    # Whether speech is still coming out of the speakers. Tracked apart from
    # codex_state because the two end at different moments: the text stream
    # finishes seconds before the audio it produced has finished playing.
    codex_speaking: bool = False
    codex_models: list[tuple[str, str]] = field(
        default_factory=lambda: [("gpt-5.6-luna", "gpt-5.6-luna")]
    )
    codex_efforts: list[str] = field(default_factory=lambda: ["low"])
    codex_efforts_by_model: dict[str, list[str]] = field(default_factory=dict)
    codex_default_effort_by_model: dict[str, str] = field(default_factory=dict)

    tts_enabled: bool = True
    tts_provider: str = DEFAULT_PROVIDER
    tts_voice: str = default_voice(DEFAULT_PROVIDER)
    tts_queue: list[str] = field(default_factory=list)

    turn_silence: float = 3.0
    # Seconds until the pending turn is submitted, or None when none is.
    turn_countdown: float | None = None
    confidence: float = 0.60
    language: str = "en"
    moonshine: str = "medium-streaming"
    tokens: int = 0
    echoes_cut: int = 0

    partial_source: str = ""
    partial_text: str = ""


@dataclass
class TuiHooks:
    """Callbacks the host script supplies. Every one is optional."""

    on_user_text: Callable[[str], None] | None = None
    on_command: Callable[[str], None] | None = None
    on_policy: Callable[[str], None] | None = None
    on_codex_model: Callable[[str], bool | None] | None = None
    on_codex_effort: Callable[[str], bool | None] | None = None
    on_mute: Callable[[bool], None] | None = None
    on_them_mute: Callable[[bool], None] | None = None
    on_tts: Callable[[bool], bool | None] | None = None
    on_tts_provider: Callable[[str], bool | None] | None = None
    on_turn_silence: Callable[[float], float | None] | None = None
    on_interrupt: Callable[[], None] | None = None
    on_save: Callable[[list[Entry]], None] | None = None
    on_quit: Callable[[], None] | None = None


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def render_entry_body(entry: Entry) -> Text:
    """Render an entry's selectable body without its timestamp and source."""
    if entry.kind == "note":
        return Text(entry.text, style="#6f757e")

    if entry.kind == "command":
        body = Text(f"$ {entry.text}", style="#9aa3ad")
        for line in entry.output:
            body.append(f"\n{line}", style="#6f757e")
        if entry.exit_code is not None:
            body.append(f"\n[command exit: {entry.exit_code}]", style="#5a6068")
        return body

    body = Text(entry.text, style=BODY_STYLE)
    if entry.streaming:
        body.append(" ▌", style="#6cc06c")
    if entry.interrupted:
        body.append("  ⊥", style="#c96a5c")
        body.append("\ncut off — user started speaking", style="#c96a5c")
    return body


def meter(level: float, width: int = 20, style: str = "#6cc06c") -> Text:
    filled = max(0, min(width, round(level * width)))
    bar = Text()
    bar.append("■" * filled, style=style)
    bar.append("□" * (width - filled), style="#2f343b")
    return bar


IDLE = "idle"
SPEAKING = "speaking"

# What a channel's mute box says about the channel it sits under.
MUTE_LABEL = "mute"
MUTED_LABEL = "muted"


def codex_activity(stream_state: str, speaking: bool) -> str:
    """Say what Codex is doing, counting speech as doing something.

    The stream state wins while there is one, because "replying to Them" says
    more than "speaking" and both are true at once — sentences play while the
    rest of the answer is still arriving. What this fixes is the tail: the
    stream ends when the last token lands, and for several seconds after that
    Codex is still talking. That used to read as idle.
    """
    if stream_state != IDLE:
        return stream_state
    return SPEAKING if speaking else IDLE


def format_seconds(seconds: float) -> str:
    """Render a turn-silence window the way the field accepts it back."""
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def countdown_bar(remaining: float, window: float, width: int = 10) -> Text:
    """Draw the silence a turn has left to wait, draining as it runs out.

    Colour is the message: the wait is green until it is nearly over, then
    amber, because the last moments are when the turn is about to be sent and
    speaking again would still stop it.
    """
    left = 0.0 if window <= 0 else max(0.0, min(1.0, remaining / window))
    filled = max(0, min(width, round(left * width)))
    bar = Text()
    bar.append("■" * filled, style="#6cc06c" if left > 0.25 else "#d7b562")
    bar.append("□" * (width - filled), style="#2f343b")
    bar.append(f" {remaining:.1f}s", style="#9aa3ad")
    return bar


def options_including(
    options: list[tuple[str, str]], current: str
) -> list[tuple[str, str]]:
    """Guarantee the active value is selectable.

    ``Select`` raises ``InvalidSelectValueError`` for a value it does not list,
    so a host-configured model or effort that predates the discovered catalog
    has to be offered as an option of its own.
    """
    if any(value == current for _, value in options):
        return list(options)
    return [(current, current), *options]


def _kv(rows: list[tuple[RenderableType, RenderableType]]) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", style="#6f757e", no_wrap=True)
    grid.add_column(justify="right", ratio=1, overflow="ellipsis")
    for key, value in rows:
        grid.add_row(key, value)
    return grid


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------


class EntryRow(Vertical):
    """A selectable transcript entry that re-renders in place while streaming."""

    def __init__(self, entry: Entry) -> None:
        super().__init__()
        self.entry = entry
        if entry.kind == "command":
            self.add_class("command")

    def compose(self) -> ComposeResult:
        if self.entry.kind == "command":
            yield Static(render_entry_body(self.entry), classes="entry-body")
            return
        with Horizontal(classes="transcript-entry"):
            yield Static(self.entry.stamp, classes="entry-stamp")
            source = "·" if self.entry.kind == "note" else self.entry.source
            yield Static(
                Text(source, style=SOURCE_STYLES.get(source, "#5a6068")),
                classes="entry-source",
            )
            yield Static(render_entry_body(self.entry), classes="entry-body")

    def sync(self) -> None:
        """Re-render this row in place.

        A row mounted in the same frame has not composed its children yet.
        ``compose`` renders the entry as it stands then, so there is nothing
        to update and querying for a body that does not exist would raise.
        """
        for body in self.query(".entry-body").results(Static):
            body.update(render_entry_body(self.entry))


class Sidebar(Vertical):
    """Live status column with one operational response-policy picker."""

    def __init__(self, state: SessionState, hooks: TuiHooks, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.hooks = hooks
        # What each picker currently lists, and the Select.Changed messages our
        # own writes have posted and must therefore ignore on the way back.
        self._listed: dict[str, list[tuple[str, str]]] = {}
        self._echoes: dict[str, deque[str]] = {}

    def _policy_options(self) -> list[tuple[str, str]]:
        return [(POLICIES[key], key) for key in POLICIES]

    def _effort_options(self) -> list[tuple[str, str]]:
        return [(effort, effort) for effort in self.state.codex_efforts]

    def _speech_options(self) -> list[tuple[str, str]]:
        options = [(label, name) for name, label in PROVIDER_LABELS.items()]
        options.append((NO_VOICE_LABEL, NO_VOICE))
        return options

    def _speech_selection(self) -> str:
        """Name what the speech picker is showing.

        A silenced session shows silence rather than the engine it would use
        if it were speaking. The engine is still remembered underneath, which
        is what the picker comes back to when the voice is turned on again.
        """
        return self.state.tts_provider if self.state.tts_enabled else NO_VOICE

    def _mute_box(self, widget_id: str, channel: Channel) -> Checkbox:
        return Checkbox(
            MUTED_LABEL if channel.muted else MUTE_LABEL,
            value=channel.muted,
            id=widget_id,
            compact=True,
        )

    def _picker(
        self, widget_id: str, options: list[tuple[str, str]], current: str
    ) -> Select:
        options = options_including(options, current)
        self._listed[f"#{widget_id}"] = options
        return Select(
            options,
            value=current,
            allow_blank=False,
            compact=True,
            id=widget_id,
        )

    def compose(self) -> ComposeResult:
        yield Static(id="panel-clock")
        with Vertical(id="policy-row"):
            yield Static("AI agent responds to:", id="policy-label")
            yield self._picker(
                "policy-select", self._policy_options(), self.state.policy
            )
        yield Static(id="panel-audio")
        # Each channel's own mute box sits under its meter, so the control and
        # the level it silences read as one thing.
        with Vertical(id="mic-row"):
            yield Static(id="panel-mic")
            yield self._mute_box("mic-mute", self.state.mic)
        with Vertical(id="them-row"):
            yield Static(id="panel-them")
            yield self._mute_box("them-mute", self.state.them)
        yield Static(id="panel-out")
        with Vertical(id="model-row"):
            yield Static("AI model", id="model-label")
            yield self._picker(
                "model-select", self.state.codex_models, self.state.codex_model
            )
        with Vertical(id="reasoning-row"):
            yield Static("reasoning effort", id="reasoning-label")
            yield self._picker(
                "reasoning-select", self._effort_options(), self.state.codex_effort
            )
        yield Static(id="panel-codex")
        with Vertical(id="speech-row"):
            yield Static("speech engine", id="speech-label")
            yield self._picker(
                "speech-select", self._speech_options(), self._speech_selection()
            )
        yield Static(id="panel-bottom")
        with Horizontal(id="silence-row"):
            yield Static("turn silence", id="silence-label")
            yield Input(
                value=format_seconds(self.state.turn_silence),
                id="silence-input",
                compact=True,
            )
            yield Static("s", id="silence-unit")
        yield Static(id="panel-countdown")
        yield Static(id="panel-session")

    def on_mount(self) -> None:
        self.sync()

    def sync(self) -> None:
        self.sync_clock()
        self.query_one("#panel-audio", Static).update(Group(*self._audio_head()))
        self.query_one("#panel-out", Static).update(Group(*self._audio_foot()))
        self.sync_audio()
        self.query_one("#panel-codex", Static).update(Group(*self._codex()))
        self.query_one("#panel-bottom", Static).update(Group(*self._bottom()))
        self.query_one("#panel-session", Static).update(Group(*self._session()))
        self.sync_countdown()
        self._sync_select("#policy-select", self._policy_options(), self.state.policy)
        self._sync_select(
            "#model-select", self.state.codex_models, self.state.codex_model
        )
        self._sync_select(
            "#reasoning-select", self._effort_options(), self.state.codex_effort
        )
        self._sync_select(
            "#speech-select", self._speech_options(), self._speech_selection()
        )
        self._sync_checkbox("#mic-mute", self.state.mic.muted)
        self._sync_checkbox("#them-mute", self.state.them.muted)

    def sync_audio(self) -> None:
        self.query_one("#panel-mic", Static).update(
            Group(*self._channel(self.state.mic, "#6ba7ff"))
        )
        self.query_one("#panel-them", Static).update(
            Group(*self._channel(self.state.them, "#d7b562"))
        )

    def _sync_checkbox(self, selector: str, muted: bool) -> None:
        """Show the mute state a channel is actually in.

        The box carries the whole message — an offer to ``mute`` while the
        channel is live, and a red ``muted`` once it is not.

        The ``Checkbox.Changed`` this posts is answered by a handler that
        compares the box against the state it came from, so a write made here
        cannot loop back as a fresh mute request.
        """
        box = self.query_one(selector, Checkbox)
        box.value = muted
        box.label = MUTED_LABEL if muted else MUTE_LABEL

    def sync_codex(self) -> None:
        """Cheap per-frame repaint — only the panel naming what Codex is doing."""
        self.query_one("#panel-codex", Static).update(Group(*self._codex()))

    def sync_countdown(self) -> None:
        """Cheap per-frame repaint — only the one line the countdown occupies."""
        self.query_one("#panel-countdown", Static).update(self._countdown())

    def _sync_select(
        self,
        selector: str,
        options: list[tuple[str, str]],
        current: str,
    ) -> None:
        """Push host-driven changes into a picker without re-firing its hook.

        Every write below posts a ``Select.Changed`` that arrives back here
        asynchronously, so each one is recorded as an echo to swallow. Without
        that the pickers answer their own messages and drive each other in a
        loop, queueing model and effort switches nobody asked for.
        """
        select = self.query_one(selector, Select)
        options = options_including(options, current)
        if self._listed.get(selector) != options:
            self._listed[selector] = options
            reset = options[0][1]  # set_options resets the selection to it
            if select.value != reset:
                self._echo(selector, reset)
            select.set_options(options)
        # Land on `current` with exactly one Changed either way: a plain write
        # when the value moved, a forced watcher run when only its label did —
        # otherwise the closed picker keeps showing the label it replaced.
        self._echo(selector, current)
        if select.value == current:
            select.mutate_reactive(Select.value)
        else:
            select.value = current

    def _echo(self, selector: str, value: str) -> None:
        """Record a Select.Changed one of our own writes is about to post."""
        self._echoes.setdefault(selector, deque()).append(value)

    def _is_echo(self, event: Select.Changed) -> bool:
        """Report whether this change is one :meth:`_sync_select` just made."""
        echoes = self._echoes.get(f"#{event.select.id}")
        if not echoes:
            return False
        if echoes.popleft() == event.value:
            return True
        # Our bookkeeping drifted — drop it and trust the widget.
        echoes.clear()
        return False

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._is_echo(event) or event.value is Select.NULL:
            return
        handler = {
            "policy-select": self._policy_selected,
            "model-select": self._model_selected,
            "reasoning-select": self._effort_selected,
            "speech-select": self._provider_selected,
        }.get(event.select.id)
        if handler is not None:
            handler(str(event.value))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Mute or unmute the channel whose box the user just ticked.

        A box that already agrees with its channel is :meth:`_sync_checkbox`
        repainting state the app has applied, not a request to change it.
        """
        channel = {"mic-mute": self.state.mic, "them-mute": self.state.them}.get(
            event.checkbox.id or ""
        )
        if channel is None or channel.muted == event.value:
            return
        cast("VoiceCodexApp", self.app).set_channel_muted(channel, bool(event.value))

    def _policy_selected(self, value: str) -> None:
        if value == self.state.policy:
            return
        self.state.policy = value
        if self.hooks.on_policy:
            self.hooks.on_policy(value)

    def _provider_selected(self, value: str) -> None:
        """Ask the host to change how Codex answers — either engine, or silence.

        The picker only moves if the host accepts, so a session started
        without speech shows the silence it is actually in rather than an
        engine it never built. Choosing an engine while silent does both
        things at once: the engine is switched first, because a session that
        cannot switch should not be turned audible on the wrong one.
        """
        if value == self._speech_selection():
            return
        app = cast("VoiceCodexApp", self.app)
        if value == NO_VOICE:
            app.set_tts_enabled(False)
            self.sync()
            return
        if value != self.state.tts_provider and not (
            self.hooks.on_tts_provider and self.hooks.on_tts_provider(value)
        ):
            self.sync()
            return
        self.state.tts_provider = value
        self.state.tts_voice = default_voice(value)
        if not self.state.tts_enabled:
            app.set_tts_enabled(True)
        self.sync()

    def _adopt_efforts_for(self, model: str) -> None:
        """Offer the efforts the newly selected model supports."""
        available_efforts = self.state.codex_efforts_by_model.get(model, [])
        if not available_efforts:
            return
        self.state.codex_efforts = available_efforts
        if self.state.codex_effort not in available_efforts:
            self.state.codex_effort = (
                self.state.codex_default_effort_by_model.get(model)
                or available_efforts[0]
            )

    def _model_selected(self, value: str) -> None:
        if value == self.state.codex_model:
            return
        previous = (
            self.state.codex_model,
            self.state.codex_effort,
            self.state.codex_efforts,
        )
        if self.hooks.on_codex_model and self.hooks.on_codex_model(value) is False:
            self.sync()
            return
        self.state.codex_model = value
        self._adopt_efforts_for(value)
        if (
            self.state.codex_effort != previous[1]
            and self.hooks.on_codex_effort
            and self.hooks.on_codex_effort(self.state.codex_effort) is False
        ):
            # The whole switch is off, effort list included.
            (
                self.state.codex_model,
                self.state.codex_effort,
                self.state.codex_efforts,
            ) = previous
        self.sync()

    def _effort_selected(self, value: str) -> None:
        if value == self.state.codex_effort:
            return
        if self.hooks.on_codex_effort and self.hooks.on_codex_effort(value) is False:
            self.sync()
            return
        self.state.codex_effort = value
        self.sync()

    def sync_clock(self) -> None:
        """Cheap per-tick repaint — only the panel holding the session clock."""
        state = self.state
        elapsed = int((datetime.now(UTC) - state.started).total_seconds())
        clock = f"{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
        live = Text("◉ ", style="#6cc06c" if state.live else "#c96a5c")
        live.append(state.status, style="#9aa3ad")
        self.query_one("#panel-clock", Static).update(
            _kv([(live, Text(clock, style="#6f757e"))])
        )

    def _audio_head(self) -> list[RenderableType]:
        return [Text(), Text("AUDIO", style="#5a6068")]

    def _channel(self, channel: Channel, style: str) -> list[RenderableType]:
        """Name one capture channel, its device, and how loud it currently is."""
        head = Table.grid(expand=True)
        head.add_column(justify="left", no_wrap=True)
        head.add_column(justify="right", ratio=1, overflow="ellipsis")
        # The mute state is not restated here. The box below says it.
        head.add_row(
            Text(channel.label, style=style), Text(channel.device, style="#9aa3ad")
        )
        # A muted channel reads as silent because nothing it hears is used.
        return [head, meter(0.0 if channel.muted else channel.level, style=style)]

    def _audio_foot(self) -> list[RenderableType]:
        return [
            _kv([("out", Text(self.state.out_device, style="#9aa3ad"))]),
            Text(),
            Text("CODEX", style="#5a6068"),
        ]

    def _codex(self) -> list[RenderableType]:
        state = self.state
        blocks: list[RenderableType] = []

        sandbox = Text(state.codex_sandbox, style="#9aa3ad")
        if state.codex_sandbox == "full-access":
            sandbox = Text(f"{state.codex_sandbox} ⚠", style="#d7b562")
        blocks.append(
            _kv(
                [
                    (
                        "effort",
                        Text(
                            f"{state.codex_effort} · {state.codex_tier}",
                            style="#9aa3ad",
                        ),
                    ),
                    ("sandbox", sandbox),
                    ("thread", Text(state.codex_thread, style="#9aa3ad")),
                    ("state", self._activity()),
                ]
            )
        )
        return blocks

    def _activity(self) -> Text:
        """Name what Codex is doing, speech included."""
        activity = codex_activity(self.state.codex_state, self.state.codex_speaking)
        return Text(
            activity,
            style="#9aa3ad" if activity == IDLE else "#6cc06c",
        )

    def _countdown(self) -> Text:
        """Show the silence still to wait, or nothing when none is running."""
        state = self.state
        if state.turn_countdown is None:
            return Text()
        return countdown_bar(state.turn_countdown, state.turn_silence)

    def _bottom(self) -> list[RenderableType]:
        state = self.state
        blocks: list[RenderableType] = [Text()]

        head = Table.grid(expand=True)
        head.add_column(justify="left", no_wrap=True)
        head.add_column(justify="right", ratio=1)
        head.add_row(
            Text("TTS QUEUE", style="#5a6068"),
            Text(
                PROVIDER_LABELS[state.tts_provider] if state.tts_enabled else "off",
                style="#5a6068",
            ),
        )
        blocks.append(head)
        blocks.append(
            Text(
                state.tts_voice if state.tts_enabled else "—",
                style="#9aa3ad" if state.tts_enabled else "#5a6068",
            )
        )
        if state.tts_queue:
            for line in state.tts_queue[:3]:
                blocks.append(
                    Text(
                        f"\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} {line}",
                        style="#6f757e",
                        overflow="ellipsis",
                    )
                )
        else:
            blocks.append(Text("— empty —", style="#5a6068"))
        blocks.append(Text())

        blocks.append(Text("SESSION", style="#5a6068"))
        return blocks

    def _session(self) -> list[RenderableType]:
        state = self.state
        return [
            _kv(
                [
                    ("confidence", Text(f"{state.confidence:.2f}", style="#9aa3ad")),
                    ("language", Text(state.language, style="#9aa3ad")),
                    ("moonshine", Text(state.moonshine, style="#9aa3ad")),
                    ("tokens", Text(f"{state.tokens:,}", style="#9aa3ad")),
                    ("echoes cut", Text(str(state.echoes_cut), style="#9aa3ad")),
                ]
            )
        ]


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


class VoiceCodexApp(App):
    ALLOW_SELECT = True

    CSS = """
    Screen { background: #0f1113; color: #cdd6e4; }

    #body { height: 1fr; }

    #left { width: 1fr; }
    #transcript-label {
        height: 1;
        padding: 0 1;
        color: #5a6068;
        text-align: center;
    }

    #transcript {
        height: 1fr;
        padding: 1 1 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: #0f1113;
        scrollbar-color: #2f343b;
    }
    EntryRow { height: auto; margin-bottom: 1; }
    EntryRow > .transcript-entry { height: auto; }
    EntryRow .entry-stamp {
        width: 9;
        color: #5a6068;
        padding-right: 1;
    }
    EntryRow .entry-source {
        width: 11;
        padding-right: 1;
    }
    EntryRow .entry-body { width: 1fr; height: auto; }
    EntryRow.command {
        margin: 0 0 1 20;
        padding-left: 1;
        border-left: solid #2f343b;
    }

    #partial {
        height: auto;
        min-height: 2;
        padding: 0 1;
        color: #6f757e;
        border-top: solid #23272b;
    }

    #promptbar { height: 1; padding: 0 1; }
    #prompt-mark { width: 2; color: #6f757e; }
    #input {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
        background: #0f1113;
        color: #cdd6e4;
    }
    #input:focus { border: none; background: #0f1113; }
    #input-hint { width: auto; color: #5a6068; }
    #silence-row { height: auto; margin-bottom: 1; }
    #silence-label { width: 13; color: #6f757e; }
    #silence-input {
        width: 8;
        border: none;
        padding: 0;
        background: #14171a;
        color: #cdd6e4;
    }
    #silence-input:focus { background: #0f1113; }
    #silence-input.invalid { color: #c96a5c; background: #2a1a1a; }
    #silence-unit { width: auto; color: #6f757e; }

    #keys {
        height: 5;
        padding: 1 1 0 1;
        color: #6f757e;
        border-top: solid #23272b;
    }

    #sidebar {
        width: 34;
        padding: 1;
        border-left: solid #23272b;
        color: #9aa3ad;
    }
    #sidebar Static { height: auto; }
    #mic-row, #them-row { height: auto; margin-bottom: 1; }
    #sidebar Checkbox {
        height: 1;
        width: auto;
        border: none;
        padding: 0;
        margin: 0;
        background: #0f1113;
        color: #6f757e;
    }
    #sidebar Checkbox:focus { color: #cdd6e4; background: #0f1113; }
    /* An empty box hides its mark in the well; a ticked one shows it in the
       same red the channel's "muted" label uses. Both states have to be
       styled: overriding only one would make the two look identical. */
    #sidebar Checkbox > .toggle--button {
        background: #2f343b;
        color: #2f343b;
    }
    #sidebar Checkbox.-on > .toggle--button {
        background: #2f343b;
        color: #c96a5c;
    }
    #sidebar Checkbox.-on { color: #c96a5c; }
    #panel-clock { margin-bottom: 1; }
    #model-row, #reasoning-row, #speech-row { height: auto; margin-bottom: 1; }
    #policy-row { height: auto; margin-bottom: 1; }
    #model-label { width: 6; color: #6f757e; }
    #reasoning-label { color: #6f757e; }
    #speech-label { color: #6f757e; }
    #policy-label { color: #6f757e; }
    #sidebar Select {
        width: 1fr;
        margin: 0;
        background: #0f1113;
        color: #9aa3ad;
    }
    #sidebar Select > SelectCurrent {
        border: none;
        padding: 0;
        background: #0f1113;
    }
    #sidebar Select:focus > SelectCurrent { color: #cdd6e4; }
    #sidebar SelectOverlay {
        border: solid #2f343b;
        background: #141719;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit_app", "quit", priority=True),
        Binding("ctrl+c", "clear_input_or_quit", "clear or quit", priority=True),
        Binding(
            "ctrl+shift+c",
            "copy_selected_transcript",
            "copy",
            priority=True,
        ),
        Binding("ctrl+p", "cycle_policy", "policy", priority=True),
        Binding("ctrl+k", "toggle_mute", "mute mic", priority=True),
        Binding("ctrl+t", "toggle_tts", "tts", priority=True),
        Binding("ctrl+x", "interrupt", "interrupt codex", priority=True),
        Binding("ctrl+s", "save", "save transcript", priority=True),
        Binding("escape", "revert_turn_silence", "revert turn silence", show=False),
    ]

    # The countdown redraws ten times a second. A 20-cell bar over a 3s window
    # only has 150ms of resolution per cell, so anything faster repaints the
    # same picture; anything slower is visibly steppy.
    COUNTDOWN_INTERVAL_SECONDS = 0.1

    def __init__(
        self, state: SessionState, hooks: TuiHooks, countdown=None, speech=None
    ) -> None:
        super().__init__()
        self.state = state
        self.hooks = hooks
        self.countdown = countdown
        # The session's speech, polled for whether it is still talking. Absent
        # when the session was started silent.
        self.speech = speech
        self.entries: list[Entry] = []
        self._streaming: EntryRow | None = None
        self._command_row: EntryRow | None = None

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("TRANSCRIPT", id="transcript-label")
                yield VerticalScroll(id="transcript")
                yield Static(id="partial")
                with Horizontal(id="promptbar"):
                    yield Static(
                        "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}",
                        id="prompt-mark",
                    )
                    yield Input(id="input")
                    yield Static("User Text always gets a reply", id="input-hint")
                yield Static(id="keys")
            yield Sidebar(self.state, self.hooks, id="sidebar")

    def on_mount(self) -> None:
        self.query_one("#keys", Static).update(self._keys_text())
        self._sync_partial()
        self.set_interval(0.25, self._tick)
        self.set_interval(self.COUNTDOWN_INTERVAL_SECONDS, self._tick_countdown)
        self.set_interval(self.COUNTDOWN_INTERVAL_SECONDS, self._tick_speaking)
        self.query_one("#input", Input).focus()

    # -- chrome ------------------------------------------------------------

    def _keys_text(self) -> Text:
        keys = Text()
        pairs = [
            ("^P", "policy"),
            ("^K", "mute mic"),
            ("^T", "tts"),
            ("^X", "interrupt codex"),
            ("/", "commands"),
        ]
        for index, (key, label) in enumerate(pairs):
            if index:
                keys.append("  ")
            keys.append(key, style="#9aa3ad")
            keys.append(f" {label}")
        keys.append("\n")
        keys.append("^S", style="#9aa3ad")
        keys.append(" save transcript")
        keys.append("  ")
        keys.append("^Q", style="#9aa3ad")
        keys.append(" quit")
        keys.append("\n")
        keys.append("^⇧C", style="#9aa3ad")
        keys.append(" copy")
        keys.append("  ")
        keys.append("^V", style="#9aa3ad")
        keys.append(" paste")
        return keys

    def _sync_partial(self) -> None:
        state = self.state
        if state.partial_text:
            line = Text("◌ ", style="#5a6068")
            line.append(
                f"{state.partial_source}  ",
                style=SOURCE_STYLES.get(state.partial_source, "#6f757e"),
            )
            line.append(state.partial_text, style="#8a929c")
        elif state.mic.muted and state.them.muted:
            line = Text(
                "◌ mic and speaker muted — nothing transcribing", style="#6f757e"
            )
        elif state.mic.muted:
            line = Text("◌ mic muted — Them still transcribing", style="#6f757e")
        elif state.them.muted:
            line = Text("◌ speaker muted — mic still hot", style="#6f757e")
        else:
            line = Text("◌ silence — mic hot, nothing pending", style="#6f757e")
        self.query_one("#partial", Static).update(line)

    def _tick(self) -> None:
        # The clock and the live indicator sit in the sidebar's top panel.
        with suppress(NoMatches):
            self.query_one("#sidebar", Sidebar).sync_clock()

    def _tick_countdown(self) -> None:
        """Redraw the silence countdown, and only while there is one.

        An idle session does no work at all here. The one repaint after the
        countdown ends is what restores the configured window in its place,
        so the check is against the state as well as the clock.
        """
        remaining = None if self.countdown is None else self.countdown.remaining()
        if remaining is None and self.state.turn_countdown is None:
            return
        self.state.turn_countdown = remaining
        with suppress(NoMatches):
            self.query_one("#sidebar", Sidebar).sync_countdown()

    def _tick_speaking(self) -> None:
        """Repaint only when speech starts or stops, not on every frame."""
        speaking = self.speech is not None and self.speech.is_speaking()
        if speaking == self.state.codex_speaking:
            return
        self.state.codex_speaking = speaking
        with suppress(NoMatches):
            self.query_one("#sidebar", Sidebar).sync_codex()

    def refresh_sidebar(self) -> None:
        self.query_one("#sidebar", Sidebar).sync()

    def refresh_audio(self) -> None:
        self.query_one("#sidebar", Sidebar).sync_audio()

    # -- transcript --------------------------------------------------------

    def _stamp(self) -> str:
        return datetime.now(UTC).astimezone().strftime("%H:%M:%S")

    def add_entry(self, entry: Entry) -> EntryRow:
        entry.stamp = entry.stamp or self._stamp()
        self.entries.append(entry)
        row = EntryRow(entry)
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(row)
        transcript.scroll_end(animate=False)
        return row

    def _selected_transcript_entries(self) -> list[Entry]:
        selections = self.screen.selections
        return [
            row.entry
            for row in self.query(EntryRow)
            if row in selections or any(child in selections for child in row.query("*"))
        ]

    def action_copy_selected_transcript(self) -> None:
        """Copy selected transcript rows as timestamp, author, and content columns."""
        entries = self._selected_transcript_entries()
        if not entries:
            raise SkipAction()
        rows = [
            "\t".join((entry.stamp, entry.source or "System", entry.text))
            for entry in entries
        ]
        self.copy_to_clipboard("\n".join(rows))

    # -- actions -----------------------------------------------------------

    def action_clear_input_or_quit(self) -> None:
        input_widget = self.query_one("#input", Input)
        if input_widget.value:
            input_widget.value = ""
            return
        self.action_quit_app()

    def _apply_turn_silence(self, text: str) -> None:
        """Adopt a typed window, or mark the field as holding a bad value.

        A rejected value is left in place rather than overwritten. The typist
        is mid-correction, and replacing their text with the old number would
        discard the keystrokes that were nearly right.
        """
        field = self.query_one("#silence-input", Input)
        seconds = parse_turn_silence(text)
        applied = (
            None
            if seconds is None or self.hooks.on_turn_silence is None
            else self.hooks.on_turn_silence(seconds)
        )
        if applied is None:
            field.add_class("invalid")
            return
        field.remove_class("invalid")
        self.state.turn_silence = applied
        field.value = format_seconds(applied)
        self.refresh_sidebar()
        self.query_one("#input", Input).focus()

    def action_revert_turn_silence(self) -> None:
        """Put the field back to the window in force, and leave it."""
        field = self.query_one("#silence-input", Input)
        if not field.has_focus:
            raise SkipAction
        field.remove_class("invalid")
        field.value = format_seconds(self.state.turn_silence)
        self.query_one("#input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "silence-input":
            self._apply_turn_silence(event.value)
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.add_entry(Entry(kind="note", text=f"command {text}"))
            if self.hooks.on_command:
                self.hooks.on_command(text)
            return
        self.add_entry(Entry(kind="speech", source=USER_TEXT, text=text))
        if self.hooks.on_user_text:
            self.hooks.on_user_text(text)

    def action_cycle_policy(self) -> None:
        order = list(POLICIES)
        self.state.policy = order[(order.index(self.state.policy) + 1) % len(order)]
        self.refresh_sidebar()
        self.add_entry(
            Entry(kind="note", text=f"response policy → {POLICIES[self.state.policy]}")
        )
        if self.hooks.on_policy:
            self.hooks.on_policy(self.state.policy)

    def action_toggle_mute(self) -> None:
        self.set_channel_muted(self.state.mic, not self.state.mic.muted)

    def set_channel_muted(self, channel: Channel, muted: bool) -> None:
        """Stop or resume listening on one capture channel.

        The hook is what actually blocks the audio: the mic hook drops what
        the microphone hears, the speaker hook drops what the sink monitor
        hears. Everything else here is the interface agreeing with it.
        """
        channel.muted = muted
        hook = (
            self.hooks.on_mute if channel is self.state.mic else self.hooks.on_them_mute
        )
        self.refresh_sidebar()
        self._sync_partial()
        self.add_entry(
            Entry(
                kind="note",
                text=f"{channel.label} {'muted' if muted else 'live'}",
            )
        )
        if hook:
            hook(muted)

    def action_toggle_tts(self) -> None:
        self.set_tts_enabled(not self.state.tts_enabled)

    def set_tts_enabled(self, enabled: bool) -> bool:
        """Speak Codex responses or stop; report whether the session could.

        Both the key binding and the sidebar's "No voice reply" arrive here,
        so the queue, the note, and the picker say the same thing however the
        voice was turned off.
        """
        if self.hooks.on_tts and self.hooks.on_tts(enabled) is False:
            self.add_entry(Entry(kind="note", text="tts unavailable for this session"))
            return False
        self.state.tts_enabled = enabled
        if not enabled:
            self.state.tts_queue.clear()
        self.refresh_sidebar()
        self.add_entry(Entry(kind="note", text=f"tts {'on' if enabled else 'off'}"))
        return True

    def action_interrupt(self) -> None:
        if self.hooks.on_interrupt:
            self.hooks.on_interrupt()
        if self._streaming is not None:
            self._streaming.entry.streaming = False
            self._streaming.entry.interrupted = True
            self._streaming.sync()
            self._streaming = None
        self.state.codex_state = "idle"
        self.state.tts_queue.clear()
        self.refresh_sidebar()

    def action_save(self) -> None:
        if self.hooks.on_save:
            self.hooks.on_save(list(self.entries))
        self.add_entry(
            Entry(kind="note", text=f"saved transcript · {len(self.entries)} entries")
        )

    def action_quit_app(self) -> None:
        if self.hooks.on_quit:
            self.hooks.on_quit()
        self.exit()


# --------------------------------------------------------------------------
# The pluggable facade
# --------------------------------------------------------------------------


class VoiceCodexTUI:
    """Thread-safe control surface over :class:`VoiceCodexApp`.

    Every method may be called from any thread — audio callbacks, the Codex
    stream worker, the TTS pipeline. Calls made before :meth:`run` starts the
    app are applied to the state object and appear once the UI mounts.
    """

    def __init__(
        self,
        state: SessionState | None = None,
        countdown=None,
        speech=None,
        **hooks,
    ) -> None:
        self.state = state or SessionState()
        self.hooks = TuiHooks(**hooks)
        self.app = VoiceCodexApp(self.state, self.hooks, countdown, speech)
        self._ready = threading.Event()
        self._app_thread: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        self._app_thread = threading.get_ident()
        self.app.call_later(self._ready.set)
        self.app.run()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._call(self.app.exit)

    def _call(self, fn, *args) -> None:
        if not self._ready.is_set():
            return
        if threading.get_ident() == self._app_thread:
            fn(*args)
        else:
            with suppress(RuntimeError):
                self.app.call_from_thread(fn, *args)

    # -- transcript --------------------------------------------------------

    def _show_partial(self, source: str, text: str) -> None:
        """Show revisable partial text for ``source`` on the live line."""
        self.state.partial_source = source
        self.state.partial_text = text
        self._call(self.app._sync_partial)

    def _clear_partial(self) -> None:
        self._show_partial("", "")

    # The following methods implement the runtime's TranscriptPresentation
    # boundary. Keeping them here means the runtime never imports Textual
    # widgets directly.

    def update(self, speaker: str, text: str) -> None:
        self._show_partial(speaker, text)

    def finish_turn(self, speaker: str) -> None:
        if self.state.partial_source == speaker:
            self._clear_partial()

    def close_speaker(self, speaker: str) -> None:
        self.finish_turn(speaker)

    def commit(self, speaker: str, text: str) -> None:
        """Append a finished turn and clear the live line."""
        self.state.partial_source = ""
        self.state.partial_text = ""
        self._call(self._commit_impl, speaker, text)

    def _commit_impl(self, speaker: str, text: str) -> None:
        self.app.add_entry(Entry(kind="speech", source=speaker, text=text))
        self.app._sync_partial()

    def note(self, text: str) -> None:
        """Append a dim system line (policy changes, echo suppression, …)."""
        self._call(lambda: self.app.add_entry(Entry(kind="note", text=text)))

    # -- Codex turn --------------------------------------------------------

    def begin_codex(self) -> None:
        self._clear_partial()

    def codex_message_open(self, reply_to: str) -> None:
        self.state.codex_state = f"replying to {reply_to}"
        self._call(self._codex_begin_impl, reply_to)

    def _codex_begin_impl(self, reply_to: str) -> None:
        entry = Entry(kind="speech", source=CODEX, reply_to=reply_to, streaming=True)
        self.app._streaming = self.app.add_entry(entry)
        self.app.refresh_sidebar()

    def codex_delta(self, delta: str) -> None:
        self._call(self._codex_delta_impl, delta)

    def codex_message_close(self) -> None:
        """Keep the row open until the Codex turn itself has completed."""

    def _codex_delta_impl(self, delta: str) -> None:
        row = self.app._streaming
        if row is None:
            self._codex_begin_impl(USER_VOICE)
            row = self.app._streaming
        if row is None:
            raise RuntimeError("Could not create a streaming Codex transcript row.")
        row.entry.text += delta
        row.sync()
        self.app.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def end_codex(self) -> None:
        self.state.codex_state = "idle"
        self._call(self._codex_end_impl)

    def _codex_end_impl(self) -> None:
        # An interrupted turn is marked by the interrupt itself, which clears
        # the streaming row before this runs. A turn that ends normally has
        # nothing to mark.
        row = self.app._streaming
        if row is not None:
            row.entry.streaming = False
            row.sync()
            self.app._streaming = None
        self.app.refresh_sidebar()

    def command_started(self, command: str) -> None:
        self.state.codex_state = "running command"
        self._call(self._command_impl, command)

    def _command_impl(self, command: str) -> None:
        self.app._streaming = None
        self.app._command_row = self.app.add_entry(Entry(kind="command", text=command))
        self.app.refresh_sidebar()

    def command_output(self, delta: str) -> None:
        self._call(self._command_output_impl, delta)

    def _command_output_impl(self, line: str) -> None:
        row = self.app._command_row
        if row is None:
            return
        row.entry.output.append(line)
        row.sync()
        self.app.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def command_completed(self, exit_code: int | None) -> None:
        # A command the SDK reports without an exit code is still finished;
        # -1 distinguishes that from a real zero.
        self._call(self._command_exit_impl, -1 if exit_code is None else exit_code)

    def tool_called(self, server: str, tool: str) -> None:
        self.note(f"tool {server}.{tool}")

    def tool_completed(self, status: str) -> None:
        self.note(f"tool status: {status}")

    def token_usage(self, total_tokens: int) -> None:
        self.set_session(tokens=total_tokens)

    def error(self, message: str) -> None:
        self.note(message)

    def _command_exit_impl(self, code: int) -> None:
        row = self.app._command_row
        if row is None:
            return
        row.entry.exit_code = code
        row.sync()
        self.app._command_row = None
        self.app.refresh_sidebar()

    # -- panels ------------------------------------------------------------

    def set_audio(
        self,
        channel: str,
        device: str | None = None,
        level: float | None = None,
    ) -> None:
        target = {"mic": self.state.mic, "them": self.state.them}.get(channel)
        if target is None:
            return
        if device is not None:
            target.device = device
        if level is not None:
            target.level = max(0.0, min(1.0, level))
        if device is None and level is not None:
            self._call(self.app.refresh_audio)
        else:
            self._call(self.app.refresh_sidebar)

    def set_output(self, device: str) -> None:
        self.state.out_device = device
        self._call(self.app.refresh_sidebar)

    def set_codex(self, **fields) -> None:
        """Update any of model/effort/tier/sandbox/thread/state."""
        for key, value in fields.items():
            attribute = f"codex_{key}"
            if hasattr(self.state, attribute):
                setattr(self.state, attribute, value)
        self._call(self.app.refresh_sidebar)

    def set_codex_catalog(
        self,
        models: list[tuple[str, str]],
        efforts_by_model: dict[str, list[str]],
        default_effort_by_model: dict[str, str],
    ) -> None:
        """Install asynchronously-discovered Codex model and effort choices.

        A configured model the catalog omits stays selectable: the pickers keep
        their active value on offer themselves.
        """
        self.state.codex_models = models
        self.state.codex_efforts_by_model = efforts_by_model
        self.state.codex_default_effort_by_model = default_effort_by_model
        efforts = efforts_by_model.get(self.state.codex_model, [])
        if efforts:
            self.state.codex_efforts = efforts
            if self.state.codex_effort not in efforts:
                self.state.codex_effort = (
                    default_effort_by_model.get(self.state.codex_model) or efforts[0]
                )
        self._call(self.app.refresh_sidebar)

    def set_tts_queue(self, sentences: list[str]) -> None:
        self.state.tts_queue = list(sentences)
        self._call(self.app.refresh_sidebar)

    def set_session(self, **fields) -> None:
        """Update language/turn_silence/confidence/moonshine/tokens/echoes_cut."""
        for key, value in fields.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self._call(self.app.refresh_sidebar)

    def set_status(self, status: str, live: bool = True) -> None:
        self.state.status = status
        self.state.live = live
        self._call(self.app.refresh_sidebar)
