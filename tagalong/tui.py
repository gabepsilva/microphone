#!/usr/bin/env python3
"""Textual TUI shell for tagalong.

This module owns no audio, transcription, or Codex logic. It renders state and
forwards user intent. Everything it displays arrives through
:class:`VoiceCodexTUI`, a thread-safe facade, and everything the user does
leaves through :class:`TuiHooks` callbacks. The runtime uses that facade as its
display boundary when started with ``tagalong.py``. The entry point calls
``tagalong.cli.run_entrypoint``, and `tests/test_entrypoints.py` holds it
to that contract.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, cast

# These must precede Textual imports. Textual needs Ctrl-C as an application
# key so it can clear typed text before closing the app. The Kitty keyboard
# protocol keeps Ctrl-Shift-C distinct for table-copy.
os.environ.pop("TEXTUAL_ALLOW_SIGNALS", None)
os.environ.pop("TEXTUAL_DISABLE_KITTY_KEY", None)

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.widget import Widget
from textual.widgets import Checkbox, Input, Link, Markdown, Select, Static, TextArea

from .attachments import (
    DEFAULT_IMAGE_CLIPBOARD,
    AttachmentStore,
    DraftAttachments,
    ImageClipboard,
)
from .commands import (
    CommandSpec,
    command_query,
    match_commands,
    preferred_index,
)
from .domain import (
    AUDIO,
    RESPONSE_POLICIES,
    TAGA,
    TEXT,
    VOICE,
    UserTextMessage,
    parse_turn_silence,
)
from .speech import (
    DEFAULT_PROVIDER,
    NO_VOICE,
    NO_VOICE_LABEL,
    PROVIDER_LABELS,
    default_voice,
)


@dataclass(frozen=True, slots=True)
class PromptPorts:
    """Injectable clipboard and store for the prompt (tests override these)."""

    clipboard: ImageClipboard = field(default_factory=lambda: DEFAULT_IMAGE_CLIPBOARD)
    store: AttachmentStore = field(default_factory=AttachmentStore)


# The picker's own name for transcribing nothing. A Select needs a value for
# every entry and None is not one, so it is spelled rather than left out. The
# word is the startup menu's, because the interface should not introduce a
# second vocabulary for the channel its meter already labels.
NO_THEM = "none"
NO_THEM_LABEL = "None"
NO_MICROPHONE = "__none__"
NO_MICROPHONE_LABEL = "None"
REPOSITORY_URL = "https://github.com/gabepsilva/microphone"

# --------------------------------------------------------------------------
# Sources and palette
#
# The speaker names and the response policies are the domain's, not the
# interface's. Restating them here would let the two drift into two
# vocabularies that compare unequal with nothing to catch it.
# --------------------------------------------------------------------------

SOURCE_STYLES = {
    VOICE: "bold #6ba7ff",  # bright blue
    TEXT: "bold #7f9bd1",  # softer blue
    AUDIO: "bold #d7b562",  # bright yellow — untrusted context
    TAGA: "bold #6cc06c",  # bright green
}
BODY_STYLE = "#cdd6e4"

POLICIES = {name: policy.sidebar_label for name, policy in RESPONSE_POLICIES.items()}

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """One rendered row in the transcript."""

    kind: str  # "speech" | "note" | "command" | "reasoning"
    source: str = ""
    text: str = ""
    stamp: str = ""
    reply_to: str = ""  # not rendered; carried for the on_save export
    interrupted: bool = False
    output: list[str] = field(default_factory=list)
    exit_code: int | None = None
    streaming: bool = False
    # How long a reasoning entry spent thinking, known only once it has.
    seconds: float | None = None


@dataclass
class Channel:
    label: str
    # Whether the channel is hearing anything right now. A bare presence flag
    # rather than a level: it only changes when speech starts or stops, and
    # the interface repaints on change, so a quiet channel costs nothing.
    active: bool = False
    muted: bool = False


@dataclass
class SessionState:
    status: str = "listening"
    live: bool = True
    policy: str = "both"
    started: datetime = field(default_factory=lambda: datetime.now(UTC))

    mic: Channel = field(default_factory=lambda: Channel("Microphone"))
    audio: Channel = field(default_factory=lambda: Channel("Audio Stream"))
    microphone: str | None = None
    microphones: list[tuple[str, str]] = field(default_factory=list)
    # The application transcribed as Audio, and the ones currently offered. The
    # list is refreshed while the session runs, because an application is only
    # in the audio graph while it is playing.
    audio_stream: str | None = None
    audio_streams: list[tuple[str, str]] = field(default_factory=list)

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

    on_user_text: Callable[[UserTextMessage], None] | None = None
    on_command: Callable[[str], None] | None = None
    # Catalog for the live slash-command palette. Pure discovery — running a
    # command still goes through ``on_command`` with the chosen ``/name``.
    list_commands: Callable[[], Sequence[CommandSpec]] | None = None
    on_policy: Callable[[str], None] | None = None
    on_codex_model: Callable[[str], bool | None] | None = None
    on_codex_effort: Callable[[str], bool | None] | None = None
    on_mute: Callable[[bool], None] | None = None
    on_microphone: Callable[[str | None], bool | None] | None = None
    on_audio_mute: Callable[[bool], None] | None = None
    on_audio_stream: Callable[[str | None], bool | None] | None = None
    on_tts: Callable[[bool], bool | None] | None = None
    on_tts_provider: Callable[[str], bool | None] | None = None
    on_turn_silence: Callable[[float], float | None] | None = None
    on_interrupt: Callable[[], None] | None = None
    on_save: Callable[[list[Entry]], None] | None = None
    on_quit: Callable[[], None] | None = None


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def render_reasoning_body(entry: Entry) -> Text:
    """Render a Taga reasoning section: the word, then the cost, then it.

    While the model is thinking the row says only that it is. The summary
    arrives in pieces and reads as a half-formed answer beside the real one,
    so it is held back until the thinking is over and can be labelled with
    how long it took.
    """
    body = Text("thinking", style="italic #7a7f96")
    if entry.streaming:
        body.append(" ▌", style="#7a7f96")
        return body
    if entry.seconds is not None:
        body.append(f" · {entry.seconds:.1f}s", style="#5a6068")
    if entry.text:
        body.append(f"\n{entry.text}", style="italic #7a7f96")
    return body


CUT_OFF_LINE = "cut off — user started speaking"


def cut_off_mark() -> Text:
    """UI chrome for a turn the user interrupted. Not part of the model text."""
    mark = Text()
    mark.append("  ⊥", style="#c96a5c")
    mark.append(f"\n{CUT_OFF_LINE}", style="#c96a5c")
    return mark


def uses_markdown_body(entry: Entry) -> bool:
    """Finished Taga answers mount clickable Textual Markdown; nothing else does.

    Streaming stays plain Text so half-open fences do not reflow under the
    cursor. Voice, Text, and Audio stay literal so transcription noise cannot
    restyle asterisks. Empty finished rows have nothing to parse.
    """
    return (
        entry.kind == "speech"
        and entry.source == TAGA
        and not entry.streaming
        and bool(entry.text)
    )


def render_entry_body(entry: Entry) -> Text:
    """Plain body for a :class:`Static` row (notes, reasoning, commands, speech).

    Finished Taga answers are not drawn through this function: the row mounts
    :class:`~textual.widgets.Markdown` instead so links can open. Call
    :func:`uses_markdown_body` before choosing a host widget.
    """
    if entry.kind == "note":
        return Text(entry.text, style="#6f757e")

    if entry.kind == "reasoning":
        return render_reasoning_body(entry)

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
    # Cut-off chrome for markdown rows is a sibling widget (see EntryRow).
    # Static-only rows carry it here so one host is enough.
    if entry.interrupted and not uses_markdown_body(entry):
        body.append_text(cut_off_mark())
    return body


SOUND_ON = "●"
SOUND_OFF = "○"


def sound_dot(active: bool, style: str = "#6cc06c") -> Text:
    """Show whether a channel is hearing anything."""
    return Text(SOUND_ON if active else SOUND_OFF, style=style if active else "#2f343b")


IDLE = "idle"
SPEAKING = "speaking"

# What a channel's mute box says about the channel it sits under.
MUTE_LABEL = "mute"
MUTED_LABEL = "muted"


def codex_activity(stream_state: str, speaking: bool) -> str:
    """Say what Taga is doing, counting speech as doing something.

    The stream state wins while there is one, because "replying to Audio" says
    more than "speaking" and both are true at once — sentences play while the
    rest of the answer is still arriving. What this fixes is the tail: the
    stream ends when the last token lands, and for several seconds after that
    Taga is still talking. That used to read as idle.
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


def empty_transcript_content() -> RenderableType:
    """Welcome pane for an empty transcript, Grok-style mark + shortcuts.

    Shown only until the first entry lands; the mark matches the sidebar logo.
    """
    # Header: brand mark beside the product line, like Grok's empty screen.
    title = Text()
    title.append("TagAlong", style="bold #cdd6e4")
    title.append("\n")
    title.append("Mic + meeting audio, with Taga in the room.", style="#6f757e")

    header = Table.grid(padding=(0, 3))
    header.add_column(no_wrap=True, vertical="middle")
    header.add_column()
    header.add_row(
        Text.assemble(("T", "bold #cdd6e4"), ("»", "bold #6cc06c")),
        title,
    )

    callout = Text()
    callout.append("Ready when you are.\n", style="bold #d7b562")
    callout.append(
        "Speak, type below, or pick a meeting stream in the sidebar.",
        style="#6f757e",
    )

    shortcuts = Table.grid(padding=(0, 6))
    shortcuts.add_column(no_wrap=True)
    shortcuts.add_column(justify="right", no_wrap=True)
    for label, key in (
        ("Cycle response policy", "^P"),
        ("Mute microphone", "^K"),
        ("Toggle voice reply", "^T"),
        ("Interrupt Taga", "^X"),
        ("Toggle sidebar", "^B"),
        ("Save transcript", "^S"),
        ("Send message", "↵"),
    ):
        shortcuts.add_row(
            Text(label, style="#cdd6e4"),
            Text(key, style="#5a6068"),
        )

    panel = Table.grid(padding=(0, 0, 1, 0))
    panel.add_column()
    panel.add_row(header)
    panel.add_row(Text(""))
    panel.add_row(callout)
    panel.add_row(Text(""))
    panel.add_row(shortcuts)
    return panel


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------


def count_visual_lines(text: str, width: int) -> int:
    """How many terminal rows ``text`` needs at ``width`` columns.

    Counts hard newlines and soft-wraps each line by display cell width so a
    long unwrapped paragraph still grows the prompt. Pure and layout-free so
    tests (and early mount, before TextArea has a wrap width) stay deterministic.
    """
    columns = max(1, width)
    if text == "":
        return 1
    rows = 0
    for line in text.split("\n"):
        if line == "":
            rows += 1
            continue
        cells = cell_len(line)
        rows += max(1, (cells + columns - 1) // columns)
    return rows


class PromptInput(TextArea):
    """Multiline chat prompt: Enter submits, Shift+Enter inserts a newline.

    Textual's :class:`Input` is single-line, so Shift cannot break a line there.
    This prompt is a compact :class:`TextArea` that keeps the same ``value``
    surface the rest of the app and tests already use.

    Image paste is owned here: the OS clipboard port, on-disk store, and draft
    token map live on the widget so it does not reach into the application.
    """

    # Tall enough for a short paragraph; short enough that the transcript still
    # owns the screen. Height tracks *visual* rows (hard breaks and soft wrap).
    MAX_LINES = 8

    BINDINGS: ClassVar[list[Binding]] = [
        # priority=True so this runs before TextArea's key handler can insert "\n".
        # Enter is handled only here — not also in _on_key.
        Binding("enter", "submit", "Submit", show=False, priority=True),
        # Replaces TextArea's paste so OS clipboard images can become tokens.
        Binding("ctrl+v", "paste_or_image", "Paste", show=False),
    ]

    class Submitted(Message):
        """Posted when Enter submits the prompt (not Shift+Enter)."""

        def __init__(self, prompt: PromptInput, message: UserTextMessage) -> None:
            self.prompt = prompt
            self.message = message
            super().__init__()

        @property
        def value(self) -> str:
            return self.message.text

        @property
        def images(self) -> tuple[str, ...]:
            return self.message.images

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        ports: PromptPorts | None = None,
    ) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            compact=True,
            highlight_cursor_line=False,
            tab_behavior="focus",
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        resolved = ports if ports is not None else PromptPorts()
        self.clipboard_port: ImageClipboard = resolved.clipboard
        self.store = resolved.store
        self.draft = DraftAttachments()

    @property
    def value(self) -> str:
        """Text content, named like :class:`Input` so call sites stay uniform."""
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.text = value
        self.fit_height()

    def on_mount(self) -> None:
        self.fit_height()

    def clear_draft(self) -> None:
        """Empty the field and drop staged attachment tokens."""
        self.value = ""
        self.draft.clear()

    def action_submit(self) -> None:
        message = UserTextMessage(
            text=self.text,
            images=self.draft.resolve(self.text),
        )
        self.post_message(self.Submitted(self, message))

    def action_paste_or_image(self) -> None:
        """Paste an OS clipboard image as ``[Image #N]``, else paste text.

        OS images are tried first: a screenshot on the system clipboard is
        almost always what Ctrl+V means after a capture tool runs. When no
        image is present, fall through to Textual's text paste (in-app copy
        and bracketed terminal paste).
        """
        if self.paste_image_from_clipboard():
            return
        self.action_paste()
        self.fit_height()

    def paste_image_from_clipboard(self) -> bool:
        """Stage a clipboard image and insert its token. Return whether one landed."""
        if self.read_only:
            return False
        image = self.clipboard_port.read_image()
        if image is None:
            return False
        try:
            path = self.store.save(image)
        except ValueError:
            return False
        token = self.draft.add(path)
        self.insert(token)
        self.fit_height()
        return True

    def insert_line_break(self) -> None:
        """Insert a newline at the cursor, replacing any active selection.

        Uses only public TextArea APIs so a Textual upgrade cannot break us by
        renaming a private keyboard helper.
        """
        start, end = self.selection
        if start != end:
            self.delete(start, end)
        self.insert("\n")
        self.fit_height()

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self.fit_height()

    def on_resize(self, _event: events.Resize) -> None:
        # Width changes reflow soft wrap; remeasure so height follows.
        self.fit_height()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            if self.read_only:
                return
            # Bare enter is a priority binding (submit). Shift+enter is not a
            # TextArea insert by default, so we add the break ourselves.
            event.stop()
            event.prevent_default()
            self.insert_line_break()
            return
        await super()._on_key(event)

    def wrap_columns(self) -> int:
        """Column budget used to estimate soft-wrapped height.

        Prefer TextArea's own wrap width once layout has assigned one. Fall
        back to the widget or parent width so we still grow before the first
        full rewrap, and to a sane default when the test harness leaves the
        field one cell wide.
        """
        if self.wrap_width > 1:
            return self.wrap_width
        if self.size.width > 1:
            return self.size.width
        parent = self.parent
        if isinstance(parent, Widget) and parent.size.width > 1:
            return max(1, parent.size.width - 2)
        return 40

    def visual_line_count(self) -> int:
        """How many terminal rows the content occupies, including soft wrap."""
        return count_visual_lines(self.text, self.wrap_columns())

    def fit_height(self) -> None:
        """Grow with visual rows, capped so the transcript keeps the floor."""
        lines = max(1, min(self.visual_line_count(), self.MAX_LINES))
        self.styles.height = lines


def palette_window(count: int, index: int, *, max_visible: int) -> tuple[int, int]:
    """Return ``[start, end)`` so ``index`` stays visible in a capped list."""
    if count <= max_visible:
        return 0, count
    half = max_visible // 2
    start = max(0, min(index - half, count - max_visible))
    return start, start + max_visible


def render_command_palette(
    items: Sequence[CommandSpec],
    index: int,
    *,
    max_visible: int = 8,
) -> Text:
    """Render palette rows without touching widget state (pure, testable)."""
    if not items:
        return Text("  no matching commands", style="#6f757e")
    start, end = palette_window(len(items), index, max_visible=max_visible)
    visible = items[start:end]
    name_width = max(len(spec.name) for spec in visible)
    lines = Text()
    for offset, spec in enumerate(visible):
        absolute = start + offset
        selected = absolute == index
        marker = "▸" if selected else " "
        style = "#cdd6e4" if selected else "#9aa3ad"
        name_style = "bold #6ba7ff" if selected else "#7f9bd1"
        lines.append(f"{marker} /", style=style)
        lines.append(f"{spec.name:<{name_width}}", style=name_style)
        if spec.description:
            lines.append(f"  {spec.description}", style="#6f757e")
        if absolute < end - 1:
            lines.append("\n")
    return lines


class CommandPalette(Static):
    """Slash-command menu that drops down from the prompt over the key hints.

    Owns selection and display only. Filtering is pure
    (:func:`tagalong.commands.match_commands`); the host supplies the catalog
    through :attr:`TuiHooks.list_commands`. Choosing a row becomes a ``/name``
    submitted through the same path as a typed command.

    Open/close of the surrounding key strip is the app's job via
    :meth:`VoiceCodexApp._consume_palette` — the widget does not reach for
    siblings.
    """

    MAX_VISIBLE = 8

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: tuple[CommandSpec, ...] = ()
        self._index = 0
        self._open = False
        self.display = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def items(self) -> tuple[CommandSpec, ...]:
        return self._items

    @property
    def index(self) -> int:
        return self._index

    def close(self) -> None:
        """Hide the menu and drop selection state."""
        self._open = False
        self._items = ()
        self._index = 0
        self.display = False
        self.update("")

    def show(
        self,
        items: Sequence[CommandSpec],
        *,
        prefer: str | None = None,
    ) -> None:
        """Open with ``items``, keeping ``prefer`` selected when still present."""
        self._items = tuple(items)
        self._index = preferred_index(self._items, prefer)
        self._open = True
        self.display = True
        self._paint()

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        if not self._items:
            return
        self._index = (self._index + delta) % len(self._items)
        self._paint()

    def selected(self) -> CommandSpec | None:
        """The highlighted command, or ``None`` when the list is empty."""
        if not self._items:
            return None
        return self._items[self._index]

    def _window(self) -> tuple[int, int]:
        return palette_window(
            len(self._items), self._index, max_visible=self.MAX_VISIBLE
        )

    def _paint(self) -> None:
        self.update(
            render_command_palette(
                self._items, self._index, max_visible=self.MAX_VISIBLE
            )
        )


class EntryRow(Vertical):
    """A selectable transcript entry that re-renders in place while streaming."""

    def __init__(self, entry: Entry) -> None:
        super().__init__()
        self.entry = entry
        if entry.kind == "command":
            self.add_class("command")

    def _content_body(self) -> Widget:
        """Model text host: Markdown when finished Taga, Static otherwise."""
        if uses_markdown_body(self.entry):
            return Markdown(
                self.entry.text,
                classes="entry-body",
                open_links=True,
            )
        return Static(render_entry_body(self.entry), classes="entry-body")

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
            # entry-main owns content + optional cut-off chrome so interrupt
            # state never has to be baked into the markdown source.
            with Vertical(classes="entry-main"):
                yield self._content_body()
                if self.entry.interrupted:
                    yield Static(cut_off_mark(), classes="entry-cutoff")

    def _sync_cutoff(self, main: Widget) -> None:
        """Keep cut-off chrome in sync without rewriting the answer text."""
        existing = list(main.query(".entry-cutoff").results(Static))
        if self.entry.interrupted:
            if existing:
                existing[0].update(cut_off_mark())
            else:
                main.mount(Static(cut_off_mark(), classes="entry-cutoff"))
            return
        for cutoff in existing:
            cutoff.remove()

    def sync(self) -> None:
        """Re-render this row in place.

        A row mounted in the same frame has not composed its children yet.
        ``compose`` renders the entry as it stands then, so there is nothing
        to update and querying for a body that does not exist would raise.

        Streaming Taga rows start as :class:`Static` and become
        :class:`~textual.widgets.Markdown` when the turn closes so links can
        open. That is a widget swap, not an in-place ``update``.
        """
        if self.entry.kind == "command":
            for body in self.query(".entry-body").results(Static):
                body.update(render_entry_body(self.entry))
            return

        mains = list(self.query(".entry-main"))
        if not mains:
            return
        main = mains[0]

        bodies = list(main.query(".entry-body"))
        if not bodies:
            return
        body = bodies[0]
        want_markdown = uses_markdown_body(self.entry)
        if want_markdown and isinstance(body, Markdown):
            body.update(self.entry.text)
        elif not want_markdown and isinstance(body, Static):
            body.update(render_entry_body(self.entry))
        else:
            cutoffs = list(main.query(".entry-cutoff"))
            body.remove()
            if cutoffs:
                main.mount(self._content_body(), before=cutoffs[0])
            else:
                main.mount(self._content_body())

        self._sync_cutoff(main)


class Transcript(VerticalScroll):
    """The transcript's scroll region, which reports scrolling by hand.

    Only a window of the history is mounted, so the app has to know when the
    view is being moved deliberately: reading back is what pages older rows
    in, and arriving at the bottom is what returns the view to the live end.

    What it listens to is the hand — the wheel, and the scrollbar's own drag
    and click messages — never the scroll position. The app moves the view
    itself constantly to follow new output, and mounting or dropping rows
    moves it again a frame later as the layout settles. Watching the position
    cannot tell any of that from a reader scrolling, and answering the app's
    own scrolling with a page-in would have the interface chasing itself.
    """

    class ScrolledBack(Message):
        """The view was moved by hand towards older entries."""

    class ScrolledForward(Message):
        """The view was moved by hand towards newer entries."""

    def on_mouse_scroll_up(self, _event: events.MouseScrollUp) -> None:
        self.post_message(self.ScrolledBack())

    def on_mouse_scroll_down(self, _event: events.MouseScrollDown) -> None:
        self.post_message(self.ScrolledForward())

    def on_scroll_up(self, _event: ScrollUp) -> None:
        self.post_message(self.ScrolledBack())

    def on_scroll_down(self, _event: ScrollDown) -> None:
        self.post_message(self.ScrolledForward())

    def on_scroll_to(self, event: ScrollTo) -> None:
        """Report the scrollbar handle being dragged, which the wheel misses."""
        if event.y is None:
            return
        moved = self.ScrolledBack if event.y < self.scroll_y else self.ScrolledForward
        self.post_message(moved())


class Sidebar(VerticalScroll):
    """Live status column with one operational response-policy picker.

    Settings have grown past a short terminal height, so the column scrolls
    rather than clipping the lower panels. A stable gutter keeps the bar in
    its own right-hand lane so it does not paint over pickers. The whole
    column can be hidden with ``Ctrl-B`` to give the transcript the width.
    """

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

    def _microphone_options(self) -> list[tuple[str, str]]:
        return [(NO_MICROPHONE_LABEL, NO_MICROPHONE), *self.state.microphones]

    def _microphone_selection(self) -> str:
        return NO_MICROPHONE if self.state.microphone is None else self.state.microphone

    def _audio_options(self) -> list[tuple[str, str]]:
        return [(NO_THEM_LABEL, NO_THEM), *self.state.audio_streams]

    def _audio_selection(self) -> str:
        return NO_THEM if self.state.audio_stream is None else self.state.audio_stream

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
            yield Static("Taga agent responds to:", id="policy-label")
            yield self._picker(
                "policy-select", self._policy_options(), self.state.policy
            )
        yield Static(id="panel-audio-head")
        # Each channel's own mute box sits under its meter, so the control and
        # the level it silences read as one thing.
        with Vertical(id="mic-row"):
            yield Static(id="panel-mic")
            yield self._picker(
                "mic-select",
                self._microphone_options(),
                self._microphone_selection(),
            )
            yield self._mute_box("mic-mute", self.state.mic)
        with Vertical(id="audio-row"):
            yield Static(id="panel-audio")
            yield self._picker(
                "audio-select", self._audio_options(), self._audio_selection()
            )
            yield self._mute_box("audio-mute", self.state.audio)
        with Horizontal(id="silence-row"):
            yield Static("Silence Turn", id="silence-label")
            yield Input(
                value=format_seconds(self.state.turn_silence),
                id="silence-input",
                compact=True,
            )
            yield Static(" sec", id="silence-unit")
        yield Static(id="panel-countdown")
        yield Static(id="panel-codex-head")
        with Vertical(id="model-row"):
            yield self._picker(
                "model-select", self.state.codex_models, self.state.codex_model
            )
        with Vertical(id="reasoning-row"):
            yield Static("reasoning effort", id="reasoning-label")
            yield self._picker(
                "reasoning-select", self._effort_options(), self.state.codex_effort
            )
        yield Static(id="panel-codex")
        yield Static(id="panel-bottom")
        with Vertical(id="speech-row"):
            yield Static("speech engine", id="speech-label")
            yield self._picker(
                "speech-select", self._speech_options(), self._speech_selection()
            )
        yield Static(id="panel-tts")
        yield Static(id="panel-session")
        yield Link(" GitHub ↗", url=REPOSITORY_URL, id="repository-link")

    def on_mount(self) -> None:
        self.sync()

    def sync(self) -> None:
        self.sync_clock()
        self.query_one("#panel-audio-head", Static).update(Group(*self._audio_head()))
        self.query_one("#panel-codex-head", Static).update(Group(*self._codex_head()))
        self.sync_audio()
        self.query_one("#panel-codex", Static).update(Group(*self._codex()))
        self.query_one("#panel-bottom", Static).update(Group(*self._tts_head()))
        self.query_one("#panel-tts", Static).update(Group(*self._bottom()))
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
            "#mic-select",
            self._microphone_options(),
            self._microphone_selection(),
        )
        self._sync_select(
            "#audio-select", self._audio_options(), self._audio_selection()
        )
        self._sync_select(
            "#speech-select", self._speech_options(), self._speech_selection()
        )
        self._sync_checkbox("#mic-mute", self.state.mic.muted)
        self._sync_checkbox("#audio-mute", self.state.audio.muted)

    def sync_audio(self) -> None:
        """Repaint both channel lines without re-laying-out the interface.

        Each line is exactly one row high whatever it says, so Textual does
        not need to arrange anything to draw it. That matters because this is
        the panel the capture threads drive: with ``layout=True`` every
        report cost a pass over every widget in the application, transcript
        included, and the transcript grows all session.
        """
        self.query_one("#panel-mic", Static).update(
            Group(*self._channel(self.state.mic, "#6ba7ff")), layout=False
        )
        self.query_one("#panel-audio", Static).update(
            Group(*self._channel(self.state.audio, "#d7b562")), layout=False
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
        """Cheap per-frame repaint — only the panel naming what Taga is doing.

        ``layout=False`` for the reason :meth:`sync_audio` gives: the panel is
        a fixed number of rows whatever it says, and asking for a layout would
        spend a pass over every widget in the application to draw it.
        """
        self.query_one("#panel-codex", Static).update(
            Group(*self._codex()), layout=False
        )

    def sync_countdown(self) -> None:
        """Cheap per-frame repaint — only the one line the countdown occupies.

        This one runs ten times a second, so it is the repaint that can least
        afford a layout pass. See :meth:`sync_audio`.
        """
        self.query_one("#panel-countdown", Static).update(
            self._countdown(), layout=False
        )

    def sync_session(self) -> None:
        """Cheap repaint — only the session counters.

        Token usage arrives repeatedly while Codex streams. Sending it
        through the full :meth:`sync` would re-offer the options of all four
        pickers mid-answer, which is far more work than redrawing five
        numbers and risks disturbing a picker the user has open.
        """
        self.query_one("#panel-session", Static).update(
            Group(*self._session()), layout=False
        )

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
            "mic-select": self._microphone_selected,
            "audio-select": self._audio_selected,
            "speech-select": self._provider_selected,
        }.get(event.select.id)
        if handler is not None:
            handler(str(event.value))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Mute or unmute the channel whose box the user just ticked.

        A box that already agrees with its channel is :meth:`_sync_checkbox`
        repainting state the app has applied, not a request to change it.
        """
        channel = {"mic-mute": self.state.mic, "audio-mute": self.state.audio}.get(
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

    def _microphone_selected(self, value: str) -> None:
        """Ask the host to open another microphone, or to close the current one."""
        if value == self._microphone_selection():
            return
        microphone = None if value == NO_MICROPHONE else value
        if self.hooks.on_microphone and self.hooks.on_microphone(microphone) is False:
            self.sync()
            return
        self.state.microphone = microphone
        self.sync()

    def _audio_selected(self, value: str) -> None:
        """Ask the host to listen to another application, or to none.

        The picker only moves if the host accepts, so a session shows the far
        end it actually has rather than the one it was asked for. Opening one
        is slow — a speech model has to load — and deliberately not waited on
        here: the name is adopted now and the channel arrives behind it.
        """
        if value == self._audio_selection():
            return
        application = None if value == NO_THEM else value
        if (
            self.hooks.on_audio_stream
            and self.hooks.on_audio_stream(application) is False
        ):
            self.sync()
            return
        self.state.audio_stream = application
        self.sync()

    def _provider_selected(self, value: str) -> None:
        """Ask the host to change how Taga answers — either engine, or silence.

        Choosing an engine while silent does both things at once: the switch
        is asked for first, so a host that refuses it leaves the session as
        silent as it was rather than audible on an engine it never got. The
        unmute that follows is the session's, not one engine's, so it holds
        for whichever engine the switch finishes with.

        "No voice reply" only stops generation; the engine stays.
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
        """Show the application mark at the top of the sidebar."""
        self.query_one("#panel-clock", Static).update(
            Text.assemble(
                ("T", "bold #cdd6e4"),
                ("»", "bold #6cc06c"),
            )
        )

    def _audio_head(self) -> list[RenderableType]:
        return [Rule(style="#23272b"), Text("AUDIO", style="#5a6068")]

    def _channel(self, channel: Channel, style: str) -> list[RenderableType]:
        """Name one capture channel and whether it hears anything.

        The icon sits on the channel's own line rather than under it, so the
        panel keeps a fixed height and can repaint without the interface
        laying itself out again.
        """
        # A muted channel reads as silent because nothing it hears is used.
        name = sound_dot(channel.active and not channel.muted, style)
        name.append(" ")
        name.append(channel.label, style=style)
        # The mute state is not restated here. The box below says it.
        return [name]

    def _codex_head(self) -> list[RenderableType]:
        return [Rule(style="#23272b"), Text("CODEX", style="#5a6068")]

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
                ]
            )
        )
        blocks.append(Text("Taga Session", style="#6f757e"))
        blocks.append(Text(state.codex_thread, style="#9aa3ad"))
        blocks.append(_kv([("state", self._activity())]))
        return blocks

    def _activity(self) -> Text:
        """Name what Taga is doing, speech included."""
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

    def _tts_head(self) -> list[RenderableType]:
        return [Text(), Rule(style="#23272b"), Text("TTS QUEUE", style="#5a6068")]

    def _bottom(self) -> list[RenderableType]:
        state = self.state
        voice = Text(
            state.tts_voice if state.tts_enabled else "—",
            style="#9aa3ad" if state.tts_enabled else "#5a6068",
        )
        blocks: list[RenderableType] = [_kv([("voice", voice)]), Text()]
        blocks.append(Rule(style="#23272b"))

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

    #transcript-area { height: 1fr; }
    #transcript {
        height: 1fr;
        padding: 1 1 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: #0f1113;
        scrollbar-color: #2f343b;
    }
    /* Sibling of the scroll view so entry mount order stays a pure window. */
    #empty-transcript {
        height: 1fr;
        content-align: center middle;
        color: #6f757e;
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
    EntryRow .entry-main {
        width: 1fr;
        height: auto;
    }
    EntryRow .entry-body { width: 1fr; height: auto; }
    EntryRow .entry-cutoff { width: 1fr; height: auto; color: #c96a5c; }
    /* Textual Markdown pads like a document; the transcript wants chat density. */
    EntryRow Markdown.entry-body {
        width: 1fr;
        height: auto;
        padding: 0;
        margin: 0;
    }
    EntryRow Markdown.entry-body > * {
        margin-top: 0;
        margin-bottom: 0;
    }
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

    /* Drops down from the prompt over the shortcut strip. Closed state is
       Widget.display=False so the keys row keeps its usual floor. */
    #command-palette {
        height: auto;
        max-height: 10;
        padding: 0 1;
        background: #141719;
        border-top: solid #2f343b;
        border-bottom: solid #2f343b;
        color: #9aa3ad;
    }

    #promptbar {
        height: auto;
        min-height: 1;
        padding: 0 1;
        align: left top;
    }
    #prompt-mark { width: 2; color: #6f757e; }
    #input {
        width: 1fr;
        height: 1;
        min-height: 1;
        max-height: 8;
        border: none;
        padding: 0;
        background: #0f1113;
        color: #cdd6e4;
    }
    #input:focus { border: none; background: #0f1113; }
    #input .text-area--cursor-line { background: #0f1113; }
    #input-hint { width: auto; color: #5a6068; }
    #silence-row { height: auto; }
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
        height: 6;
        padding: 1 1 0 1;
        color: #6f757e;
        border-top: solid #23272b;
    }

    #sidebar {
        width: 34;
        height: 1fr;
        /* Extra right padding so pickers sit clear of the scrollbar lane. */
        padding: 1 2 1 1;
        border-left: solid #23272b;
        color: #9aa3ad;
        overflow-y: auto;
        /* Reserve a right-hand lane so the bar never covers pickers/labels. */
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-background: #0f1113;
        scrollbar-color: #2f343b;
    }
    #panel-clock { margin-bottom: 1; text-align: center; }
    #sidebar Static { height: auto; }
    /* The countdown alternates between a bar and nothing at all, and is
       repainted without a layout pass ten times a second. Pinning the row it
       occupies is what makes that safe: an auto height would have to be
       re-measured to grow back, and the repaint deliberately does not ask. */
    #panel-countdown { height: 1; }
    #mic-row, #audio-row { height: auto; margin-bottom: 1; }
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
    #model-row, #reasoning-row, #speech-row { height: auto; margin-bottom: 1; }
    #policy-row { height: auto; margin-bottom: 1; }
    #reasoning-label { color: #6f757e; }
    #speech-label { color: #6f757e; }
    #policy-label { color: #6f757e; }
    #repository-link {
        height: 1;
        margin-top: 1;
        dock: bottom;
        color: #6f757e;
        text-style: none;
    }
    #repository-link:hover, #repository-link:focus {
        color: #cdd6e4;
        text-style: underline;
    }
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
        Binding("ctrl+b", "toggle_sidebar", "sidebar", priority=True),
        Binding("escape", "dismiss_overlay", "dismiss", show=False),
        # Palette navigation steals these only while the menu is open and the
        # prompt has focus; otherwise SkipAction lets TextArea / focus work.
        Binding("up", "palette_move(-1)", "palette up", show=False, priority=True),
        Binding("down", "palette_move(1)", "palette down", show=False, priority=True),
        Binding(
            "tab", "palette_complete", "complete command", show=False, priority=True
        ),
    ]

    # The countdown redraws ten times a second. A 20-cell bar over a 3s window
    # only has 150ms of resolution per cell, so anything faster repaints the
    # same picture; anything slower is visibly steppy.
    COUNTDOWN_INTERVAL_SECONDS = 0.1

    # How many transcript rows stay mounted as widgets. Textual measures every
    # widget in the application on each layout pass, and a mounted row costs
    # around 128KB, so an unbounded transcript makes every repaint anywhere in
    # the interface steadily slower and the session steadily larger. Only the
    # widgets are capped: ``self.entries`` keeps every entry, and that is what
    # the save hook exports.
    #
    # The number is small because the layout pass is what it buys back, and
    # that pass walks every widget in the application rather than the visible
    # ones: a row is five widgets, so 300 of them put ~1500 widgets under every
    # repaint and cost ~300ms a frame, against ~100ms for 80. A terminal shows
    # perhaps twenty rows, so anything beyond a few screens is paying layout
    # for history nobody is looking at — and scrolling back is what reaches
    # that history anyway.
    MAX_MOUNTED_ROWS = 80

    # Scrolling back mounts older entries a page at a time, and gives the far
    # end of the window back once the mounted run reaches its ceiling. History
    # is therefore reachable however long the session runs, at a layout cost
    # that stays bounded — the ceiling is what is spent to buy the scrollback,
    # and it only applies while the view is held back off the live end.
    #
    # A page is around two screens, so reading back moves in strides rather
    # than a screen at a time, and the ceiling stays within a small multiple of
    # the live window: a scrollback that mounts more than the live end does
    # would make reading history the slowest thing the interface does.
    SCROLLBACK_PAGE_ROWS = 40
    MAX_SCROLLBACK_ROWS = 200

    # Codex streams faster than a terminal can usefully redraw. Deltas land on
    # the entry immediately and the row is repainted on this interval instead
    # of once per token, which bounds the repaints a long answer costs without
    # changing what it finally says.
    STREAM_FLUSH_INTERVAL_SECONDS = 0.05

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
        self._reasoning_row: EntryRow | None = None
        # Rows whose entry has changed since the last repaint, oldest first.
        self._dirty: list[EntryRow] = []
        # Which slice of ``entries`` is mounted, and whether the view is
        # following the live end. While it is, ``_window_end`` is the length
        # of the record; while it is not, new entries land in the record and
        # wait there, and the window moves only where the reader takes it.
        self._window: list[EntryRow] = []
        self._window_start = 0
        self._tailing = True
        # Injectable for tests; production uses the system clipboard and the
        # TagAlong attachments cache. The prompt widget holds the live draft.
        self.prompt_ports = PromptPorts()

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("TRANSCRIPT", id="transcript-label")
                with Vertical(id="transcript-area"):
                    yield Static(
                        empty_transcript_content(),
                        id="empty-transcript",
                    )
                    yield Transcript(id="transcript")
                yield Static(id="partial")
                with Horizontal(id="promptbar"):
                    yield Static(
                        "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}",
                        id="prompt-mark",
                    )
                    yield PromptInput(id="input", ports=self.prompt_ports)
                    yield Static("Text always gets a reply", id="input-hint")
                # Expands downward over the shortcut strip while open.
                yield CommandPalette(id="command-palette")
                yield Static(id="keys")
            yield Sidebar(self.state, self.hooks, id="sidebar")

    def on_mount(self) -> None:
        self.query_one("#keys", Static).update(self._keys_text())
        self._sync_empty_transcript()
        self._sync_partial()
        self.set_interval(self.COUNTDOWN_INTERVAL_SECONDS, self._tick_countdown)
        self.set_interval(self.COUNTDOWN_INTERVAL_SECONDS, self._tick_speaking)
        self.set_interval(self.STREAM_FLUSH_INTERVAL_SECONDS, self.flush_stream)
        self.query_one("#input", PromptInput).focus()

    # -- chrome ------------------------------------------------------------

    def _keys_text(self) -> Text:
        keys = Text()
        pairs = [
            ("^P", "policy"),
            ("^K", "mute mic"),
            ("^T", "tts"),
            ("^X", "interrupt codex"),
            ("^B", "sidebar"),
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
        keys.append(" paste text/image")
        keys.append("\n")
        keys.append("↵", style="#9aa3ad")
        keys.append(" send")
        keys.append("  ")
        keys.append("⇧↵", style="#9aa3ad")
        keys.append(" newline")
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
        elif state.mic.muted and state.audio.muted:
            line = Text(
                "◌ mic and speaker muted — nothing transcribing", style="#6f757e"
            )
        elif state.mic.muted:
            line = Text("◌ mic muted — Audio still transcribing", style="#6f757e")
        elif state.audio.muted:
            line = Text("◌ speaker muted — mic still hot", style="#6f757e")
        else:
            line = Text("◌ silence — mic hot, nothing pending", style="#6f757e")
        self.query_one("#partial", Static).update(line)

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
        # Posted from a capture thread, which can arrive after the sidebar
        # has gone — during shutdown, or before it has mounted.
        with suppress(NoMatches):
            self.query_one("#sidebar", Sidebar).sync_audio()

    def refresh_session(self) -> None:
        with suppress(NoMatches):
            self.query_one("#sidebar", Sidebar).sync_session()

    # -- transcript --------------------------------------------------------

    def _stamp(self) -> str:
        return datetime.now(UTC).astimezone().strftime("%H:%M:%S")

    def add_entry(self, entry: Entry) -> EntryRow:
        entry.stamp = entry.stamp or self._stamp()
        was_empty = not self.entries
        self.entries.append(entry)
        row = EntryRow(entry)
        if not self._tailing:
            # The view is held back in history. The entry is in the record and
            # will be mounted when the view returns to the live end; mounting
            # it now would grow the content under what is being read.
            if was_empty:
                self._sync_empty_transcript()
            return row
        transcript = self.query_one("#transcript", Transcript)
        transcript.mount(row)
        self._window.append(row)
        self._trim_rows()
        transcript.scroll_end(animate=False)
        if was_empty:
            self._sync_empty_transcript()
        return row

    def clear_transcript(self) -> None:
        """Forget every rendered and saved transcript row for a new session."""
        for row in self._window:
            row.remove()
        self.entries.clear()
        self._streaming = None
        self._command_row = None
        self._reasoning_row = None
        self._dirty.clear()
        self._window.clear()
        self._window_start = 0
        self._tailing = True
        self._sync_empty_transcript()

    def _sync_empty_transcript(self) -> None:
        """Show the welcome pane only while the transcript has no entries."""
        empty_session = not self.entries
        with suppress(NoMatches):
            self.query_one("#empty-transcript", Static).display = empty_session
            self.query_one("#transcript", Transcript).display = not empty_session

    @property
    def _window_end(self) -> int:
        """One past the newest entry the window holds."""
        return self._window_start + len(self._window)

    def _held_open_by(self, row: EntryRow) -> bool:
        """Is this row still being written to?

        The open Taga message, the running command and the reasoning being
        narrated are re-rendered in place as their output arrives. Unmounting
        one would leave a stream writing to a row nobody can see.
        """
        return any(
            row is open_row
            for open_row in (self._streaming, self._command_row, self._reasoning_row)
        )

    def _trim_rows(self) -> None:
        """Unmount the oldest rows once the transcript outgrows its window.

        Only widgets are dropped. Every entry stays in ``self.entries``, saving
        still writes all of them, and scrolling back mounts them again — the
        window is where the history is shown, not where it is kept.

        A row still being written to is never unmounted, and neither is anything
        newer than it, because the mounted rows have to stay one unbroken run of
        entries for scrolling to extend them at either end. So an open Taga
        message holds the window open past its size until it closes, and the
        next trim collects what it held.
        """
        while len(self._window) > self.MAX_MOUNTED_ROWS:
            if self._held_open_by(self._window[0]):
                return
            self._window.pop(0).remove()
            self._window_start += 1

    # -- scrollback --------------------------------------------------------

    def on_transcript_scrolled_back(self, _: Transcript.ScrolledBack) -> None:
        """Stop following the live end, and page older entries in at the top."""
        self._tailing = False
        self.call_after_refresh(self._page_in_older)

    def on_transcript_scrolled_forward(self, _: Transcript.ScrolledForward) -> None:
        self.call_after_refresh(self._page_in_newer)

    async def _page_in_older(self) -> None:
        """Mount the previous page of entries once the view nears the top.

        Mounting above the view would push what is being read down the screen,
        so the row at the top is noted first and scrolled back under the eye
        afterwards. Anchoring to that row rather than to a height keeps it exact
        however tall the rows mounted above it turn out to be.
        """
        transcript = self.query_one("#transcript", Transcript)
        if self._window_start == 0 or not self._near_top(transcript):
            return
        anchor = self._top_row(transcript)
        start = max(0, self._window_start - self.SCROLLBACK_PAGE_ROWS)
        rows = [
            self._row_for(entry) for entry in self.entries[start : self._window_start]
        ]
        await transcript.mount_all(rows, before=0)
        self._window[:0] = rows
        self._window_start = start
        self.call_after_refresh(self._hold_view, transcript, anchor)

    async def _page_in_newer(self) -> None:
        """Arriving at the bottom means the live end, however far back it is.

        Not the next page down: walking forward a page at a time would take as
        many turns of the wheel as the reader spent coming up, and would leave
        them at the bottom of a window that is not the bottom of the session
        with nothing to say so.
        """
        transcript = self.query_one("#transcript", Transcript)
        if not transcript.is_vertical_scroll_end:
            return
        if self._window_end < len(self.entries):
            await self._move_window_to_live_end(transcript)
        self._tailing = True
        self._trim_rows()
        transcript.scroll_end(animate=False)

    async def _move_window_to_live_end(self, transcript: Transcript) -> None:
        """Put the window back over the newest entries, wherever it had gone.

        What overlaps the window it already had is kept mounted rather than
        mounted again, and the window is stretched back over any row still
        being written to, so nothing open is ever dropped from under a stream.
        """
        start = max(0, len(self.entries) - self.MAX_MOUNTED_ROWS)
        open_rows = [
            index for index, row in enumerate(self._window) if self._held_open_by(row)
        ]
        if open_rows:
            start = min(start, self._window_start + open_rows[0])
        stale = max(0, start - self._window_start)
        for row in self._window[:stale]:
            row.remove()
        del self._window[:stale]
        self._window_start = max(start, self._window_start)
        arrived = [self._row_for(entry) for entry in self.entries[self._window_end :]]
        self._window.extend(arrived)
        await transcript.mount_all(arrived)

    def _hold_view(self, transcript: Transcript, anchor: EntryRow | None) -> None:
        """Put the row that was at the top back at the top."""
        if anchor is not None and anchor.is_mounted:
            transcript.scroll_to_widget(anchor, top=True, animate=False)
        self._drop_newest_overflow()

    def _near_top(self, transcript: Transcript) -> bool:
        """Is the view close enough to the top to want the page above it?

        A screen ahead of the edge, so the rows are already mounted by the time
        the scrolling that asked for them arrives there.
        """
        return transcript.scroll_offset.y <= transcript.size.height

    def _top_row(self, transcript: Transcript) -> EntryRow | None:
        top = transcript.content_region.y
        for row in self._window:
            if row.region.bottom > top:
                return row
        return None

    def _row_for(self, entry: Entry) -> EntryRow:
        """The row for an entry: the one being written to, or a fresh one.

        A row created while the view was held back was never mounted, and the
        stream is still writing to it. Mounting that row rather than a second
        one for the same entry is what makes the writing appear when the entry
        finally comes into view.
        """
        for row in (self._streaming, self._command_row, self._reasoning_row):
            if row is not None and row.entry is entry:
                return row
        return EntryRow(entry)

    def _drop_newest_overflow(self) -> None:
        """Give the far end of the window back after paging older rows in.

        The rows dropped are below the view, so nothing on screen moves.
        """
        while len(self._window) > self.MAX_SCROLLBACK_ROWS:
            if self._held_open_by(self._window[-1]):
                return
            self._window.pop().remove()

    def mark_dirty(self, row: EntryRow) -> None:
        """Note that a row needs repainting at the next flush."""
        if row not in self._dirty:
            self._dirty.append(row)

    def flush_stream(self) -> None:
        """Repaint every row that has changed since the last flush.

        An idle session does no work here, which is why this can run on a
        timer at all.
        """
        if not self._dirty:
            return
        for row in self._dirty:
            row.sync()
        self._dirty.clear()
        if not self._tailing:
            # Reading history. A row growing at the live end must not drag the
            # view off what is being read.
            return
        with suppress(NoMatches):
            self.query_one("#transcript", Transcript).scroll_end(animate=False)

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
        prompt = self.query_one("#input", PromptInput)
        if prompt.value or prompt.draft:
            prompt.clear_draft()
            self._sync_command_palette("")
            return
        self.action_quit_app()

    # -- slash-command palette ---------------------------------------------
    #
    # Lifecycle is intentional: every open goes through ``_sync_command_palette``
    # and every close through ``_consume_palette``. That pair owns the key-strip
    # visibility so dismiss / submit / clear cannot leave the chrome half-open.

    def _command_catalog(self) -> tuple[CommandSpec, ...]:
        if self.hooks.list_commands is None:
            return ()
        return tuple(self.hooks.list_commands())

    def _command_palette(self) -> CommandPalette | None:
        with suppress(NoMatches):
            return self.query_one("#command-palette", CommandPalette)
        return None

    def _palette_active(self) -> bool:
        """True when palette keys should steal navigation from the prompt."""
        palette = self._command_palette()
        if palette is None or not palette.is_open:
            return False
        with suppress(NoMatches):
            return self.query_one("#input", PromptInput).has_focus
        return False

    def _set_keys_visible(self, visible: bool) -> None:
        """Show or hide the shortcut strip the palette drops over."""
        with suppress(NoMatches):
            self.query_one("#keys", Static).display = visible

    def _consume_palette(self) -> tuple[bool, CommandSpec | None]:
        """Close the menu if open; restore the key strip.

        Returns ``(was_open, selection)``. ``selection`` is only meaningful
        when ``was_open`` is true, and may still be ``None`` when the list was
        empty (no matches). That distinction is what dismiss vs submit need.
        """
        palette = self._command_palette()
        if palette is None or not palette.is_open:
            return False, None
        selected = palette.selected()
        palette.close()
        self._set_keys_visible(True)
        return True, selected

    def _sync_command_palette(self, text: str) -> None:
        """Open, filter, or close the palette from the current prompt text."""
        palette = self._command_palette()
        if palette is None:
            return
        query = command_query(text)
        catalog = self._command_catalog()
        if query is None or not catalog:
            self._consume_palette()
            return
        prefer = None
        if palette.is_open and (selected := palette.selected()) is not None:
            prefer = selected.name
        palette.show(match_commands(catalog, query), prefer=prefer)
        self._set_keys_visible(False)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "input":
            return
        self._sync_command_palette(event.text_area.text)

    def action_palette_move(self, delta: int) -> None:
        if not self._palette_active():
            raise SkipAction
        palette = self._command_palette()
        if palette is None:
            raise SkipAction
        palette.move(delta)

    def action_palette_complete(self) -> None:
        """Tab-complete the highlighted command name into the prompt."""
        if not self._palette_active():
            raise SkipAction
        palette = self._command_palette()
        if palette is None:
            raise SkipAction
        spec = palette.selected()
        if spec is None:
            raise SkipAction
        prompt = self.query_one("#input", PromptInput)
        completed = f"/{spec.name}"
        prompt.value = completed
        # Land the cursor after the name so the typist can add arguments.
        prompt.cursor_location = (0, len(completed))
        self._sync_command_palette(completed)

    def action_dismiss_overlay(self) -> None:
        """Escape: close the command palette, else revert the silence field."""
        was_open, _selected = self._consume_palette()
        if was_open:
            return
        self.action_revert_turn_silence()

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
        self.query_one("#input", PromptInput).focus()

    def action_revert_turn_silence(self) -> None:
        """Put the field back to the window in force, and leave it."""
        field = self.query_one("#silence-input", Input)
        if not field.has_focus:
            raise SkipAction
        field.remove_class("invalid")
        field.value = format_seconds(self.state.turn_silence)
        self.query_one("#input", PromptInput).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "silence-input":
            self._apply_turn_silence(event.value)

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.message.text.strip()
        images = event.message.images
        event.prompt.clear_draft()
        # Prefer the highlighted row when the menu is open so Enter on ``/ne``
        # runs ``/new`` rather than an unknown partial.
        was_open, selected = self._consume_palette()
        if was_open and selected is not None:
            text = f"/{selected.name}"
        if not text:
            return
        if text.startswith("/"):
            if self.hooks.on_command:
                self.hooks.on_command(text)
            else:
                self.add_entry(Entry(kind="note", text=f"command {text}"))
            return
        self.add_entry(Entry(kind="speech", source=TEXT, text=text))
        if self.hooks.on_user_text:
            self.hooks.on_user_text(UserTextMessage(text=text, images=images))

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
            self.hooks.on_mute
            if channel is self.state.mic
            else self.hooks.on_audio_mute
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
        """Generate spoken replies or stop; report whether the host accepted.

        Both the key binding and the sidebar's "No voice reply" arrive here,
        so the note and the picker say the same thing however the voice was
        turned off. Off means no speech is generated; the engine stays in
        place, which is why the session's host has no reason to refuse. The
        refusal is still honoured rather than assumed away: a host that says
        no leaves the interface showing the voice it actually has.
        """
        if self.hooks.on_tts and self.hooks.on_tts(enabled) is False:
            self.add_entry(Entry(kind="note", text="tts could not be changed"))
            return False
        self.state.tts_enabled = enabled
        self.refresh_sidebar()
        self.add_entry(Entry(kind="note", text=f"tts {'on' if enabled else 'off'}"))
        return True

    def action_interrupt(self) -> None:
        if self.hooks.on_interrupt:
            self.hooks.on_interrupt()
        if self._streaming is not None:
            self._streaming.entry.streaming = False
            self._streaming.entry.interrupted = True
            self.mark_dirty(self._streaming)
            self._streaming = None
        # Draw the cut-off mark, and whatever text arrived just before it,
        # without waiting for the next flush.
        self.flush_stream()
        self.state.codex_state = "idle"
        self.refresh_sidebar()

    def action_toggle_sidebar(self) -> None:
        """Show or hide the whole settings column."""
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.display = not sidebar.display
        if not sidebar.display:
            # Focus may have been on a sidebar picker; put it back on the prompt.
            self.query_one("#input", PromptInput).focus()

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
        # When the open reasoning section started, or None when none is open.
        # Only the Codex stream worker touches it, one turn at a time.
        self._thinking_started: float | None = None

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

    def _post(self, fn, *args) -> None:
        """Schedule a repaint without waiting for it to happen.

        :meth:`_call` waits on the application thread. That is fine for the
        occasional change, but the capture threads call in from inside an
        audio callback, where waiting on a repaint stalls the very capture
        the interface is drawing. ``call_later`` hands the work over and
        returns; it is safe from any thread and never blocks.
        """
        if not self._ready.is_set():
            return
        with suppress(RuntimeError):
            self.app.call_later(fn, *args)

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

    def reset_transcript(self) -> None:
        """Clear the transcript without changing the session's controls."""
        self.state.partial_source = ""
        self.state.partial_text = ""
        self._call(self.app.clear_transcript)

    # -- Codex turn --------------------------------------------------------

    def begin_codex(self) -> None:
        self._clear_partial()

    def codex_message_open(self, reply_to: str) -> None:
        self.state.codex_state = f"replying to {reply_to}"
        self._call(self._codex_begin_impl, reply_to)

    def _codex_begin_impl(self, reply_to: str) -> None:
        entry = Entry(kind="speech", source=TAGA, reply_to=reply_to, streaming=True)
        self.app._streaming = self.app.add_entry(entry)
        self.app.refresh_sidebar()

    def codex_delta(self, delta: str) -> None:
        self._call(self._codex_delta_impl, delta)

    def codex_message_close(self) -> None:
        """Keep the row open until the Codex turn itself has completed."""

    def _codex_delta_impl(self, delta: str) -> None:
        row = self.app._streaming
        if row is None:
            self._codex_begin_impl(VOICE)
            row = self.app._streaming
        if row is None:
            raise RuntimeError("Could not create a streaming Taga transcript row.")
        row.entry.text += delta
        self.app.mark_dirty(row)

    def end_codex(self) -> None:
        # A turn cut off while it was still thinking never completes its
        # reasoning item, so the open section is closed here instead. It
        # queues ahead of the end-of-turn repaint, and does nothing when the
        # item completed on its own.
        self.reasoning_completed()
        self.state.codex_state = "idle"
        self._call(self._codex_end_impl)

    def _codex_end_impl(self) -> None:
        # An interrupted turn is marked by the interrupt itself, which clears
        # the streaming row before this runs. A turn that ends normally has
        # nothing to mark.
        row = self.app._streaming
        if row is not None:
            row.entry.streaming = False
            self.app.mark_dirty(row)
            self.app._streaming = None
        # The turn is over, so whatever the flush timer has not drawn yet is
        # drawn now. Nothing else will arrive to trigger it.
        self.app.flush_stream()
        self.app.refresh_sidebar()

    def reasoning_started(self) -> None:
        self.state.codex_state = "thinking"
        # Timed from here, on the thread the notification arrived on, rather
        # than from the repaint this schedules: what is being measured is how
        # long the model thought, not when the interface got around to it.
        self._thinking_started = time.monotonic()
        self._call(self._reasoning_impl)

    def _reasoning_impl(self) -> None:
        self.app._streaming = None
        self.app._reasoning_row = self.app.add_entry(
            Entry(kind="reasoning", source=TAGA, streaming=True)
        )
        self.app.refresh_sidebar()

    def reasoning_delta(self, delta: str) -> None:
        self._call(self._reasoning_delta_impl, delta)

    def _reasoning_delta_impl(self, delta: str) -> None:
        row = self.app._reasoning_row
        if row is None:
            # Summary text without a started item: open the section here
            # rather than drop what the model said.
            self._reasoning_impl()
            row = self.app._reasoning_row
        if row is None:
            raise RuntimeError("Could not create a streaming reasoning row.")
        row.entry.text += delta
        self.app.mark_dirty(row)

    def reasoning_completed(self) -> None:
        elapsed = (
            None
            if self._thinking_started is None
            else time.monotonic() - self._thinking_started
        )
        self._thinking_started = None
        self._call(self._reasoning_end_impl, elapsed)

    def _reasoning_end_impl(self, elapsed: float | None) -> None:
        row = self.app._reasoning_row
        if row is None:
            return
        row.entry.streaming = False
        row.entry.seconds = elapsed
        self.app.mark_dirty(row)
        self.app._reasoning_row = None
        self.app.flush_stream()

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
        self.app.mark_dirty(row)

    def command_completed(self, exit_code: int | None) -> None:
        # A command the SDK reports without an exit code is still finished;
        # -1 distinguishes that from a real zero.
        self._call(self._command_exit_impl, -1 if exit_code is None else exit_code)

    def tool_called(self, server: str, tool: str) -> None:
        self.note(f"tool {server}.{tool}")

    def tool_completed(self, status: str) -> None:
        self.note(f"tool status: {status}")

    def token_usage(self, total_tokens: int) -> None:
        """Record the running token count, repainting only where it shows.

        This arrives repeatedly while Codex streams, so it deliberately does
        not go through :meth:`set_session` and its full sidebar refresh.
        """
        self.state.tokens = total_tokens
        self._call(self.app.refresh_session)

    def error(self, message: str) -> None:
        self.note(message)

    def _command_exit_impl(self, code: int) -> None:
        row = self.app._command_row
        if row is None:
            return
        row.entry.exit_code = code
        self.app.mark_dirty(row)
        self.app._command_row = None
        self.app.flush_stream()
        self.app.refresh_sidebar()

    # -- panels ------------------------------------------------------------

    def set_audio(
        self,
        channel: str,
        *,
        active: bool,
    ) -> None:
        """Say whether a capture channel hears anything.

        The sound reports come from an audio callback thread, so they are
        posted rather than waited on, and they repaint only the two channel
        lines instead of the whole sidebar.
        """
        target = {"mic": self.state.mic, "audio": self.state.audio}.get(channel)
        if target is None:
            return
        target.active = active
        self._post(self.app.refresh_audio)

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

    def set_audio_streams(self, applications: list[tuple[str, str]]) -> None:
        """Install the applications currently on offer.

        The one being listened to stays selectable whether or not it is in the
        list: an application that stops playing leaves the graph, and dropping
        it from the picker would make the session look as though it had been
        pointed somewhere else.
        """
        self.state.audio_streams = list(applications)
        self._call(self.app.refresh_sidebar)

    def set_microphones(self, microphones: list[tuple[str, str]]) -> None:
        """Install the input devices currently available for selection."""
        self.state.microphones = list(microphones)
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
