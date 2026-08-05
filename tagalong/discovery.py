"""Human slash syntax over the typed action catalog.

Slash commands are not actions. Typing ``/new`` resolves to ``session.new``;
typing ``/help`` renders :func:`list_commands`. An agent that wants the same
information calls :func:`list_commands` directly — there is no external
``/help`` and no ``command.invoke``.

The summaries for adapters that target a catalog action come from that
action, so the palette, ``/help``, and a generated tool schema cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .commands import CommandSpec
from .control.actions import CATALOG, ActionSpec


@dataclass(frozen=True)
class SlashAdapter:
    """One human-typed name and the catalog action it runs, if any."""

    name: str
    aliases: tuple[str, ...] = ()
    action_id: str | None = None
    summary: str | None = None


# ``/help`` has no typed equivalent: it is the human renderer of this module.
SLASH_ADAPTERS: tuple[SlashAdapter, ...] = (
    SlashAdapter("new", aliases=("clear",), action_id="session.new"),
    SlashAdapter(
        "help",
        aliases=("?",),
        summary="List available slash commands",
    ),
)


@dataclass(frozen=True)
class ListedCommand:
    """One row of ``commands.list``: human name, copy, and target action."""

    name: str
    summary: str
    aliases: tuple[str, ...] = ()
    action_id: str | None = None

    def matches(self, token: str) -> bool:
        """True when *token* is this command's name or one of its aliases."""
        return token == self.name or token in self.aliases

    def spec(self) -> CommandSpec:
        """Palette / help row for this listing entry."""
        return CommandSpec(self.name, self.summary, self.aliases)


def _action(action_id: str, catalog: Sequence[ActionSpec]) -> ActionSpec:
    for spec in catalog:
        if spec.id == action_id:
            return spec
    raise KeyError(f"no such action: {action_id}")


def list_commands(
    adapters: Sequence[SlashAdapter] = SLASH_ADAPTERS,
    catalog: Sequence[ActionSpec] = CATALOG,
) -> tuple[ListedCommand, ...]:
    """Return structured discovery for slash adapters — ``commands.list``.

    This is a query, not a mutation and not an action. ``/help`` renders the
    result; an agent reads it as data.
    """
    listing: list[ListedCommand] = []
    for adapter in adapters:
        if adapter.action_id is None:
            summary = adapter.summary or ""
        else:
            summary = _action(adapter.action_id, catalog).summary
        listing.append(
            ListedCommand(
                name=adapter.name,
                summary=summary,
                aliases=adapter.aliases,
                action_id=adapter.action_id,
            )
        )
    return tuple(listing)


def render_command_help(
    listing: Sequence[ListedCommand], topic: str | None = None
) -> str:
    """Format ``commands.list`` as the text ``/help`` shows.

    An unknown topic is a message, not an exception: the typist asked a
    question and gets an answer they can correct from.
    """
    if topic is None:
        lines = ["commands:", *(entry.spec().listing_line() for entry in listing)]
        return "\n".join(lines)
    token = topic.removeprefix("/")
    for entry in listing:
        if entry.matches(token):
            return entry.spec().detail_line()
    return f"unknown command: /{token}"
