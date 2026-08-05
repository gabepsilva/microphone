"""Slash adapters and commands.list — discovery without a TUI."""

from __future__ import annotations

import pytest

from tagalong.control.actions import CATALOG, ActionSpec, Scope
from tagalong.discovery import (
    SLASH_ADAPTERS,
    ListedCommand,
    SlashAdapter,
    list_commands,
    render_command_help,
)


def test_commands_list_is_structured_data_not_help_text() -> None:
    listing = list_commands()

    assert [(entry.name, entry.action_id, entry.aliases) for entry in listing] == [
        ("new", "session.new", ("clear",)),
        ("help", None, ("?",)),
    ]


def test_a_slash_adapter_takes_its_summary_from_the_catalog_action() -> None:
    """Palette copy and /help cannot drift from the action an agent calls."""
    session_new = next(spec for spec in CATALOG if spec.id == "session.new")
    listing = {entry.name: entry for entry in list_commands()}

    assert listing["new"].summary == session_new.summary
    assert listing["new"].summary == "Start a fresh session and clear the transcript"
    assert listing["help"].summary == "List available slash commands"
    assert listing["help"].action_id is None


def test_slash_commands_are_not_catalog_actions() -> None:
    assert "command.invoke" not in {spec.id for spec in CATALOG}
    assert "commands.list" not in {spec.id for spec in CATALOG}
    assert {adapter.name for adapter in SLASH_ADAPTERS} == {"new", "help"}


def test_render_help_formats_the_listing_without_a_display() -> None:
    listing = (
        ListedCommand(
            "new", "Fresh session", aliases=("clear",), action_id="session.new"
        ),
        ListedCommand("help", "List commands", aliases=("?",)),
    )

    assert render_command_help(listing) == (
        "commands:\n  /new (/clear): Fresh session\n  /help (/?): List commands"
    )
    assert (
        render_command_help(listing, "clear") == "/new (aliases: /clear): Fresh session"
    )
    assert render_command_help(listing, "/help") == "/help (aliases: /?): List commands"
    assert render_command_help(listing, "missing") == "unknown command: /missing"


def test_list_commands_refuses_an_adapter_aimed_at_a_missing_action() -> None:
    adapters = (SlashAdapter("boom", action_id="session.explode"),)

    with pytest.raises(KeyError, match=r"session\.explode"):
        list_commands(adapters)


def test_list_commands_uses_the_catalog_it_is_given() -> None:
    catalog = (ActionSpec("session.new", "Reset everything", Scope.SESSION),)
    adapters = (SlashAdapter("new", action_id="session.new"),)

    assert list_commands(adapters, catalog)[0].summary == "Reset everything"
