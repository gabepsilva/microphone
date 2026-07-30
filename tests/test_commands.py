"""Typed-command parsing, discovery, and dispatch."""

from __future__ import annotations

import pytest

from tagalong.commands import (
    Command,
    CommandRouter,
    CommandSpec,
    command_query,
    match_commands,
)


class Display:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_a_registered_command_receives_its_whitespace_delimited_arguments() -> None:
    display = Display()
    routed: list[Command] = []
    commands = CommandRouter(display)
    commands.register("new", routed.append)

    commands.handle("/new keep these settings")

    assert routed == [Command("new", ("keep", "these", "settings"))]
    assert display.notes == []


def test_an_unknown_command_is_reported_without_calling_any_handler() -> None:
    display = Display()
    commands = CommandRouter(display)

    commands.handle("/save")

    assert display.notes == ["unknown command: /save"]


def test_an_alias_dispatches_to_the_canonical_handler() -> None:
    display = Display()
    routed: list[Command] = []
    commands = CommandRouter(display)
    commands.register("new", routed.append, aliases=("clear",))

    commands.handle("/clear")

    assert routed == [Command("new", ())]


def test_handlers_property_exposes_registered_callables() -> None:
    commands = CommandRouter(Display())
    commands.register("help", lambda _command: None)

    assert set(commands.handlers) == {"help"}


# --------------------------------------------------------------------------
# Registration guardrails
# --------------------------------------------------------------------------


def test_register_rejects_a_duplicate_name() -> None:
    commands = CommandRouter(Display())
    commands.register("new", lambda _command: None)

    with pytest.raises(ValueError, match="already registered"):
        commands.register("new", lambda _command: None)


def test_register_rejects_an_alias_that_collides_with_a_name() -> None:
    commands = CommandRouter(Display())
    commands.register("help", lambda _command: None)

    with pytest.raises(ValueError, match="alias already registered"):
        commands.register("new", lambda _command: None, aliases=("help",))


def test_register_rejects_blank_or_spaced_names() -> None:
    commands = CommandRouter(Display())

    with pytest.raises(ValueError, match="invalid command name"):
        commands.register("", lambda _command: None)
    with pytest.raises(ValueError, match="invalid command name"):
        commands.register("new session", lambda _command: None)


def test_register_rejects_blank_or_spaced_aliases() -> None:
    commands = CommandRouter(Display())

    with pytest.raises(ValueError, match="invalid command alias"):
        commands.register("new", lambda _command: None, aliases=("",))
    with pytest.raises(ValueError, match="invalid command alias"):
        commands.register("help", lambda _command: None, aliases=("a b",))


def test_register_rejects_a_self_alias_and_duplicate_aliases() -> None:
    commands = CommandRouter(Display())

    with pytest.raises(ValueError, match="alias repeats"):
        commands.register("new", lambda _command: None, aliases=("new",))
    with pytest.raises(ValueError, match="duplicate command alias"):
        commands.register("help", lambda _command: None, aliases=("?", "?"))


# --------------------------------------------------------------------------
# Palette query + matching (pure)
# --------------------------------------------------------------------------


def test_command_query_opens_only_for_a_single_slash_token() -> None:
    assert command_query("/") == ""
    assert command_query("/ne") == "ne"
    assert command_query("/NEW") == "new"
    assert command_query("") is None
    assert command_query("hello") is None
    assert command_query("/new keep") is None
    assert command_query("/new\n") is None
    assert command_query(" /new") is None


def test_match_commands_lists_registration_order_for_an_empty_query() -> None:
    specs = (
        CommandSpec("help", "List commands"),
        CommandSpec("new", "Fresh session", aliases=("clear",)),
    )

    assert match_commands(specs, "") == specs


def test_match_commands_ranks_prefix_before_substring_and_subsequence() -> None:
    # "ne": prefix of new, substring of renew (re*ne*w), subsequence of note (n…e).
    specs = (
        CommandSpec("renew", "re-new something"),
        CommandSpec("new", "fresh"),
        CommandSpec("note", "write a note"),
    )

    matched = match_commands(specs, "ne")

    assert [spec.name for spec in matched] == ["new", "renew", "note"]


def test_match_commands_matches_aliases_and_description() -> None:
    specs = (
        CommandSpec("new", "Start a fresh session", aliases=("clear",)),
        CommandSpec("help", "List available slash commands"),
    )

    assert [spec.name for spec in match_commands(specs, "cle")] == ["new"]
    assert [spec.name for spec in match_commands(specs, "slash")] == ["help"]


def test_match_commands_uses_subsequence_when_letters_are_scattered() -> None:
    specs = (CommandSpec("help", "List commands"), CommandSpec("new", "Fresh"))

    assert [spec.name for spec in match_commands(specs, "hp")] == ["help"]


def test_subsequence_treats_an_empty_query_as_a_match() -> None:
    from tagalong.commands import _is_subsequence

    assert _is_subsequence("", "help") is True
    assert _is_subsequence("hz", "help") is False


def test_router_match_and_specs_mirror_registration() -> None:
    commands = CommandRouter(Display())
    commands.register(
        "new",
        lambda _command: None,
        description="Fresh session",
        aliases=("clear",),
    )
    commands.register("help", lambda _command: None, description="List commands")

    assert [spec.name for spec in commands.specs()] == ["new", "help"]
    assert [spec.name for spec in commands.match("he")] == ["help"]
    assert commands.resolve("clear") == "new"
    assert commands.resolve("help") == "help"
    assert commands.resolve("missing") == "missing"
