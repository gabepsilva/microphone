"""Typed-command parsing and dispatch."""

from __future__ import annotations

from tagalong.commands import Command, CommandRouter


class Display:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


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
