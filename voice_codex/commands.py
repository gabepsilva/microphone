"""Parse and dispatch typed slash commands without teaching the TUI their meaning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One slash command, split into its name and whitespace-delimited arguments."""

    name: str
    arguments: tuple[str, ...]


class CommandRouter:
    """Route registered slash commands and report commands this session lacks."""

    def __init__(self, display) -> None:
        self.display = display
        self.handlers: dict[str, Callable[[Command], None]] = {}

    def register(self, name: str, handler: Callable[[Command], None]) -> None:
        """Make ``/name`` available for the remainder of this running session."""
        self.handlers[name] = handler

    def handle(self, text: str) -> None:
        """Parse a slash command and pass it to its handler, if one is registered."""
        words = text.removeprefix("/").split()
        command = Command(words[0] if words else "", tuple(words[1:]))
        handler = self.handlers.get(command.name)
        if handler is None:
            self.display.note(f"unknown command: {text}")
            return
        handler(command)
