"""Parse, discover, and dispatch typed slash commands.

The router is the session's command catalog: registration is the only place a
handler is attached to a name. Filtering for the live palette is pure so the
TUI never learns what a query *means* — it only renders what :func:`match_commands`
returns and submits the chosen ``/name`` back through :meth:`CommandRouter.handle`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One slash command, split into its name and whitespace-delimited arguments."""

    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    """A discoverable slash command: what the palette shows, not how it runs."""

    name: str
    description: str
    aliases: tuple[str, ...] = ()


def command_query(text: str) -> str | None:
    """Return the filter fragment when the command palette should be open.

    The menu is open only for a single-token slash query: leading ``/``, no
    newline, and no whitespace after the slash. Once the typist has started
    arguments (``/new keep``), the menu closes so free-form args are unobstructed.
    The returned string is the case-folded name fragment without the slash.
    """
    if not text.startswith("/"):
        return None
    if "\n" in text or "\r" in text:
        return None
    rest = text[1:]
    if any(character.isspace() for character in rest):
        return None
    return rest.casefold()


def _is_subsequence(query: str, name: str) -> bool:
    """True when every character of ``query`` appears in order inside ``name``."""
    if not query:
        return True
    position = 0
    for character in name:
        if character == query[position]:
            position += 1
            if position == len(query):
                return True
    return False


def _name_score(name: str, query: str) -> tuple[int, int, int] | None:
    """Rank how well ``name`` matches ``query``. Lower tuples sort first.

    Tiers:
      0 — prefix of the name (``ne`` → ``new``)
      1 — contiguous substring
      2 — subsequence (``nw`` → ``new``)
    Within a tier, earlier occurrence and shorter names win so ``new`` beats
    ``newer`` for the same prefix.
    """
    if name.startswith(query):
        return (0, 0, len(name))
    index = name.find(query)
    if index >= 0:
        return (1, index, len(name))
    if _is_subsequence(query, name):
        return (2, 0, len(name))
    return None


def _spec_score(spec: CommandSpec, query: str) -> tuple[int, int, int, int] | None:
    """Best score across the command's name and aliases, then description."""
    best: tuple[int, int, int] | None = None
    for candidate in (spec.name, *spec.aliases):
        score = _name_score(candidate.casefold(), query)
        if score is not None and (best is None or score < best):
            best = score
    if best is not None:
        # Fourth key: name length for stable ordering among equal scores.
        return (*best, len(spec.name))
    if query and query in spec.description.casefold():
        return (3, 0, len(spec.name), len(spec.name))
    return None


def match_commands(specs: Sequence[CommandSpec], query: str) -> tuple[CommandSpec, ...]:
    """Filter and rank command specs for a palette query.

    An empty query lists every command in registration order. A non-empty query
    returns only matches, ordered by match quality then name length.
    """
    if not query:
        return tuple(specs)
    ranked: list[tuple[tuple[int, int, int, int], CommandSpec]] = []
    for spec in specs:
        score = _spec_score(spec, query)
        if score is not None:
            ranked.append((score, spec))
    ranked.sort(key=lambda item: item[0])
    return tuple(spec for _, spec in ranked)


class CommandRouter:
    """Route registered slash commands and expose them for discovery."""

    def __init__(self, display) -> None:
        self.display = display
        self._handlers: dict[str, Callable[[Command], None]] = {}
        self._specs: list[CommandSpec] = []
        self._aliases: dict[str, str] = {}

    # Back-compat for call sites that still touch ``handlers`` directly.
    @property
    def handlers(self) -> dict[str, Callable[[Command], None]]:
        return self._handlers

    def register(
        self,
        name: str,
        handler: Callable[[Command], None],
        *,
        description: str = "",
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Make ``/name`` (and any aliases) available for this session."""
        if not name or any(character.isspace() for character in name):
            raise ValueError(f"invalid command name: {name!r}")
        if name in self._handlers or name in self._aliases:
            raise ValueError(f"command already registered: {name}")
        normalized_aliases: list[str] = []
        for alias in aliases:
            if not alias or any(character.isspace() for character in alias):
                raise ValueError(f"invalid command alias: {alias!r}")
            if alias == name:
                raise ValueError(f"alias repeats command name: {alias}")
            if alias in self._handlers or alias in self._aliases:
                raise ValueError(f"command alias already registered: {alias}")
            if alias in normalized_aliases:
                raise ValueError(f"duplicate command alias: {alias}")
            normalized_aliases.append(alias)
        self._handlers[name] = handler
        self._specs.append(
            CommandSpec(
                name=name,
                description=description,
                aliases=tuple(normalized_aliases),
            )
        )
        for alias in normalized_aliases:
            self._aliases[alias] = name

    def specs(self) -> tuple[CommandSpec, ...]:
        """Every registered command, in registration order."""
        return tuple(self._specs)

    def resolve(self, token: str) -> str:
        """Map a typed name or alias to the canonical command name."""
        if token in self._handlers:
            return token
        return self._aliases.get(token, token)

    def match(self, query: str) -> tuple[CommandSpec, ...]:
        """Filter this session's catalog for the palette."""
        return match_commands(self._specs, query)

    def handle(self, text: str) -> None:
        """Parse a slash command and pass it to its handler, if one is registered."""
        words = text.removeprefix("/").split()
        raw_name = words[0] if words else ""
        name = self.resolve(raw_name)
        handler = self._handlers.get(name)
        if handler is None:
            self.display.note(f"unknown command: {text}")
            return
        handler(Command(name, tuple(words[1:])))
