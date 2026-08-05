#!/usr/bin/env python3
"""Keep the interface, the runtime, and the adapters pointing the right way.

The package has a shape that nothing enforced. The interface knows about the
runtime; the runtime does not know about the interface. The core modules
describe what a session does and hold no idea of how audio is captured or
spoken. Everything a core module needs from either side arrives through a
protocol or a plain data class in ``presentation``.

That shape drifted once already, and quietly. ``Entry`` was a plain dataclass
that lived in ``tui.py`` because that is where it was first rendered.
``recording`` needed it, could not import it without dragging Textual into a
module that writes text files, and so took ``entry: Any`` and reached through
``getattr`` instead. Nothing failed. Coverage did not move, no test broke, and
the type checker had nothing to check — the dependency was real but spelled in
a way no tool could see. It was found by reading.

The rules below are the ones that were already true when this gate was
written, so it starts green and stays a ratchet rather than a backlog:

  * Textual belongs to ``tui`` alone. It is the one module allowed to know
    what draws the session.
  * A core module may not import ``tui``. The composition root may — it is
    the place whose whole job is knowing about both sides.
  * A core module may not import a concrete audio or speech adapter. What it
    needs from one is a protocol, not a process.
  * No import cycles anywhere in the package, whichever direction they run.

Imports written inside a function count. Deferring one delays the cost of
loading a module; it does not make the dependency any less real, and both
existing deferred imports of ``tui`` are in the composition root, where they
are allowed.

What this cannot see is the failure that started it: a module that duck-types
around a shape it refuses to import is invisible here, because there is no
import to find. The defence against that one is a shared shape in
``presentation`` and a type checker that can follow it. This gate keeps that
shape reachable, by making the wrong way to get at it fail.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path("tagalong")

# The one module allowed to know what the session looks like on screen.
INTERFACE = "tui"

# Third-party interface machinery, which belongs to the interface module only.
INTERFACE_LIBRARIES = ("textual",)

# What a session is and does, in terms no device or process appears in. These
# stay reusable, testable without hardware, and readable without the widgets.
CORE = frozenset(
    {
        "catalog",
        "codex",
        "commands",
        "config",
        "domain",
        "listener",
        "presentation",
        "recording",
        "session",
    }
)

# The modules that own a device, a process, or a network client. A core module
# reaches these through a protocol, never by name.
ADAPTERS = frozenset(
    {
        # Owns the OS clipboard alongside its pure token and storage logic.
        "attachments",
        "capture",
        "choosers",
        "piper_tts",
        "playback",
        "queued_tts",
        # Chooses between the two speech adapters and can swap them mid-session.
        "speech",
        "streams",
        "tts",
    }
)

# Wiring the two together is the whole job of these, so they may import both.
COMPOSITION = frozenset({"cli", "startup"})


def _imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    """Name every sibling module imported here, at any depth, with its line."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.append((node.module.split(".")[0], node.lineno))
            elif node.level == 1 and node.module is None:
                for alias in node.names:
                    found.append((alias.name.split(".")[0], node.lineno))
            elif node.level == 0 and node.module:
                head, _, rest = node.module.partition(".")
                if head == PACKAGE.name and rest:
                    found.append((rest.split(".")[0], node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head, _, rest = alias.name.partition(".")
                if head == PACKAGE.name and rest:
                    found.append((rest.split(".")[0], node.lineno))
    return found


def _imported_libraries(tree: ast.AST) -> list[tuple[str, int]]:
    """Name every third-party package imported here, with its line."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
    return found


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one import cycle as the path that closes it, or None."""
    visiting: list[str] = []
    done: set[str] = set()

    def walk(module: str) -> list[str] | None:
        if module in visiting:
            return [*visiting[visiting.index(module) :], module]
        if module in done:
            return None
        visiting.append(module)
        for imported in sorted(graph.get(module, ())):
            cycle = walk(imported)
            if cycle is not None:
                return cycle
        visiting.pop()
        done.add(module)
        return None

    for module in sorted(graph):
        cycle = walk(module)
        if cycle is not None:
            return cycle
    return None


def check_package(sources: list[Path]) -> list[str]:
    """Report every boundary the package's own modules cross the wrong way."""
    failures: list[str] = []
    graph: dict[str, set[str]] = {}

    for path in sources:
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        siblings = _imported_modules(tree)
        graph[module] = {name for name, _ in siblings}

        if module != INTERFACE:
            for library, line in _imported_libraries(tree):
                if library in INTERFACE_LIBRARIES:
                    failures.append(
                        f"{path}:{line}: {module} imports {library}; "
                        f"only {INTERFACE}.py may know what draws the session"
                    )

        if module in CORE:
            for name, line in siblings:
                if name == INTERFACE:
                    failures.append(
                        f"{path}:{line}: core module {module} imports "
                        f"{INTERFACE}; the interface knows the runtime, not "
                        "the other way round"
                    )
                elif name in ADAPTERS:
                    failures.append(
                        f"{path}:{line}: core module {module} imports the "
                        f"{name} adapter; take a protocol from presentation "
                        "rather than a device or a process"
                    )

    cycle = _find_cycle(graph)
    if cycle is not None:
        failures.append(f"import cycle: {' -> '.join(cycle)}")

    return failures


def main() -> int:
    sources = sorted(PACKAGE.glob("*.py"))
    if not sources:
        print(f"error: {PACKAGE} holds no modules to check.")
        return 1

    known = {path.stem for path in sources}
    unplaced = known - CORE - ADAPTERS - COMPOSITION - {INTERFACE, "__init__"}
    if unplaced:
        # A new module is a boundary decision. Making that decision is the
        # point; leaving it unmade would let the next one land unchecked.
        print(
            "error: place these modules in boundary_gate.py — core, adapter, "
            f"or composition: {', '.join(sorted(unplaced))}"
        )
        return 1

    failures = check_package(sources)
    for failure in failures:
        print(f"error: {failure}")

    if failures:
        print(f"\n{len(failures)} boundary failure(s).")
        return 1

    print(
        f"boundaries: {len(sources)} modules, "
        f"{len(CORE)} of them core, none reaching the wrong way."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
