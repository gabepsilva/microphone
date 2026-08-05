#!/usr/bin/env python3
"""Fail when a catalog action has no handler and is not explicitly deferred.

The same advertised-versus-runnable drift has appeared twice: ``/help`` listing
commands the router could not run, and MCP advertising tools that answered
INAPPLICABLE. A list nobody can forget to update is the structural fix.

Every :data:`~tagalong.control.actions.CATALOG` id must either appear as a
``controller.register("…")`` string in the application adapter or sit in
:data:`DEFERRED_ACTIONS` with a reason. Wiring an action still listed as
deferred also fails — the deferred set is not a place to hide finished work.

Registration is collected by reading the source, the same way
``worker_gate`` reads thread construction: importing the adapter and standing
up stub collaborators would measure coverage of fakes rather than of the
wiring the gate exists to check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tagalong.control.actions import CATALOG

APPLICATION = Path("tagalong/application.py")

# Actions intentionally without a handler. Empty after milestone 7 wired the
# last catalog entries; a future deferral needs a one-line reason here and a
# DEVIATIONS.md entry that names the test pinning the absence.
DEFERRED_ACTIONS: frozenset[str] = frozenset()


def registered_action_ids(source: str) -> frozenset[str]:
    """Return action ids passed as the first argument of ``*.register(...)``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "register":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return frozenset(found)


def production_handler_ids(path: Path = APPLICATION) -> frozenset[str]:
    """Action ids the production application adapter registers."""
    return registered_action_ids(path.read_text(encoding="utf-8"))


def missing_handlers(
    registered: frozenset[str] | set[str],
    *,
    deferred: frozenset[str] | set[str] = DEFERRED_ACTIONS,
) -> list[str]:
    catalog_ids = {spec.id for spec in CATALOG}
    return sorted((catalog_ids - set(deferred)) - set(registered))


def stale_deferred(
    registered: frozenset[str] | set[str],
    *,
    deferred: frozenset[str] | set[str] = DEFERRED_ACTIONS,
) -> list[str]:
    return sorted(set(deferred) & set(registered))


def unknown_deferred(
    *, deferred: frozenset[str] | set[str] = DEFERRED_ACTIONS
) -> list[str]:
    catalog_ids = {spec.id for spec in CATALOG}
    return sorted(set(deferred) - catalog_ids)


def check(
    registered: frozenset[str] | set[str] | None = None,
    *,
    deferred: frozenset[str] | set[str] = DEFERRED_ACTIONS,
) -> list[str]:
    """Return human-readable problems, or an empty list when the catalog is sound."""
    wired = production_handler_ids() if registered is None else registered
    problems: list[str] = []
    for action_id in missing_handlers(wired, deferred=deferred):
        problems.append(
            f"{action_id}: catalog action has no handler and is not deferred"
        )
    for action_id in stale_deferred(wired, deferred=deferred):
        problems.append(
            f"{action_id}: deferred but a handler is registered — remove the deferral"
        )
    for action_id in unknown_deferred(deferred=deferred):
        problems.append(f"{action_id}: deferred id is not in the catalog")
    return problems


def main() -> int:
    problems = check()
    if not problems:
        print(
            f"catalog handlers: {len(CATALOG)} actions wired"
            f" ({len(DEFERRED_ACTIONS)} deferred)."
        )
        return 0
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
