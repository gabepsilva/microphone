#!/usr/bin/env python3
"""Fail when a catalog action has no production handler and is not deferred.

The same advertised-versus-runnable drift has appeared twice: ``/help`` listing
commands the router could not run, and MCP advertising tools that answered
INAPPLICABLE. A list nobody can forget to update is the structural fix.

Every :data:`~tagalong.control.actions.CATALOG` id must either appear as a
``controller.register("…")`` string in ``tagalong/application.py`` or sit in
:data:`DEFERRED_ACTIONS` with a reason. Wiring an action still listed as
deferred also fails — the deferred set is not a place to hide finished work.

Registration alone is not enough: milestone 3 shipped a composition root that
quietly skipped a binder while the binders themselves stayed complete. The
gate therefore also requires ``tagalong/cli.py`` to call every name in
:data:`REQUIRED_BINDERS`.

Both checks read source (AST), the same way ``worker_gate`` reads thread
construction: importing the adapter and standing up stub collaborators would
measure coverage of fakes rather than of the wiring the gate exists to check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tagalong.control.actions import CATALOG

APPLICATION = Path("tagalong/application.py")
COMPOSITION_ROOT = Path("tagalong/cli.py")

# Actions intentionally without a handler. Empty after milestone 7 wired the
# last catalog entries; a future deferral needs a one-line reason here and a
# DEVIATIONS.md entry that names the test pinning the absence.
DEFERRED_ACTIONS: frozenset[str] = frozenset()

# Binder entry points the production composition root must invoke. Adding a
# fifth binder means adding its name here so a skipped call fails the gate.
REQUIRED_BINDERS: frozenset[str] = frozenset(
    {
        "bind_first_slice",
        "bind_audio_slice",
        "bind_settings_slice",
        "bind_session_transcript_slice",
    }
)


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


def called_names(source: str) -> frozenset[str]:
    """Return bare and attribute names used as call targets in *source*."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.add(func.id)
        elif isinstance(func, ast.Attribute):
            found.add(func.attr)
    return frozenset(found)


def production_handler_ids(path: Path = APPLICATION) -> frozenset[str]:
    """Action ids the production application adapter registers."""
    return registered_action_ids(path.read_text(encoding="utf-8"))


def composition_binder_ids(path: Path = COMPOSITION_ROOT) -> frozenset[str]:
    """Binder names the production composition root calls."""
    return called_names(path.read_text(encoding="utf-8"))


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


def missing_binders(
    called: frozenset[str] | set[str],
    *,
    required: frozenset[str] | set[str] = REQUIRED_BINDERS,
) -> list[str]:
    return sorted(set(required) - set(called))


def check(
    registered: frozenset[str] | set[str] | None = None,
    *,
    deferred: frozenset[str] | set[str] = DEFERRED_ACTIONS,
    binders: frozenset[str] | set[str] | None = None,
    required_binders: frozenset[str] | set[str] = REQUIRED_BINDERS,
) -> list[str]:
    """Return human-readable problems, or an empty list when the catalog is sound."""
    wired = production_handler_ids() if registered is None else registered
    called = composition_binder_ids() if binders is None else binders
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
    for name in missing_binders(called, required=required_binders):
        problems.append(f"{name}: composition root does not call this binder")
    return problems


def main() -> int:
    problems = check()
    if not problems:
        print(
            f"catalog handlers: {len(CATALOG)} actions wired"
            f" ({len(DEFERRED_ACTIONS)} deferred);"
            f" {len(REQUIRED_BINDERS)} binders called from composition root."
        )
        return 0
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
