#!/usr/bin/env python3
"""Fail when a catalog action has no handler and is not explicitly deferred.

The same advertised-versus-runnable drift has appeared twice: ``/help`` listing
commands the router could not run, and MCP advertising tools that answered
INAPPLICABLE. A list nobody can forget to update is the structural fix.

Every :data:`~tagalong.control.actions.CATALOG` id must either be registered by
a fully-wired production session or appear in :data:`DEFERRED_ACTIONS` with a
reason. Wiring an action still listed as deferred also fails — the deferred
set is not a place to hide finished work.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tagalong.application import (
    bind_audio_slice,
    bind_first_slice,
    bind_session_transcript_slice,
    bind_settings_slice,
)
from tagalong.control import Controller
from tagalong.control.actions import CATALOG

# Actions intentionally without a handler. Empty after milestone 7 wired the
# last catalog entries; a future deferral needs a one-line reason here and a
# DEVIATIONS.md entry that names the test pinning the absence.
DEFERRED_ACTIONS: frozenset[str] = frozenset()


class _Talk:
    generation = 0

    def ingest(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def interrupt(self) -> None:
        return None

    def start_fresh_thread(self) -> object | None:
        return None

    def adopt_fresh_thread(self, started: object) -> None:
        del started

    def request_model(self, model: str) -> bool:
        del model
        return True

    def request_reasoning_effort(self, effort: str) -> bool:
        del effort
        return True


class _Speech:
    def set_enabled(self, enabled: bool) -> bool:
        del enabled
        return True

    def set_provider(self, provider: str) -> bool:
        del provider
        return True


class _Policy:
    def set_policy(self, policy: str) -> None:
        del policy


class _Silence:
    def set(self, seconds: float) -> float:
        return seconds


class _Capture:
    def select(self, name, *, on_applied=None, on_failed=None) -> bool:
        del name, on_applied, on_failed
        return True

    def set_muted(self, muted: bool) -> None:
        del muted


class _Attachments:
    def upload(self, data: bytes) -> str:
        del data
        return "id"

    def resolve(self, ids) -> tuple:
        del ids
        return ()


class _Turns:
    def end_turn(self) -> None:
        return None


class _Rows:
    def transcript_entries(self) -> list:
        return []


def production_handler_ids() -> frozenset[str]:
    """Action ids a fully-wired production session registers."""
    controller = Controller()
    talk = _Talk()
    speech = _Speech()
    bind_first_slice(
        controller, conversation=talk, tts=speech, attachments=_Attachments()
    )
    bind_settings_slice(controller, (talk, speech, _Policy(), _Silence()))
    bind_audio_slice(controller, microphone=_Capture(), audio=_Capture())
    bind_session_transcript_slice(
        controller,
        (talk, _Turns(), _Attachments(), _Rows()),
        directory=Path("."),
    )
    return controller.registered()


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
