#!/usr/bin/env python3
"""Fail when electron/src/protocol/actions.ts drifts from CATALOG.

The Electron client must not invent action ids. TypeScript gets a committed
const map generated from the Python catalog; this gate re-renders that file
and rejects drift — the same lockfile discipline as ``uv.lock``. Runtime
``capabilities`` discovery already exists on the socket and is product work
(#96), not a CI gate.

The generator imports :data:`~tagalong.control.actions.CATALOG`, so this
target lives in ``VERIFY_QUICK`` (needs ``uv``). The Bun lane treats the
committed file as input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tagalong.control.actions import CATALOG, ActionSpec

ACTIONS_PATH = Path("electron/src/protocol/actions.ts")

_HEADER = """\
/** Generated from tagalong.control.actions.CATALOG. Do not edit by hand. */
"""


def ts_key(action_id: str) -> str:
    """``tts.set_enabled`` → ``tts_set_enabled``."""
    return action_id.replace(".", "_")


def render_actions_ts(catalog: tuple[ActionSpec, ...] = CATALOG) -> str:
    """Return the TypeScript source that should be committed at ACTIONS_PATH."""
    lines = [
        _HEADER.rstrip("\n"),
        "export const ACTIONS = {",
    ]
    for spec in catalog:
        key = ts_key(spec.id)
        lines.append(f'  {key}: "{spec.id}",')
    lines.extend(
        [
            "} as const;",
            "",
            "export type ActionId = (typeof ACTIONS)[keyof typeof ACTIONS];",
            "",
        ]
    )
    return "\n".join(lines)


def drift_problems(
    committed: str,
    *,
    expected: str | None = None,
) -> list[str]:
    """Return human-readable problems when *committed* does not match *expected*."""
    want = render_actions_ts() if expected is None else expected
    if committed == want:
        return []
    return [
        f"{ACTIONS_PATH}: out of sync with CATALOG — "
        "run `make electron-actions-write` and commit the result"
    ]


def check(*, path: Path | None = None) -> list[str]:
    # Resolve ACTIONS_PATH at call time so tests can monkeypatch the module attr.
    target = ACTIONS_PATH if path is None else path
    if not target.is_file():
        return [f"{target}: missing; run `make electron-actions-write`"]
    return drift_problems(target.read_text(encoding="utf-8"))


def write(*, path: Path | None = None) -> None:
    target = ACTIONS_PATH if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_actions_ts(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"regenerate {ACTIONS_PATH} from CATALOG",
    )
    args = parser.parse_args(argv)
    if args.write:
        write()
        print(f"wrote {ACTIONS_PATH} ({len(CATALOG)} actions)")
        return 0
    problems = check()
    if not problems:
        print(f"electron actions: {ACTIONS_PATH} matches CATALOG ({len(CATALOG)} ids)")
        return 0
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
