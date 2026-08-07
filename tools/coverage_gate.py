#!/usr/bin/env python3
"""Enforce a per-file branch-coverage floor that only ever ratchets upward.

A single project-wide threshold lets a well-covered module pay for an
uncovered one: at a 30% global floor the project measured 43%, and every point
of that slack sat in cli.py at 23%. Hundreds of untested lines could be added
there without turning the gate red.

Each file therefore carries its own floor, recorded at the value it had when
the gate was introduced. Raise an entry when coverage improves so the gain is
locked in. Never lower one — that is the whole point of a ratchet.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

COVERAGE_PATH = Path("coverage.json")

# Recorded 2026-07-26. Raise as coverage improves; never lower.
#
# The tagalong entries were re-recorded on 2026-07-27 when cli.py was split
# into per-concern modules. The single 83% floor it used to carry was an
# average, and averages are what this gate exists to stop: it hid a 26% wiring
# root and a 58% listener behind well-covered choosers and Codex handling.
# Splitting made both visible and both were tested rather than floored down.
FLOORS = {
    # The gates themselves. tests/test_quality_gates.py plants a violation for
    # each and asserts it is caught, because a gate that matches nothing still
    # reports green.
    "tools/context_budget.py": 93.0,
    "tools/coverage_gate.py": 75.0,
    "tools/mutation_gate.py": 78.0,
    # Recorded 2026-08-05 with the hosted lane split from issue #87.
    "tools/orchestration_gate.py": 98.0,
    # Raised 2026-08-07 from 92 with Electron floor ratchet paths (#97 phase 3).
    "tools/ratchet_gate.py": 93.0,
    "tools/test_integrity.py": 93.0,
    "tools/worker_gate.py": 97.0,
    # Recorded 2026-08-05 with the catalog handler gate for issue #81 milestone 8.
    "tools/catalog_gate.py": 97.0,
    "tagalong.py": 83.0,
    "tagalong/__init__.py": 100.0,
    # Recorded 2026-08-05 with the first TUI slice adapter for issue #81.
    "tagalong/application.py": 100.0,
    # Recorded 2026-08-05 with slash adapters over the typed catalog.
    "tagalong/discovery.py": 100.0,
    # Recorded 2026-08-05 with the local JSON-RPC transport and MCP adapter.
    "tagalong/transport.py": 100.0,
    "tagalong/mcp.py": 100.0,
    # Raised 2026-07-30 from 94 when microphone retarget/release paths
    # closed the remaining null-stream branches.
    "tagalong/capture.py": 96.0,
    "tagalong/catalog.py": 100.0,
    # Raised 2026-07-28 from 94, then to 100 when the virtual meeting sink
    # left and the application-stream chooser that replaced it arrived fully
    # covered.
    "tagalong/choosers.py": 100.0,
    # Wiring only, and every connection it makes is asserted in tests/test_cli.py.
    # Raised 2026-07-28 from 98.
    "tagalong/cli.py": 99.0,
    # Recorded 2026-07-30 with the slash-command palette catalog and matcher.
    "tagalong/commands.py": 100.0,
    "tagalong/codex.py": 99.0,
    "tagalong/config.py": 100.0,
    # Recorded 2026-08-05 with the UI-neutral application core. It has no
    # audio, process, or network boundary — every branch is reachable from a
    # test — so it is recorded at what it measured and nothing less.
    "tagalong/control/__init__.py": 100.0,
    "tagalong/control/actions.py": 100.0,
    "tagalong/control/actors.py": 100.0,
    "tagalong/control/controller.py": 100.0,
    "tagalong/control/events.py": 100.0,
    "tagalong/control/outcomes.py": 100.0,
    "tagalong/control/policy.py": 100.0,
    "tagalong/control/state.py": 100.0,
    "tagalong/domain.py": 98.0,
    "tagalong/listener.py": 100.0,
    # Recorded 2026-07-27 with the speech-provider split. Each measured the
    # same figure across three consecutive runs, so none of these carries the
    # thread-interleaving slack the tts.py note below describes: piper_tts.py
    # 95.73, playback.py 98.41, speech.py 100.00.
    "tagalong/piper_tts.py": 96.0,
    "tagalong/playback.py": 98.0,
    "tagalong/presentation.py": 100.0,
    # Recorded 2026-08-05 with the shared queued-speech lifecycle.
    "tagalong/queued_tts.py": 100.0,
    # Recorded 2026-07-29 with the session tagging that lets an orphaned
    # helper be recognized and swept.
    "tagalong/session.py": 100.0,
    "tagalong/speech.py": 100.0,
    "tagalong/startup.py": 100.0,
    # Recorded 2026-07-28 with the PipeWire stream tap.
    "tagalong/streams.py": 100.0,
    # This floor sat at 80 while the measurement moved between runs: five runs
    # of the same commit gave 80.99, 81.82, 81.82, 81.82, 81.40, because the
    # guard in _play was hit or missed depending on how the synthesis thread
    # interleaved with shutdown. The note here said the fix was to fake the
    # network per branch rather than race it, and tests/test_tts.py now does
    # exactly that: every abort and shutdown path waits on an event the fake
    # player sets. Five runs of the current tree all measure 97.19, so this is
    # the honest floor rather than the top of a flapping range.
    "tagalong/tts.py": 98.0,
    "tagalong/tui.py": 95.0,
}

# A module added after this gate existed has no legacy excuse.
NEW_FILE_FLOOR = 60.0


def _check_recorded_floors(
    measured: dict[str, float],
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    improvements: list[str] = []
    for path, floor in sorted(FLOORS.items()):
        if path not in measured:
            failures.append(
                f"{path}: has a recorded floor but was not measured. Remove its "
                f"FLOORS entry if the file is gone."
            )
            continue
        actual = measured[path]
        if actual < floor:
            failures.append(f"{path}: {actual:.1f}% is below its floor of {floor:.1f}%")
        # Report a full point of headroom so the ratchet gets turned.
        elif math.floor(actual) >= floor + 1:
            improvements.append(
                f"{path}: {actual:.1f}% — raise its floor to {math.floor(actual):.0f}"
            )
    return failures, improvements


def _check_new_files(measured: dict[str, float]) -> list[str]:
    return [
        f"{path}: new file at {actual:.1f}% must reach "
        f"{NEW_FILE_FLOOR:.0f}% or record an explicit floor with a reason."
        for path, actual in sorted(measured.items())
        if path not in FLOORS and actual < NEW_FILE_FLOOR
    ]


def main() -> int:
    if not COVERAGE_PATH.exists():
        print(f"error: {COVERAGE_PATH} is missing; run `make test-coverage` first.")
        return 1

    report = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    measured = {
        path: data["summary"]["percent_covered"]
        for path, data in report["files"].items()
    }

    failures, improvements = _check_recorded_floors(measured)
    failures.extend(_check_new_files(measured))

    for failure in failures:
        print(f"error: {failure}")
    for improvement in improvements:
        print(f"note: {improvement}")

    if failures:
        print(f"\n{len(failures)} per-file coverage failure(s).")
        return 1

    print(f"per-file coverage: {len(FLOORS)} files at or above their floors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
