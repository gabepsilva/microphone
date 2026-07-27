#!/usr/bin/env python3
"""Fail when the mutation score drops below the recorded floor.

Line coverage proves a line ran. A mutation score proves a test would notice
if that line were wrong: mutmut corrupts the source, reruns the suite, and a
surviving mutant is a change no assertion detected. That is the one signal a
test written only to pass cannot fake, so it is a merge gate rather than a
report.

Raise MUTATION_SCORE_FLOOR as survivors are killed. Never lower it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MUTATION_SCORE_FLOOR = 58.0
STATS_PATH = Path("mutants/mutmut-cicd-stats.json")


def main() -> int:
    if not STATS_PATH.exists():
        print(f"error: {STATS_PATH} is missing; run `make mutation` first.")
        return 1

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    killed = stats["killed"]
    total = stats["total"]
    if total == 0:
        print("error: no mutants were generated; check [tool.mutmut] source_paths.")
        return 1

    # A suspicious or timed-out mutant is not evidence of a detecting test, so
    # only explicit kills count toward the score.
    score = 100.0 * killed / total
    print(
        f"mutation score {score:.1f}% "
        f"({killed} killed, {stats['survived']} survived, {total} mutants) "
        f"floor {MUTATION_SCORE_FLOOR:.1f}%"
    )

    if score < MUTATION_SCORE_FLOOR:
        print(
            "error: mutation score fell below the floor. Add assertions that "
            "distinguish correct output from the surviving mutants "
            "(`uv run mutmut results`, then `uv run mutmut show <mutant>`)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
