#!/usr/bin/env python3
"""Fail when the mutation score drops below the recorded floor.

Line coverage proves a line ran. A mutation score proves a test would notice
if that line were wrong: mutmut corrupts the source, reruns the suite, and a
surviving mutant is a change no assertion detected. That is the one signal a
test written only to pass cannot fake, so it is a merge gate rather than a
report.

Raise MUTATION_SCORE_FLOOR as survivors are killed. Never lower it.

The reachable ceiling is not 100%. Thirteen survivors are equivalent mutants —
they cannot change behavior, so no test can detect them:

  * `rfind(x, None, n)` is `rfind(x, 0, n)`; None is the documented default.
  * `rfind(x, 1, n)` differs only when the break is at index 0, and it never
    is: `_emit_bounded` is called with stripped text, and the buffer keeps
    being lstripped as it is consumed.
  * `encoding="UTF-8"` is `encoding="utf-8"`; codec names are normalized.
  * `encoding=None` falls back to the locale encoding, which is UTF-8 wherever
    this runs, so it is only distinguishable by changing the environment.

Two more time out rather than dying: mutating `split_at` to None makes the
chunking loop consume nothing and spin forever. A timeout is not a kill, so
they count against the score.

That puts the ceiling at 253/268, which is what the floor now sits just below.
Do not chase the difference, and do not silence it with `# pragma: no mutate`
either — an equivalent mutant is evidence the code is precise, not evidence a
test is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MUTATION_SCORE_FLOOR = 94.0
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
