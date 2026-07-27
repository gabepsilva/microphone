#!/usr/bin/env python3
"""Cap the instructions an agent loads on every task.

Coverage, mutation score, and every threshold in this repository ratchet
upward. Instruction files have no such control, and they only move one way:
each problem solved wants to leave prose behind, and nothing ever measures the
total. Left alone that crowds the model's attention with policy at the expense
of the code under change.

`AGENTS.md` is loaded for every task, so it is the file that matters. Exceeding
its budget is not, by itself, a reason to raise the budget — first check
whether the rule belongs somewhere it costs nothing:

  * a gate's failure message, delivered exactly when it is relevant
  * a comment on the config line it describes
  * a test that makes the rule unnecessary to state
  * a scoped file with an audience statement, read only when it applies

The cap can still be raised, because a budget nobody can change is a budget
that gets deleted instead. It costs a `BUDGET_RAISES` entry that `make ratchet`
checks for, so the decision is recorded where the number lives rather than left
in a merged pull request nobody reopens.

`QUALITY.md` and `CODE_REVIEW.md` are deliberately uncapped. Each is read only
when its subject comes up, and its own audience statement is what keeps it
honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Words, not tokens: reproducible without a tokenizer, and close enough at
# roughly 1.5 tokens per word for prose.
BUDGETS = {
    "AGENTS.md": 533,
}

# A budget may be raised, but never quietly. `make ratchet` rejects a raise
# unless this change also adds an entry here naming the old cap, the new one,
# and why the rule could not live in a gate message, a config comment, or a
# test instead. Entries are kept after the fact, so the running cost of every
# past decision sits next to the current number rather than in a merged PR
# nobody reopens.
#
# Reusing an existing entry does not justify a new raise: the ratchet only
# accepts one added in the same change.
BUDGET_RAISES = [
    (
        "AGENTS.md",
        400,
        533,
        "2026-07-27: owner asked for headroom to keep pull-request workflow "
        "rules loaded on every task rather than deferred to CODE_REVIEW.md.",
    ),
]


def main() -> int:
    failures: list[str] = []

    for name, budget in sorted(BUDGETS.items()):
        path = Path(name)
        if not path.exists():
            failures.append(f"{name}: budgeted but missing.")
            continue

        words = len(path.read_text(encoding="utf-8").split())
        if words > budget:
            failures.append(
                f"{name}: {words} words exceeds its {budget}-word budget by "
                f"{words - budget}. Remove a rule, or move it into a gate "
                f"message, a config comment, or a test — do not raise the cap."
            )
        else:
            print(f"context budget: {name} {words}/{budget} words")

    for failure in failures:
        print(f"error: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
