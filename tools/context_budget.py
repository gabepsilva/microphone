#!/usr/bin/env python3
"""Cap the instructions an agent loads on every task.

Coverage, mutation score, and every threshold in this repository ratchet
upward. Instruction files have no such control, and they only move one way:
each problem solved wants to leave prose behind, and nothing ever measures the
total. Left alone that crowds the model's attention with policy at the expense
of the code under change.

`AGENTS.md` is loaded for every task, so it is the file that matters. Exceeding
its budget is not a signal to raise the budget — it is a signal that a rule
belongs somewhere it costs nothing:

  * a gate's failure message, delivered exactly when it is relevant
  * a comment on the config line it describes
  * a test that makes the rule unnecessary to state

`QUALITY.md` is deliberately uncapped. It is read only when changing a gate,
threshold, scanner, or workflow, and its own audience statement is what keeps
it honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Words, not tokens: reproducible without a tokenizer, and close enough at
# roughly 1.5 tokens per word for prose.
BUDGETS = {
    "AGENTS.md": 400,
}


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
