#!/usr/bin/env python3
"""Fail when the mutation score drops below the recorded floor.

Line coverage proves a line ran. A mutation score proves a test would notice
if that line were wrong: mutmut corrupts the source, reruns the suite, and a
surviving mutant is a change no assertion detected. That is the one signal a
test written only to pass cannot fake, so it is a merge gate rather than a
report.

Raise MUTATION_SCORE_FLOOR as survivors are killed. Never lower it.

The reachable ceiling is not 100%. In domain.py and config.py, thirteen
survivors are equivalent mutants — they cannot change behavior, so no test can
detect them:

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

That put the ceiling at 253/268 while those two modules were the whole scope.

codex.py joined the scope on 2026-07-28 and brought its own equivalents, of
the same kinds plus two the earlier modules had no occasion for:

  * `flush=True` on the one startup `print` becomes `flush=False`, `None`, or
    nothing at all. The line is printed either way; only the moment it reaches
    a pipe differs, and no assertion can see that.
  * `suppress(Exception)` becomes `suppress(None)` in the four places that
    guard an interrupt. Both behave identically unless the suppressed call
    raises, and the ones a test can make raise are covered — what is left is
    the guard around a call that cannot fail in a fake.
  * `False` becomes `None` on flags only ever read for truthiness
    (`message_open`, `saw_delta`, `_sdk_loaded`), and `ensure_ascii=False`
    becomes `ensure_ascii=None`, which json.dumps treats the same way.
  * `join(timeout=3)` becomes `join(timeout=4)` and the queue wait moves from
    0.2s to 1.2s. Both are still bounded, which is the property that matters;
    pinning the exact number would be a test of the clock.

catalog.py joined on 2026-07-29. Its three survivors replace static-only
``typing.cast`` calls with other cast targets. A cast is erased at runtime, so
all three execute exactly the same parsing behavior.

tagalong/control/ joined on 2026-08-05, the whole package at once: it is pure
logic with no device, process, or network boundary, so there is no adapter in
it to make a survivor untestable. It took the run from 861 to 1,586 mutants
and leaves two:

tagalong/discovery.py joined on 2026-08-05 on the same rule. It transforms
the static catalog and slash table into listing rows and help text; there is
no collaborator to fake, so a survivor is an assertion gap.

  * ``popitem(last=False)`` becomes ``popitem(last=None)``. Both are falsy, so
    both evict the oldest idempotency key.
  * ``self._arrived.wait(timeout)`` becomes ``wait(None)``, which blocks
    forever. A test that killed it would be a test that hangs, so it times out
    rather than dying, and a timeout counts against the score.

The floor moved 94 -> 95 with that run, against 96.3% measured. It is not set
at 96 because that timeout is the difference: a mutant that times out rather
than dying is scored by how loaded the machine is, and a floor a busy CI run
can trip is a floor people learn to rerun rather than believe.

Do not chase the difference, and do not silence it with `# pragma: no mutate`
either — an equivalent mutant is evidence the code is precise, not evidence a
test is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MUTATION_SCORE_FLOOR = 95.0
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
