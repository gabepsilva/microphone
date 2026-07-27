# Voice Codex quality system

Voice Codex is developed primarily with AI assistance. A generated change is
not accepted merely because it looks plausible: it must pass repeatable checks
and carry direct behavior evidence.

**This file holds what the repository cannot**: settings that live in GitHub's
web UI, decisions taken *against* an alternative — which by definition leave no
artifact — and planned direction. Anything derivable from the Makefile, a
gate's failure output, or a comment on the config line it describes belongs
there instead, where it is maintained next to the thing that could invalidate
it. You do not need this file to write a feature or fix a bug; `make ci` and
`AGENTS.md` are sufficient for that.

The only local definition of done is:

```bash
make ci
```

Run `uv sync --locked --all-groups` first. Run `make hooks` once after cloning
to install `make verify` as a pre-commit hook and `make ci` as a pre-push hook;
re-run `make hook-check` after changing hook configuration. Hooks are a fast
local convenience, not the authority — they are per-clone, opt-in, and
bypassable. CI decides.

## Required checks

| Control | Command | Purpose |
| --- | --- | --- |
| Formatting | `make format-check` | One canonical Python style. |
| Lint | `make lint` | Imports, correctness smells, and modern Python rules. |
| Types | `make types` | Nullability and third-party API contract errors. |
| Tests | `make test-coverage` | Deterministic unit and Textual interaction behavior, plus per-file coverage floors. |
| Test integrity | `make test-integrity` | Rejects tests with no assertion and skips that name no issue. |
| Mutation score | `make mutation` | Proves the domain tests detect corrupted behavior, not just execute it. |
| Shell syntax | `make shellcheck` | Parse the privileged setup script before it is run. |
| Workflow integrity | `make workflows` | Validate GitHub Actions syntax and expressions. |
| Worker threads | `make worker-threads` | Every background thread is a daemon and every join on one is bounded. |
| Threshold ratchet | `make ratchet` | Blocks a lowered floor, a narrowed mutation scope, or a deleted rule. |
| Static policy/SAST | `make semgrep` | Blocks committed rules for dangerous APIs and project security policy. |
| Static security | `make security-static` | Blocks Bandit medium/high findings and known dependency vulnerabilities. |
| Secret detection | `make secrets` | Scans the complete Git history with Gitleaks. |

The test suite must not need an audio device, Codex account, network call,
PipeWire service, timer delay, or a real external process. Tests use fakes and
synthetic events at those boundaries. Any bug that can be reproduced without
hardware gets a regression test.

## Test-quality controls

A suite written largely by AI can be green without being evidence: coverage and
a passing run cannot distinguish a meaningful test from one that merely
executes code. Each control below is documented in full at its implementation,
and prints what to do when it fails. This is the map, not the manual.

- **Per-file coverage floors** (`tools/coverage_gate.py`) stop a well-covered
  module from paying for an uncovered one.
- **Diff coverage** (`DIFF_COVERAGE_MIN` in the Makefile) is the
  machine-checked form of "every behavior change needs a test".
- **Mutation score** (`tools/mutation_gate.py`, scope in `[tool.mutmut]`)
  proves a test would notice if a line were wrong, where coverage only proves
  the line ran.
- **Threshold ratchet** (`tools/ratchet_gate.py`) checks the gates themselves,
  since nothing else stops an agent from editing one instead of satisfying it.
- **Fail-first verification** (`tools/verify_regression.sh`) proves a
  regression test fails without its fix.
- **Test integrity** (`tools/test_integrity.py`) rejects tests that cannot fail
  and skips that name no issue.
- **Gate self-tests** (`tests/test_quality_gates.py`) plant a violation for
  every gate, because a gate that matches nothing still reports green.
- **Worker-thread contract** (`tools/worker_gate.py`) requires every
  background thread to be a daemon and every join on one to pass a timeout.
  Four classes follow this and nothing enforced it: it survived only because
  all four shared a file, so whoever wrote the next one could see the other
  three. Splitting `cli.py` removed that, and the failure it prevents — a
  process that will not exit — raises nothing and loses no coverage.
- **Context budget** (`tools/context_budget.py`) caps `AGENTS.md`, the only
  file loaded on every task. Everything else here ratchets upward; instructions
  are the one thing that must not. The cap is raisable, but only against a
  recorded `BUDGET_RAISES` entry — a threshold nobody can move gets deleted
  rather than respected, so the goal is to make moving it deliberate and
  permanently visible, not impossible.

Recorded only here: per-file floors are set at the value each file had when the
gate was added and ratchet upward only. The 80% goal was reached by isolating
the adapters from the runtime, one class per pull request, and the global floor
now simply follows what the suite achieves.

This document previously argued the opposite of what `tests/test_cli.py` now
does, and the reversal is worth recording rather than quietly overwriting. The
old position was that `main()` is process-lifetime wiring, that every
collaborator it builds is covered on its own, and that faking all of them at
once would assert the shape of the wiring rather than any behavior — a number,
not evidence.

Splitting `cli.py` into per-concern modules on 2026-07-27 changed that premise.
The collaborators are still covered individually, but they are now separately
movable, and the shape of the wiring became the thing a refactor breaks: an
unbound hook, a channel never registered, or a transcriber built without its
listener raises nothing, fails no other test, and ships. That is a behavior —
"the session that starts is the session that was asked for" — and it is worth
asserting precisely because nothing else can catch it. So `main` is now
exercised with each adapter faked at `cli`'s own import boundary, asserting the
connections rather than re-testing the parts.

That gap was the Edge TTS pipeline's error and cancellation branches, and it
is closed: each one now has a network fake and waits on an event the fake
player sets, so `voice_codex/tts.py` measures 97% and does so identically
across repeated runs. It used to flap by a point between runs, which is why
its floor sat at 80 until the races were fixed rather than floored around.

The lowest floor in the package is now `voice_codex/capture.py` at 86%, and
what it leaves uncovered is the microphone adapter itself — the subclass that
opens a PortAudio stream. That one is a decision rather than a gap: it is the
hardware boundary, and no fake can assert the thing that makes it real.

## Security policy

Three decisions, none of them visible in the commands themselves:

**Low-severity Bandit findings are recorded, not suppressed.** They stay in
`reports/bandit.json`; nothing is hidden behind `# nosec`.

**Semgrep runs offline by construction** — a digest-pinned container with
networking disabled — so it can never pull a remote rule pack. The only rules
are the committed `semgrep.yml`, and `.semgrepignore` is load-bearing rather
than incidental; its header says why.

**Subprocess argument provenance is reviewed by a human.** Semgrep already
rejects `shell=True` and string commands, so the rule left over is the one no
tool can check: every new subprocess invocation needs a test or a direct review
of where its arguments came from.

`pip-audit` and Gitleaks are merge gates whose inputs move on their own; see
the scheduled audit below. Local `voice.yaml`, transcripts, and recordings are
never committed.

## CI and merge policy

The hosted quality job runs `make ci-hosted`; a separate Gitleaks job supplies
the secret scan, because that action installs the scanner itself. Together they
are equivalent to `make ci`.

Branch protection is GitHub UI state and exists nowhere in this repository, so
it is recorded here. Protect `master` with pull requests required, force pushes
disabled, and these required checks:

- `Quality and security`
- `Secret scan`

Also enable **Do not allow bypassing the above settings**. Without it the rules
apply to everyone except the account most likely to be driving an agent, and
"do not use `--admin`" becomes a request rather than a control — the same
distinction `ratchet_gate.py` exists to make for thresholds. With it enabled,
`gh pr merge --admin` fails instead of succeeding quietly, which is the whole
point: the person merging should not be able to decide the gates do not apply.

AI review is advisory only. Deterministic checks and behavior-specific tests
are the merge authority.

### Scheduled audit

The workflow also runs weekly; `ci.yml` says why, and `dependabot.yml` covers
the fix path. A red scheduled run blocks nothing, since branch protection gates
pull requests and not schedules — it is an alarm someone has to act on.

GitHub disables scheduled workflows after 60 days of repository inactivity, so
the safety net switches itself off exactly when nobody is watching; one commit
re-arms it. Failure notifications go to whoever last edited the cron
expression, not to the repository owner.

## Tooling boundaries

Do not add tools simply because they exist. Hardware-in-the-loop tests belong
only once a controlled test device or emulator exists. Add architecture-import
rules once the project has stable package boundaries.

Docstring linting (ruff `D`) was evaluated and rejected. It checks that a
docstring exists, never that it is true, so the cheapest way to satisfy it is
to restate the function name — text that reads as documentation and is never
revalidated. If the goal is machine-checked intent at function boundaries, use
`ANN` instead: `ty` verifies annotations against real call sites.
