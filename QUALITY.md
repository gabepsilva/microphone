# Voice Codex quality system

Voice Codex is developed primarily with AI assistance. A generated change is
not accepted merely because it looks plausible: it must pass repeatable checks
and carry direct behavior evidence.

The only local definition of done is:

```bash
make ci
```

`uv sync --locked --all-groups` must be run first. CI uses the checked-in lock
file and must never resolve a new dependency graph while validating a change.

Run `make hooks` once after cloning. It installs `make verify` as a pre-commit
hook and `make ci` as a pre-push hook. Hooks make the normal local path safe,
but never replace CI: hooks can be bypassed and do not validate a clean hosted
environment. Run `make hook-check` after changing hook configuration.

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

Recorded only here: per-file floors are set at the value each file had when the
gate was added and ratchet upward only, and the global floor follows toward 60%
and then 80% as adapters are isolated from the runtime.

## Security policy

Bandit writes all findings to `reports/bandit.json` and blocks medium/high.
Low-severity findings are recorded there rather than suppressed; nothing is
hidden with `# nosec`. Semgrep already rejects `shell=True` and string
commands, so the rule that survives review is the one it cannot check: every
new subprocess invocation needs a test or a direct review of where its
arguments came from.

`pip-audit` is a merge gate. Its vulnerability feed changes over time, so an
unchanged lockfile can legitimately fail after a newly published advisory.
Gitleaks is also a merge gate and scans history, not only changed files.

Semgrep is a serverless merge gate. `make semgrep` runs a digest-pinned
container with networking disabled and the repository mounted read-only, so it
cannot download a remote rule pack: the only rules are the committed
`semgrep.yml`. Evidence lands in `reports/semgrep.json`. Use `make semgrep` for
immediate file-and-line feedback; `make ci` and the hosted quality job run it
through `make security-static`. The `.semgrepignore` at the repository root is
load-bearing — its header says why.

Generated coverage and security evidence is ignored by Git. GitHub Actions
uploads it on both success and failure. Local `voice.yaml`, transcripts, and
audio recordings are never committed.

## CI and merge policy

GitHub Actions uses immutable action revisions, Python 3.12.3, uv 0.11.16, and
`uv sync --locked --all-groups`. The hosted quality job runs `make ci-hosted`;
the separate Gitleaks job supplies the secret scan because its GitHub action
installs the scanner itself. Together they are equivalent to `make ci`.

Protect `master` in GitHub with pull requests required, force pushes disabled,
and these required checks:

- `Quality and security`
- `Secret scan`

AI review is advisory only. Deterministic checks and behavior-specific tests
are the merge authority.

## Tooling boundaries

Do not add tools simply because they exist. Hardware-in-the-loop tests belong
only once a controlled test device or emulator exists. Add architecture-import
rules once the project has stable package boundaries.

Docstring linting (ruff `D`) was evaluated and rejected. It checks that a
docstring exists, never that it is true, so the cheapest way to satisfy it is
to restate the function name — text that reads as documentation and is never
revalidated. If the goal is machine-checked intent at function boundaries, use
`ANN` instead: `ty` verifies annotations against real call sites.
