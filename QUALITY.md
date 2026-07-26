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
| Tests | `make test-coverage` | Deterministic unit and Textual interaction behavior. |
| Shell syntax | `make shellcheck` | Parse the privileged setup script before it is run. |
| Workflow integrity | `make workflows` | Validate GitHub Actions syntax and expressions. |
| Static policy/SAST | `make semgrep` | Blocks committed rules for dangerous APIs and project security policy. |
| Static security | `make security-static` | Blocks Bandit medium/high findings and known dependency vulnerabilities. |
| Secret detection | `make secrets` | Scans the complete Git history with Gitleaks. |

The test suite must not need an audio device, Codex account, network call,
PipeWire service, timer delay, or a real external process. Tests use fakes and
synthetic events at those boundaries. Any bug that can be reproduced without
hardware gets a regression test.

Branch coverage is measured for application scripts and currently has a 30%
floor. This is a deliberately visible baseline, not a claim of sufficient
feature coverage: the current monolithic audio runtime has 32% measured
coverage. Do not lower the floor. Raise it as adapters are isolated, first to
60% and then to 80%; new behavior always needs a focused acceptance or
regression test regardless of the global percentage.

## Security policy

Bandit writes all findings to `reports/bandit.json` and blocks medium/high
findings. The current low-severity subprocess findings are retained in that
report; none are hidden with `# nosec`. Each new subprocess invocation must use
an argument list, never `shell=True`, and must have a test or direct review of
its argument provenance.

`pip-audit` is a merge gate. Its vulnerability feed changes over time, so an
unchanged lockfile can legitimately fail after a newly published advisory.
Gitleaks is also a merge gate and scans history, not only changed files.

Semgrep is a serverless merge gate. `make semgrep` runs the official
digest-pinned Semgrep CLI container with networking and its version check
disabled, then mounts the repository read-only. It never downloads a remote
rule pack: the only rules are the repository's committed `semgrep.yml`. The
rules reject dynamic code execution, unsafe deserialization, disabled TLS
verification, string-based process commands, and shell-based process
invocation. Its JSON evidence is written to `reports/semgrep.json`. Use
`make semgrep` for immediate file-and-line feedback; `make ci` and the hosted
quality job run it through `make security-static`.

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

Do not add tools simply because they exist. Mutation testing belongs after the
transcript/domain logic is extracted from the current runtime and has a strong
unit-test baseline. Hardware-in-the-loop tests belong only once a controlled
test device or emulator exists. Add architecture-import rules once the project
has stable package boundaries.
