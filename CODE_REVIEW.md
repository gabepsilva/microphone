# Pull requests and code review

This file is the agent-facing workflow for landing changes. Branch protection
settings live in GitHub's UI and are recorded in [QUALITY.md](QUALITY.md);
`make ci` remains the local definition of done.

You do not need this file to implement a feature or fix a bug. Read it before
opening, updating, watching, or merging a pull request.

## Branch and open

`master` is protected. Do not commit to it directly.

1. Create or switch to a feature branch from an up-to-date `master`.
2. Commit on that branch only.
3. Push the branch and open a pull request with the GitHub CLI:

   ```bash
   gh pr create
   ```

4. In the PR description, explain any change to `pyproject.toml`, `uv.lock`,
   CI workflows, thresholds, or security policy.

## Watch CI until green

After every open or update of a pull request:

1. Monitor the required checks until they finish. Prefer:

   ```bash
   gh pr checks --watch
   ```

2. If a check fails, fix the failure on the branch, push, and watch again.
3. Do not merge while any required check is pending or red.
4. Never bypass protection: no `gh pr merge --admin`, no force push to
   `master`, and no removing or weakening a required check to get a merge.

Required checks are `Quality and security` and `Secret scan`. Together they
match `make ci` / `make ci-hosted` plus the hosted Gitleaks job.

## Merge

Merge only after every required check is green. Use the GitHub CLI (`gh pr
merge`) and confirm the PR lands on `master`. Apply the same CI-green rule when
merging any approved pull request into `master`.

## What review is for

Deterministic gates and behavior-specific tests are the merge authority. AI
review comments are advisory: useful signals, not a substitute for green CI or
for a test that would fail if the change were wrong.

Review still has to catch what tools cannot:

- A regression test that passes without the fix is not evidence; for bug fixes,
  run `make verify-regression TEST=<selection>`.
- New subprocess invocations need a test or a direct human review of where
  their arguments came from. Semgrep already rejects `shell=True` and string
  commands; provenance is the remaining rule.
- Do not satisfy a gate by deleting a test, weakening an assertion, faking the
  unit under test, or relaxing a threshold. `make ratchet` blocks some of
  those moves; intent still matters for the ones it cannot see.

If CI is green and the change is wrong, the missing piece is a better test, not
a merge bypass.
