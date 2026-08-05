#!/usr/bin/env python3
"""Keep local and hosted CI orchestration equivalent.

Splitting one ``make ci-hosted`` invocation across several Actions jobs makes
the workflow faster, but it also creates a silent failure mode: removing a
target from a Make group, omitting a group from the workflow, or forgetting a
lane in the protected aggregator all make CI greener. This gate treats those
three lists as one contract and rejects drift in the permissive direction.

The parser is deliberately narrow. These files are repository-owned policy,
not arbitrary Make or YAML, and accepting an unfamiliar spelling by guessing
would turn a parse failure into the silent omission this gate exists to stop.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAKEFILE = Path("Makefile")
HOSTED_WORKFLOW = Path(".github/workflows/ci.yml")

GROUPS = {
    "verify-quick": "VERIFY_QUICK",
    "verify-coverage": "VERIFY_COVERAGE",
    "verify-mutation": "VERIFY_MUTATION",
    "verify-security": "VERIFY_SECURITY",
}
VERIFY_GROUPS = ("verify-quick", "verify-coverage", "verify-mutation")
HOSTED_GROUPS = (*VERIFY_GROUPS, "verify-security")

REQUIRED_TARGETS = {
    "VERIFY_QUICK": {
        "format-check",
        "lint",
        "types",
        "test-integrity",
        "context-budget",
        "worker-threads",
        "ratchet",
        "shellcheck",
        "workflows",
        "orchestration",
    },
    "VERIFY_COVERAGE": {"test-coverage"},
    "VERIFY_MUTATION": {"mutation"},
    "VERIFY_SECURITY": {"security-static"},
}

AGGREGATOR_JOB = "quality-gate"
AGGREGATOR_NAME = "Quality and security"


def _make_words(source: str, name: str) -> tuple[str, ...] | None:
    match = re.search(
        rf"^{re.escape(name)}\s*:=\s*(\S(?:.*\S)?)\s*$",
        source,
        flags=re.MULTILINE,
    )
    return tuple(match.group(1).split()) if match else None


def _prerequisites(source: str, target: str) -> tuple[str, ...] | None:
    match = re.search(rf"^{re.escape(target)}:\s*(.*?)\s*$", source, flags=re.MULTILINE)
    return tuple(match.group(1).split()) if match else None


def _workflow_jobs(source: str) -> dict[str, str]:
    """Return top-level job blocks from the repository's constrained YAML."""
    lines = source.splitlines(keepends=True)
    jobs_start = next(
        (index for index, line in enumerate(lines) if line.rstrip() == "jobs:"),
        None,
    )
    if jobs_start is None:
        return {}

    starts = [
        (index, match.group(1))
        for index, line in enumerate(lines[jobs_start + 1 :], jobs_start + 1)
        if (match := re.match(r"^  ([a-zA-Z0-9_-]+):\s*$", line))
    ]
    jobs: dict[str, str] = {}
    for position, (start, job_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        jobs[job_id] = "".join(lines[start:end])
    return jobs


def _invoked_group(job: str) -> str | None:
    matches = re.findall(r"^\s+run:\s+make\s+([a-zA-Z0-9_-]+)\s*$", job, re.MULTILINE)
    groups = [target for target in matches if target in GROUPS]
    return groups[0] if len(groups) == 1 else None


def _aggregator_needs(job: str) -> set[str] | None:
    match = re.search(r"^\s+needs:\s*\[([^]]+)]\s*$", job, re.MULTILINE)
    if match is None:
        return None
    return {value.strip() for value in match.group(1).split(",") if value.strip()}


def _check_makefile(source: str, failures: list[str]) -> None:
    for group_target, variable in GROUPS.items():
        actual = _make_words(source, variable)
        required = REQUIRED_TARGETS[variable]
        if actual is None:
            failures.append(
                f"Makefile: {variable} is missing or is not a simple := list."
            )
        elif set(actual) != required or len(actual) != len(required):
            failures.append(
                f"Makefile: {variable} must contain exactly {sorted(required)}; "
                f"found {list(actual)}."
            )

        expected_reference = (f"$({variable})",)
        if _prerequisites(source, group_target) != expected_reference:
            failures.append(
                f"Makefile: {group_target} must depend exactly on $({variable})."
            )

    expected_prerequisites = {
        "verify": VERIFY_GROUPS,
        "ci-hosted": ("verify", "verify-security"),
        "security": ("security-static", "secrets"),
        "ci": ("verify", "security"),
    }
    for target, expected in expected_prerequisites.items():
        if _prerequisites(source, target) != expected:
            failures.append(
                f"Makefile: {target} must depend, in order, on {' '.join(expected)}."
            )


def _check_aggregator(job: str, lane_jobs: set[str], failures: list[str]) -> None:
    name = re.search(r"^\s+name:\s*(.+?)\s*$", job, re.MULTILINE)
    if name is None or name.group(1) != AGGREGATOR_NAME:
        failures.append(
            f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} must keep the protected "
            f"check name {AGGREGATOR_NAME!r}."
        )

    needs = _aggregator_needs(job)
    if needs != lane_jobs:
        failures.append(
            f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} needs must equal all quality "
            f"lanes {sorted(lane_jobs)}; found {sorted(needs or set())}."
        )

    if not re.search(r"^\s+if:\s*\$\{\{\s*always\(\)\s*}}\s*$", job, re.MULTILINE):
        failures.append(
            f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} must run under if: always()."
        )

    if "LANE_RESULTS: ${{ join(needs.*.result, ' ') }}" not in job:
        failures.append(
            f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} must collect every needs result."
        )

    success_list = " ".join("success" for _ in lane_jobs)
    positive_check = f'test "$LANE_RESULTS" = "{success_list}"'
    if positive_check not in job:
        failures.append(
            f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} must positively require "
            "every lane result to be success."
        )


def _check_workflow(source: str, failures: list[str]) -> None:
    jobs = _workflow_jobs(source)
    if not jobs:
        failures.append(f"{HOSTED_WORKFLOW}: jobs are missing or could not be parsed.")
        return

    lanes_by_group: dict[str, list[str]] = {group: [] for group in HOSTED_GROUPS}
    for job_id, job in jobs.items():
        group = _invoked_group(job)
        if group is not None:
            lanes_by_group[group].append(job_id)
            if re.search(r"^\s+continue-on-error:\s*true\s*$", job, flags=re.MULTILINE):
                failures.append(
                    f"{HOSTED_WORKFLOW}: lane {job_id} must not continue on error."
                )

    for group, lane_jobs in lanes_by_group.items():
        if len(lane_jobs) != 1:
            failures.append(
                f"{HOSTED_WORKFLOW}: {group} must be invoked by exactly one lane; "
                f"found {lane_jobs}."
            )

    lane_jobs = {job_ids[0] for job_ids in lanes_by_group.values() if len(job_ids) == 1}
    aggregator = jobs.get(AGGREGATOR_JOB)
    if aggregator is None:
        failures.append(f"{HOSTED_WORKFLOW}: {AGGREGATOR_JOB} job is missing.")
    else:
        _check_aggregator(aggregator, lane_jobs, failures)


def main() -> int:
    failures: list[str] = []
    if not MAKEFILE.is_file():
        failures.append(f"{MAKEFILE}: missing.")
    else:
        _check_makefile(MAKEFILE.read_text(encoding="utf-8"), failures)

    if not HOSTED_WORKFLOW.is_file():
        failures.append(f"{HOSTED_WORKFLOW}: missing.")
    else:
        _check_workflow(HOSTED_WORKFLOW.read_text(encoding="utf-8"), failures)

    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(f"\n{len(failures)} CI orchestration failure(s).")
        return 1

    print("CI orchestration: local and hosted required targets agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
