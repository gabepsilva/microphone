#!/usr/bin/env python3
"""Fail when a quality threshold moves the wrong way.

Every other gate checks the code. This one checks the gates, because none of
them can stop an agent from editing the gate instead of satisfying it: a
lowered floor, a narrowed mutmut scope, or a deleted Semgrep rule all produce
a green run. That used to be prose in AGENTS.md asking politely; asking is not
a control.

Thresholds are read from the base branch and compared with the working tree.
Raising a floor is always allowed. Lowering one fails. A floor may be dropped
only when its source file is genuinely gone.

A threshold is only guarded if this file knows where it lives, so every new
one needs an entry here. They currently sit in five places: the gate scripts,
`pyproject.toml`, `semgrep.yml`, the Makefile, and
`electron/coverage_floors.json`.

Suppression counts are deliberately not ratcheted: AGENTS.md permits a
`# noqa` or `# nosec` that carries a finding ID and a justification, so a
count-based gate would fire on legitimate use and be silenced.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

COVERAGE_GATE = "tools/coverage_gate.py"
MUTATION_GATE = "tools/mutation_gate.py"
CONTEXT_BUDGET = "tools/context_budget.py"
PYPROJECT = "pyproject.toml"
SEMGREP_RULES = "semgrep.yml"
MAKEFILE = "Makefile"
ELECTRON_COVERAGE_FLOORS = "electron/coverage_floors.json"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Ask git about the working directory, not about whoever invoked this.

    Git exports GIT_DIR and GIT_INDEX_FILE to the hooks it runs, and they
    outrank the current directory. Inherited, this gate would compare the
    thresholds of the repository that launched it rather than the one being
    checked — which is exactly the silent no-op it exists to prevent, and it
    would only happen from a hook, where nobody is watching the output.
    """
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, env=environment
    )


def _read_base(base: str, path: str) -> str | None:
    """Return `path` as of `base`, or None when it did not exist there."""
    result = _git("show", f"{base}:{path}")
    return result.stdout if result.returncode == 0 else None


def _constant(source: str, name: str):
    """Pull a module-level literal without importing the file."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def _semgrep_rule_ids(source: str) -> set[str]:
    # Parsed by regex on purpose: PyYAML is not a dependency, and adding one
    # to read six rule ids is not worth the supply-chain surface.
    return set(re.findall(r"^\s*-\s*id:\s*(\S+)", source, flags=re.MULTILINE))


def _pyproject_numbers(source: str) -> tuple[float | None, list[str]]:
    config = tomllib.loads(source)
    fail_under = (
        config.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    )
    source_paths = config.get("tool", {}).get("mutmut", {}).get("source_paths", [])
    return fail_under, source_paths


def _check_coverage_floors(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, COVERAGE_GATE)
    if base_source is None:
        return
    base_floors = _constant(base_source, "FLOORS") or {}
    now_floors = _constant(Path(COVERAGE_GATE).read_text(encoding="utf-8"), "FLOORS")
    now_floors = now_floors or {}

    for path, was in base_floors.items():
        if path not in now_floors:
            if Path(path).exists():
                failures.append(
                    f"{path}: coverage floor removed while the file still exists."
                )
            continue
        if now_floors[path] < was:
            failures.append(
                f"{path}: coverage floor lowered {was:g} -> {now_floors[path]:g}."
            )

    base_new = _constant(base_source, "NEW_FILE_FLOOR")
    now_new = _constant(
        Path(COVERAGE_GATE).read_text(encoding="utf-8"), "NEW_FILE_FLOOR"
    )
    if base_new is not None and now_new is not None and now_new < base_new:
        failures.append(f"NEW_FILE_FLOOR lowered {base_new:g} -> {now_new:g}.")


def _electron_floors_payload(source: str) -> dict:
    return json.loads(source)


def _check_electron_floor_map(
    label: str,
    was_floors: dict,
    now_floors: dict,
    failures: list[str],
) -> None:
    for path, floor in was_floors.items():
        if path not in now_floors:
            if Path(f"electron/{path}").exists():
                failures.append(
                    f"electron/{path}: {label} floor removed while the file "
                    "still exists."
                )
            continue
        if now_floors[path] < floor:
            failures.append(
                f"electron/{path}: {label} floor lowered "
                f"{floor:g} -> {now_floors[path]:g}."
            )


def _check_electron_coverage_floors(base: str, failures: list[str]) -> None:
    """Electron floors live in JSON so the Bun gate and this ratchet share them.

    Weakening means a lowered per-file line/func floor, a lowered new-file
    floor, a removed floor while ``electron/<path>`` still exists, or a newly
    added ``unmeasured_entrypoints`` key (that shrinks what the gate measures,
    the same way narrowing mutmut ``source_paths`` does).
    """
    base_source = _read_base(base, ELECTRON_COVERAGE_FLOORS)
    if base_source is None:
        return
    if not Path(ELECTRON_COVERAGE_FLOORS).exists():
        failures.append(
            f"{ELECTRON_COVERAGE_FLOORS}: removed; Electron coverage floors "
            "would no longer be enforced."
        )
        return

    was = _electron_floors_payload(base_source)
    now = _electron_floors_payload(
        Path(ELECTRON_COVERAGE_FLOORS).read_text(encoding="utf-8")
    )
    _check_electron_floor_map(
        "line", was.get("floors") or {}, now.get("floors") or {}, failures
    )
    _check_electron_floor_map(
        "func",
        was.get("func_floors") or {},
        now.get("func_floors") or {},
        failures,
    )

    for key in ("new_file_floor", "new_file_func_floor"):
        was_new = was.get(key)
        now_new = now.get(key)
        if (
            was_new is not None
            and now_new is not None
            and float(now_new) < float(was_new)
        ):
            failures.append(f"electron {key} lowered {was_new:g} -> {now_new:g}.")

    was_exempt = set((was.get("unmeasured_entrypoints") or {}).keys())
    now_exempt = set((now.get("unmeasured_entrypoints") or {}).keys())
    added_exempt = sorted(now_exempt - was_exempt)
    if added_exempt:
        failures.append(
            f"electron unmeasured_entrypoints grew; no longer measured: {added_exempt}."
        )


def _check_mutation_floor(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, MUTATION_GATE)
    if base_source is None:
        return
    was = _constant(base_source, "MUTATION_SCORE_FLOOR")
    now = _constant(
        Path(MUTATION_GATE).read_text(encoding="utf-8"), "MUTATION_SCORE_FLOOR"
    )
    if was is not None and now is not None and now < was:
        failures.append(f"MUTATION_SCORE_FLOOR lowered {was:g} -> {now:g}.")


def _added_budget_raises(base_source: str, now_source: str) -> set[tuple]:
    """Return the (file, was, now) raises this change records for the first time.

    An entry carried over from the base is a receipt for a decision already
    taken, so reusing it would let the next raise ride on the last one's
    justification. Only an entry added here counts, and only with a reason.
    """
    already = {
        tuple(entry[:3]) for entry in (_constant(base_source, "BUDGET_RAISES") or [])
    }
    return {
        tuple(entry[:3])
        for entry in (_constant(now_source, "BUDGET_RAISES") or [])
        if tuple(entry[:3]) not in already and len(entry) > 3 and str(entry[3]).strip()
    }


def _check_context_budget(base: str, failures: list[str]) -> None:
    """A cap that can be raised on demand is not a cap.

    Raising it is the reflex the budget exists to interrupt, so it costs a
    recorded decision: the same standard AGENTS.md sets for a `# noqa`, which
    needs a rule id and a justification rather than bare permission.
    """
    base_source = _read_base(base, CONTEXT_BUDGET)
    if base_source is None:
        return
    now_source = Path(CONTEXT_BUDGET).read_text(encoding="utf-8")
    was = _constant(base_source, "BUDGETS") or {}
    now = _constant(now_source, "BUDGETS") or {}
    recorded = _added_budget_raises(base_source, now_source)

    for name, budget in was.items():
        if name not in now:
            failures.append(f"{name}: context budget removed.")
        elif now[name] > budget and (name, budget, now[name]) not in recorded:
            failures.append(
                f"{name}: context budget raised {budget} -> {now[name]} with no "
                f"BUDGET_RAISES entry recording why. Move a rule into a gate "
                f"message, a config comment, or a test — or add "
                f'("{name}", {budget}, {now[name]}, "<reason>") to '
                f"{CONTEXT_BUDGET}."
            )


def _check_pyproject(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, PYPROJECT)
    if base_source is None:
        return
    was_fail_under, was_paths = _pyproject_numbers(base_source)
    now_fail_under, now_paths = _pyproject_numbers(
        Path(PYPROJECT).read_text(encoding="utf-8")
    )

    if (
        was_fail_under is not None
        and now_fail_under is not None
        and now_fail_under < was_fail_under
    ):
        failures.append(
            f"coverage fail_under lowered {was_fail_under:g} -> {now_fail_under:g}."
        )

    # Same allowance the coverage floors get: a threshold may be dropped when
    # its source file is genuinely gone. Without this, renaming the package
    # reads as narrowing the scope, because the comparison is stringwise and
    # the base branch still spells the old prefix. Dropping a module that is
    # still there remains a failure, which is the case worth catching.
    dropped = {path for path in set(was_paths) - set(now_paths) if Path(path).exists()}
    if dropped:
        failures.append(
            f"mutmut source_paths narrowed; no longer mutated: {sorted(dropped)}."
        )


def _make_variable(source: str, name: str) -> float | None:
    """Read a `NAME ?= value` assignment without invoking make."""
    match = re.search(
        rf"^{re.escape(name)}\s*\?*=\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        source,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def _check_diff_coverage_floor(base: str, failures: list[str]) -> None:
    """The diff-coverage floor lives in the Makefile, not in a gate script.

    Every other threshold this file guards sits in Python or TOML, so the one
    written in make syntax was the only one an agent could lower with every
    check still green — and it is the threshold that governs new code, which
    is where a generated change actually lands.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None:
        return
    was = _make_variable(base_source, "DIFF_COVERAGE_MIN")
    if was is None:
        return
    now = _make_variable(
        Path(MAKEFILE).read_text(encoding="utf-8"), "DIFF_COVERAGE_MIN"
    )
    if now is None:
        failures.append(
            "DIFF_COVERAGE_MIN removed from the Makefile; changed lines would "
            "no longer need tests."
        )
    elif now < was:
        failures.append(f"DIFF_COVERAGE_MIN lowered {was:g} -> {now:g}.")


def _check_semgrep_rules(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, SEMGREP_RULES)
    if base_source is None:
        return
    dropped = _semgrep_rule_ids(base_source) - _semgrep_rule_ids(
        Path(SEMGREP_RULES).read_text(encoding="utf-8")
    )
    if dropped:
        failures.append(f"Semgrep rules deleted: {sorted(dropped)}.")


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "origin/master"

    resolved = _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    if resolved.returncode != 0:
        # Never pass quietly on a missing base: a ratchet that cannot compare
        # is exactly the silent no-op this gate exists to prevent.
        print(f"error: base ref '{base}' does not resolve.")
        print("Fetch it, or pass an explicit base: make ratchet RATCHET_BASE=<ref>")
        return 1

    failures: list[str] = []
    _check_coverage_floors(base, failures)
    _check_electron_coverage_floors(base, failures)
    _check_mutation_floor(base, failures)
    _check_context_budget(base, failures)
    _check_pyproject(base, failures)
    _check_diff_coverage_floor(base, failures)
    _check_semgrep_rules(base, failures)

    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(
            f"\n{len(failures)} threshold(s) moved the wrong way against {base}. "
            "Satisfy the gate instead of relaxing it; if the change is "
            "deliberate, say so explicitly in the PR and get it reviewed."
        )
        return 1

    print(f"ratchet: no threshold weakened against {base}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
