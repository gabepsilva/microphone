"""The quality gates must reject known-bad input.

A gate that matches nothing is worse than no gate: it reports green and is
believed. This is not hypothetical here — the Semgrep rule forbidding tests
from faking the unit under test was inert on the day it was written, because
Semgrep's built-in ignore list excludes ``tests/``. It parsed, it reported
"7 rules run", and it could never have fired. That was caught by hand, which
is not a control.

Each test below plants a violation the gate is supposed to catch, and asserts
it is caught. The false-positive direction matters too: a detector that
rejects the assertion styles this suite legitimately uses would be silenced
within a week, so that is asserted as well.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


@pytest.fixture(autouse=True)
def outside_any_git_hook(monkeypatch):
    """Detach these tests from the environment of a git hook that runs them.

    Git exports ``GIT_DIR`` and ``GIT_INDEX_FILE`` to its hooks, and both
    outrank the working directory for every git that inherits them. The
    ratchet tests below build a temporary repository and ``chdir`` into it, so
    under the pre-commit hook that runs this suite they were reading and
    writing the real repository instead: ``git add -A`` staged into the index
    of the very commit being verified, and the ratchet compared the planted
    thresholds against real history rather than the planted base.

    That made the whole file fail whenever it ran from a hook — which is the
    one time these gates most need to be believable.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _load_gate(name: str):
    """Import a gate script by path; tools/ is deliberately not a package."""
    spec = importlib.util.spec_from_file_location(f"_gate_{name}", TOOLS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# tools/test_integrity.py
# --------------------------------------------------------------------------

ASSERTION_FREE = """
def test_looks_like_a_test():
    result = 2 + 2
    str(result)
"""

BARE_SKIP = """
import pytest

@pytest.mark.skip
def test_hidden():
    assert False
"""

SKIP_WITHOUT_ISSUE = """
import pytest

@pytest.mark.skipif(True, reason="flaky sometimes")
def test_hidden():
    assert False
"""

# Styles this suite really uses. None contains a bare `assert` statement.
INVERTED_ASSERTIONS = """
import pytest


def test_rejects_bad_config():
    with pytest.raises(RuntimeError, match="Unknown startup config key"):
        load_startup_config("x")


def test_import_does_not_load_the_adapter(monkeypatch):
    def prohibit(name):
        raise AssertionError("must not load")

    monkeypatch.setattr(module, "__getattr__", prohibit)
    importlib.import_module("tagalong.cli")
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("assertion-free test", ASSERTION_FREE),
        ("bare skip decorator", BARE_SKIP),
        ("skip with no issue reference", SKIP_WITHOUT_ISSUE),
    ],
)
def test_test_integrity_rejects_untrustworthy_tests(
    tmp_path, monkeypatch, label, source
) -> None:
    gate = _load_gate("test_integrity")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_planted.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(gate, "TESTS_DIR", tests_dir)

    assert gate.main() == 1, f"gate accepted a {label}"


def test_test_integrity_accepts_this_suites_inverted_assertion_styles(
    tmp_path, monkeypatch
) -> None:
    gate = _load_gate("test_integrity")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_planted.py").write_text(INVERTED_ASSERTIONS, encoding="utf-8")
    monkeypatch.setattr(gate, "TESTS_DIR", tests_dir)

    assert gate.main() == 0


# --------------------------------------------------------------------------
# tools/coverage_gate.py
# --------------------------------------------------------------------------


def _write_coverage(path: Path, files: dict[str, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "files": {
                    name: {"summary": {"percent_covered": percent}}
                    for name, percent in files.items()
                }
            }
        ),
        encoding="utf-8",
    )


def test_coverage_gate_rejects_a_file_below_its_floor(tmp_path, monkeypatch) -> None:
    gate = _load_gate("coverage_gate")
    report = tmp_path / "coverage.json"
    _write_coverage(report, {"tagalong/domain.py": 70.0})
    monkeypatch.setattr(gate, "COVERAGE_PATH", report)
    monkeypatch.setattr(gate, "FLOORS", {"tagalong/domain.py": 81.0})

    assert gate.main() == 1


def test_coverage_gate_rejects_an_untested_new_module(tmp_path, monkeypatch) -> None:
    """A new file must not inherit the monolith's low recorded floor."""
    gate = _load_gate("coverage_gate")
    report = tmp_path / "coverage.json"
    _write_coverage(report, {"tagalong/brand_new.py": 5.0})
    monkeypatch.setattr(gate, "COVERAGE_PATH", report)
    monkeypatch.setattr(gate, "FLOORS", {})

    assert gate.main() == 1


def test_coverage_gate_rejects_a_floor_for_a_file_that_vanished(
    tmp_path, monkeypatch
) -> None:
    """A floor pointing at nothing is the silent no-op this suite exists for."""
    gate = _load_gate("coverage_gate")
    report = tmp_path / "coverage.json"
    _write_coverage(report, {"tagalong/domain.py": 90.0})
    monkeypatch.setattr(gate, "COVERAGE_PATH", report)
    monkeypatch.setattr(gate, "FLOORS", {"tagalong/deleted.py": 50.0})

    assert gate.main() == 1


# --------------------------------------------------------------------------
# tools/mutation_gate.py
# --------------------------------------------------------------------------


def test_mutation_gate_rejects_a_dropped_score(tmp_path, monkeypatch) -> None:
    gate = _load_gate("mutation_gate")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"killed": 10, "survived": 90, "total": 100}))
    monkeypatch.setattr(gate, "STATS_PATH", stats)
    monkeypatch.setattr(gate, "MUTATION_SCORE_FLOOR", 42.0)

    assert gate.main() == 1


def test_mutation_gate_rejects_a_run_that_mutated_nothing(
    tmp_path, monkeypatch
) -> None:
    """Zero mutants would otherwise divide into a vacuous pass."""
    gate = _load_gate("mutation_gate")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"killed": 0, "survived": 0, "total": 0}))
    monkeypatch.setattr(gate, "STATS_PATH", stats)

    assert gate.main() == 1


# --------------------------------------------------------------------------
# tools/orchestration_gate.py and make workflows
# --------------------------------------------------------------------------


def _copied_orchestration_gate(tmp_path, monkeypatch):
    gate = _load_gate("orchestration_gate")
    makefile = tmp_path / "Makefile"
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    makefile.write_text((ROOT / "Makefile").read_text(encoding="utf-8"))
    workflow.write_text(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "MAKEFILE", makefile)
    monkeypatch.setattr(gate, "HOSTED_WORKFLOW", workflow)
    return gate, makefile, workflow


def test_orchestration_gate_accepts_the_local_and_hosted_contract(
    tmp_path, monkeypatch
) -> None:
    gate, _, _ = _copied_orchestration_gate(tmp_path, monkeypatch)

    assert gate.main() == 0


@pytest.mark.parametrize(
    ("label", "plant"),
    [
        (
            "required target dropped from its Make group",
            (
                "makefile",
                "VERIFY_QUICK := format-check lint types",
                "VERIFY_QUICK := format-check types",
            ),
        ),
        (
            "required Make group variable missing",
            (
                "makefile",
                "VERIFY_QUICK := format-check lint types",
                "VERIFY_FAST := format-check lint types",
            ),
        ),
        (
            "Make group target detached from its variable",
            (
                "makefile",
                "verify-quick: $(VERIFY_QUICK)",
                "verify-quick: lint",
            ),
        ),
        (
            "required group omitted from the local verify target",
            (
                "makefile",
                "verify: verify-quick verify-coverage verify-mutation",
                "verify: verify-quick verify-coverage",
            ),
        ),
        (
            "local make no longer defaults to parallel jobs",
            (
                "makefile",
                "MAKEFLAGS += -j$(CI_JOBS)",
                "# parallel Make default removed",
            ),
        ),
        (
            "local parallel default no longer pinned to Linux nproc",
            (
                "makefile",
                "CI_JOBS := $(shell nproc 2>/dev/null)",
                "CI_JOBS := 4",
            ),
        ),
        (
            "required group no longer invoked by a hosted lane",
            ("workflow", "run: make verify-mutation", "run: make mutation"),
        ),
        (
            "quality lane omitted from the protected aggregator",
            (
                "workflow",
                "needs: [quick, coverage, mutation, security-static]",
                "needs: [quick, coverage, security-static]",
            ),
        ),
        (
            "aggregator with no parseable needs list",
            (
                "workflow",
                "needs: [quick, coverage, mutation, security-static]",
                "depends-on: [quick, coverage, mutation, security-static]",
            ),
        ),
        (
            "aggregator that can skip instead of reporting a result",
            ("workflow", "if: ${{ always() }}", "if: ${{ !cancelled() }}"),
        ),
        (
            "aggregator that does not positively require success",
            (
                "workflow",
                'test "$LANE_RESULTS" = "success success success success"',
                'test "$LANE_RESULTS" != "failure"',
            ),
        ),
        (
            "workflow whose jobs section cannot be parsed",
            ("workflow", "jobs:\n", "tasks:\n"),
        ),
        (
            "renamed protected aggregator check",
            (
                "workflow",
                "name: Quality and security",
                "name: Quality summary",
            ),
        ),
        (
            "aggregator that does not collect every dependency result",
            (
                "workflow",
                "LANE_RESULTS: ${{ join(needs.*.result, ' ') }}",
                "LANE_RESULTS: success",
            ),
        ),
        (
            "missing protected aggregator job",
            ("workflow", "  quality-gate:\n", "  quality-summary:\n"),
        ),
        (
            "quality lane with expression-valued continue-on-error",
            (
                "workflow",
                "  mutation:\n",
                "  mutation:\n    continue-on-error: ${{ true }}\n",
            ),
        ),
        (
            "quality lane step declaring continue-on-error",
            (
                "workflow",
                "      - name: Run mutation gate\n        run: make verify-mutation",
                "      - name: Run mutation gate\n"
                "        continue-on-error: false\n"
                "        run: make verify-mutation",
            ),
        ),
        (
            "protected aggregator declaring continue-on-error",
            (
                "workflow",
                "  quality-gate:\n",
                "  quality-gate:\n    continue-on-error: true\n",
            ),
        ),
        (
            "protected secret scan declaring continue-on-error",
            (
                "workflow",
                "  secret-scan:\n",
                "  secret-scan:\n    continue-on-error: true\n",
            ),
        ),
    ],
)
def test_orchestration_gate_rejects_a_planted_omission(
    tmp_path, monkeypatch, label, plant
) -> None:
    gate, makefile, workflow = _copied_orchestration_gate(tmp_path, monkeypatch)
    path_name, before, after = plant
    path = makefile if path_name == "makefile" else workflow
    source = path.read_text(encoding="utf-8")
    assert source.count(before) == 1
    path.write_text(source.replace(before, after), encoding="utf-8")

    assert gate.main() == 1, f"orchestration gate accepted a {label}"


@pytest.mark.parametrize("missing", ["makefile", "workflow"])
def test_orchestration_gate_rejects_a_missing_policy_file(
    tmp_path, monkeypatch, missing
) -> None:
    gate, makefile, workflow = _copied_orchestration_gate(tmp_path, monkeypatch)
    path = makefile if missing == "makefile" else workflow
    path.unlink()

    assert gate.main() == 1


def test_workflow_gate_rejects_a_directory_with_no_workflows(tmp_path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)

    result = subprocess.run(
        ["make", "-f", str(ROOT / "Makefile"), "workflows"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "holds no YAML workflows to lint" in result.stdout


def test_workflow_gate_lints_every_yaml_workflow(tmp_path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    nested = workflows / "scheduled"
    nested.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    (nested / "audit.yaml").write_text("name: Audit\n", encoding="utf-8")
    (workflows / "notes.txt").write_text("not a workflow\n", encoding="utf-8")

    arguments = tmp_path / "actionlint-arguments"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$WORKFLOW_ARGUMENTS"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["WORKFLOW_ARGUMENTS"] = str(arguments)

    result = subprocess.run(
        ["make", "-f", str(ROOT / "Makefile"), "workflows"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    linted = set(arguments.read_text(encoding="utf-8").splitlines())
    assert "run" in linted
    assert "actionlint" in linted
    assert ".github/workflows/ci.yml" in linted
    assert ".github/workflows/scheduled/audit.yaml" in linted
    assert ".github/workflows/notes.txt" not in linted


# --------------------------------------------------------------------------
# tools/verify_regression.sh
# --------------------------------------------------------------------------


def _regression_repository(tmp_path: Path, uv_body: str) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    source = repo / "tagalong" / "behavior.py"
    source.parent.mkdir(parents=True)
    source.write_text('STATE = "broken"\n', encoding="utf-8")
    (repo / "tagalong.py").write_text(
        "from tagalong.behavior import STATE\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(
        ["git", "-C", repo, "config", "user.email", "quality@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "config", "user.name", "Quality Gate"],
        check=True,
    )
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repo, "commit", "-qm", "broken implementation"], check=True
    )
    source.write_text('STATE = "fixed"\n', encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f"#!/bin/sh\n{uv_body}\n", encoding="utf-8")
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return repo, env


def test_regression_verification_observes_fail_then_pass_and_restores_the_fix(
    tmp_path,
) -> None:
    repo, env = _regression_repository(
        tmp_path,
        "grep -q '\"fixed\"' tagalong/behavior.py",
    )

    result = subprocess.run(
        [TOOLS / "verify_regression.sh", "tests/test_behavior.py::test_fix"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "good: the test failed without the fix" in result.stdout
    assert "regression verified" in result.stdout
    assert (repo / "tagalong" / "behavior.py").read_text(
        encoding="utf-8"
    ) == 'STATE = "fixed"\n'
    assert (
        subprocess.run(
            ["git", "-C", repo, "stash", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_regression_verification_rejects_a_test_that_passes_without_the_fix(
    tmp_path,
) -> None:
    repo, env = _regression_repository(tmp_path, "exit 0")

    result = subprocess.run(
        [TOOLS / "verify_regression.sh", "tests/test_behavior.py::test_fix"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "test passed without the fix" in result.stdout
    assert (repo / "tagalong" / "behavior.py").read_text(
        encoding="utf-8"
    ) == 'STATE = "fixed"\n'


# --------------------------------------------------------------------------
# tools/context_budget.py
# --------------------------------------------------------------------------


def test_context_budget_rejects_an_oversized_instruction_file(
    tmp_path, monkeypatch
) -> None:
    gate = _load_gate("context_budget")
    (tmp_path / "AGENTS.md").write_text("word " * 500, encoding="utf-8")
    monkeypatch.setattr(gate, "BUDGETS", {"AGENTS.md": 400})
    monkeypatch.chdir(tmp_path)

    assert gate.main() == 1


def test_context_budget_accepts_a_file_within_budget(tmp_path, monkeypatch) -> None:
    gate = _load_gate("context_budget")
    (tmp_path / "AGENTS.md").write_text("word " * 300, encoding="utf-8")
    monkeypatch.setattr(gate, "BUDGETS", {"AGENTS.md": 400})
    monkeypatch.chdir(tmp_path)

    assert gate.main() == 0


def test_context_budget_rejects_a_budgeted_file_that_vanished(
    tmp_path, monkeypatch
) -> None:
    """Deleting the file is not a way to satisfy its budget."""
    gate = _load_gate("context_budget")
    monkeypatch.setattr(gate, "BUDGETS", {"AGENTS.md": 400})
    monkeypatch.chdir(tmp_path)

    assert gate.main() == 1


# --------------------------------------------------------------------------
# tools/worker_gate.py
# --------------------------------------------------------------------------

NON_DAEMON_THREAD = """
import threading


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run)
        self.worker.start()
"""

UNBOUNDED_JOIN = """
import threading


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run, daemon=True)

    def close(self):
        self.worker.join()
"""

EXPLICIT_NONE_JOIN_TIMEOUT = """
import threading


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run, daemon=True)

    def close(self):
        self.worker.join(timeout=None)
"""

POSITIONAL_NONE_JOIN_TIMEOUT = """
import threading


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run, daemon=True)

    def close(self):
        self.worker.join(None)
"""

UNPACKED_JOIN_KWARGS = """
import threading


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run, daemon=True)

    def close(self, **options):
        self.worker.join(**options)
"""

ANNOTATED_NON_DAEMON_THREAD = """
import threading


class Capture:
    def __init__(self):
        self.worker: threading.Thread = threading.Thread(target=self._run)
        self.worker.start()

    def close(self):
        self.worker.join()
"""

INLINE_NON_DAEMON_THREAD = """
import threading


def run_session(tui):
    threading.Thread(target=populate, args=(tui,)).start()
"""

NON_DAEMON_TIMER = """
import threading


class Listener:
    def _start_timer(self):
        self.timer = threading.Timer(3.0, self._flush)
        self.timer.start()
"""

# Spellings this package really uses. Timer takes no daemon keyword, and
# str.join is everywhere; neither may be flagged.
CONVENTION_FOLLOWING_WORKERS = """
import threading


class Listener:
    def _start_timer(self):
        self.timer = threading.Timer(3.0, self._flush)
        self.timer.daemon = True
        self.timer.start()


class Capture:
    def __init__(self):
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.drain: threading.Thread = threading.Thread(target=self._drain)
        self.drain.daemon = True

    def close(self):
        self.reader.join(timeout=3)
        self.worker.join(timeout=10)
        self.drain.join(5.0)

    def _text(self, words):
        return " ".join(word.strip() for word in words)


def run_session(tui):
    threading.Thread(target=populate, args=(tui,), daemon=True).start()
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("non-daemon worker thread", NON_DAEMON_THREAD),
        ("join with no timeout", UNBOUNDED_JOIN),
        ("inline non-daemon thread", INLINE_NON_DAEMON_THREAD),
        ("non-daemon timer", NON_DAEMON_TIMER),
        ("join with an explicit None timeout", EXPLICIT_NONE_JOIN_TIMEOUT),
        ("join with a positional None timeout", POSITIONAL_NONE_JOIN_TIMEOUT),
        ("annotated non-daemon thread", ANNOTATED_NON_DAEMON_THREAD),
        ("join whose timeout is hidden in **kwargs", UNPACKED_JOIN_KWARGS),
    ],
)
def test_worker_gate_rejects_a_thread_that_can_outlive_shutdown(label, source) -> None:
    gate = _load_gate("worker_gate")

    assert gate.check_source(source, Path("planted.py")), f"gate accepted a {label}"


def test_worker_gate_accepts_the_spellings_this_package_uses() -> None:
    """A gate that rejects Timer or str.join would be deleted within a week."""
    gate = _load_gate("worker_gate")

    assert gate.check_source(CONVENTION_FOLLOWING_WORKERS, Path("planted.py")) == []


def test_worker_gate_fails_when_it_scans_a_package_that_is_not_there(
    tmp_path, monkeypatch
) -> None:
    """Scanning nothing must not be reported as a clean run."""
    gate = _load_gate("worker_gate")
    monkeypatch.setattr(gate, "PACKAGE", tmp_path / "tagalong")
    monkeypatch.chdir(tmp_path)

    assert gate.main() == 1


def test_worker_gate_reports_a_planted_violation_through_main(
    tmp_path, monkeypatch
) -> None:
    gate = _load_gate("worker_gate")
    package = tmp_path / "tagalong"
    package.mkdir()
    (package / "capture.py").write_text(NON_DAEMON_THREAD, encoding="utf-8")
    monkeypatch.setattr(gate, "PACKAGE", package)
    monkeypatch.chdir(tmp_path)

    assert gate.main() == 1


def test_worker_gate_passes_on_this_repository() -> None:
    """The convention it enforces is the one the package already follows."""
    gate = _load_gate("worker_gate")
    monkeypatch_free_failures = [
        failure
        for path in sorted((ROOT / "tagalong").rglob("*.py"))
        for failure in gate.check_source(path.read_text(encoding="utf-8"), path)
    ]

    assert monkeypatch_free_failures == []


# --------------------------------------------------------------------------
# tools/catalog_gate.py
# --------------------------------------------------------------------------


def test_catalog_gate_passes_on_this_repository() -> None:
    gate = _load_gate("catalog_gate")
    assert gate.main() == 0


def test_catalog_gate_rejects_a_missing_handler() -> None:
    gate = _load_gate("catalog_gate")
    registered = gate.production_handler_ids() - {"session.quit"}
    problems = gate.check(registered)
    assert any(
        "session.quit" in problem and "no handler" in problem for problem in problems
    )


def test_catalog_gate_rejects_a_skipped_composition_binder() -> None:
    gate = _load_gate("catalog_gate")
    called = gate.composition_binder_ids() - {"bind_audio_slice"}
    problems = gate.check(binders=called)
    assert any(
        "bind_audio_slice" in problem and "composition root" in problem
        for problem in problems
    )


def test_catalog_gate_rejects_a_stale_deferral() -> None:
    gate = _load_gate("catalog_gate")
    problems = gate.check(
        gate.production_handler_ids(), deferred=frozenset({"session.quit"})
    )
    assert any("deferred but a handler" in problem for problem in problems)


def test_catalog_gate_rejects_an_unknown_deferral() -> None:
    gate = _load_gate("catalog_gate")
    problems = gate.check(
        gate.production_handler_ids(), deferred=frozenset({"session.explode"})
    )
    assert any("not in the catalog" in problem for problem in problems)


def test_catalog_gate_reads_binder_calls_by_name() -> None:
    gate = _load_gate("catalog_gate")
    source = (
        "bind_first_slice(controller, conversation=c)\n"
        "mod.bind_audio_slice(controller)\n"
        "other(bind_settings_slice)\n"
    )
    assert gate.called_names(source) == frozenset(
        {"bind_first_slice", "bind_audio_slice", "other"}
    )


def test_catalog_gate_handler_ids_come_from_runtime_registration() -> None:
    """A register call that never runs must not satisfy the gate."""
    gate = _load_gate("catalog_gate")
    wired = gate.production_handler_ids()
    assert "microphone.set_muted" in wired
    assert len(wired) == len(gate.CATALOG)


def test_catalog_gate_main_reports_planted_failures(monkeypatch, capsys) -> None:
    gate = _load_gate("catalog_gate")
    monkeypatch.setattr(gate, "check", lambda: ["session.quit: planted"])
    assert gate.main() == 1
    captured = capsys.readouterr()
    assert "session.quit: planted" in captured.err


# --------------------------------------------------------------------------
# tools/ratchet_gate.py
# --------------------------------------------------------------------------


def _fake_repo(tmp_path: Path, base_files: dict[str, str]) -> Path:
    """A throwaway git repo whose HEAD holds `base_files`.

    The ratchet compares the working tree against a ref, so exercising it needs
    real git history rather than a fake — git is the boundary here, not the
    subject.
    """
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    for name, content in base_files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content, encoding="utf-8")

    # Git exports GIT_DIR, GIT_INDEX_FILE, and a blank GIT_AUTHOR_NAME to the
    # hooks it runs, and every one of them outranks what this repository sets
    # for itself: inherited, the commit below lands in the real repository or
    # is refused for having no author. That makes these tests pass from a
    # terminal and fail from the pre-commit hook, which is the one place the
    # gates most need to run.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    run = lambda *args: subprocess.run(  # noqa: E731 - terse local helper
        args, cwd=repo, check=True, capture_output=True, env=environment
    )
    run("git", "init", "--quiet")
    run("git", "config", "user.email", "gate@example.invalid")
    run("git", "config", "user.name", "gate")
    run("git", "add", "-A")
    run("git", "commit", "--quiet", "-m", "base")
    return repo


BASE_COVERAGE_GATE = 'FLOORS = {"tagalong/domain.py": 81.0}\nNEW_FILE_FLOOR = 60.0\n'
BASE_MUTATION_GATE = "MUTATION_SCORE_FLOOR = 42.0\n"
BASE_CONTEXT_BUDGET = (
    'BUDGETS = {"AGENTS.md": 400}\n'
    'BUDGET_RAISES = [("AGENTS.md", 300, 400, "an earlier, already-spent raise")]\n'
)
BASE_PYPROJECT = """
[tool.coverage.report]
fail_under = 46

[tool.mutmut]
source_paths = ["tagalong/domain.py"]
"""
BASE_SEMGREP = "rules:\n  - id: python-subprocess-shell-true\n"
BASE_MAKEFILE = "DIFF_BASE ?= origin/master\nDIFF_COVERAGE_MIN ?= 90\n"

BASE_FILES = {
    "tools/coverage_gate.py": BASE_COVERAGE_GATE,
    "tools/mutation_gate.py": BASE_MUTATION_GATE,
    "tools/context_budget.py": BASE_CONTEXT_BUDGET,
    "pyproject.toml": BASE_PYPROJECT,
    "semgrep.yml": BASE_SEMGREP,
    "Makefile": BASE_MAKEFILE,
    # The module the coverage floor and the mutmut scope both name. It has to
    # exist for "dropped while the file is still there" to be the thing under
    # test; when it is missing, dropping it is legitimate and allowed.
    "tagalong/domain.py": 'VOICE = "Voice"\n',
}


@pytest.mark.parametrize(
    ("label", "path", "tampered"),
    [
        (
            "lowered per-file coverage floor",
            "tools/coverage_gate.py",
            'FLOORS = {"tagalong/domain.py": 40.0}\nNEW_FILE_FLOOR = 60.0\n',
        ),
        (
            "lowered new-file floor",
            "tools/coverage_gate.py",
            'FLOORS = {"tagalong/domain.py": 81.0}\nNEW_FILE_FLOOR = 10.0\n',
        ),
        (
            "lowered mutation floor",
            "tools/mutation_gate.py",
            "MUTATION_SCORE_FLOOR = 5.0\n",
        ),
        (
            "lowered global coverage threshold",
            "pyproject.toml",
            "\n[tool.coverage.report]\nfail_under = 5\n\n[tool.mutmut]\n"
            'source_paths = ["tagalong/domain.py"]\n',
        ),
        (
            "narrowed mutmut scope",
            "pyproject.toml",
            "\n[tool.coverage.report]\nfail_under = 46\n\n[tool.mutmut]\n"
            "source_paths = []\n",
        ),
        (
            "deleted semgrep rule",
            "semgrep.yml",
            "rules:\n  - id: something-else\n",
        ),
        (
            "raised context budget with no recorded reason",
            "tools/context_budget.py",
            'BUDGETS = {"AGENTS.md": 4000}\nBUDGET_RAISES = []\n',
        ),
        (
            "raised context budget on an empty reason",
            "tools/context_budget.py",
            'BUDGETS = {"AGENTS.md": 4000}\n'
            'BUDGET_RAISES = [("AGENTS.md", 400, 4000, "   ")]\n',
        ),
        (
            "raised context budget reusing the previous raise as cover",
            "tools/context_budget.py",
            'BUDGETS = {"AGENTS.md": 4000}\n'
            'BUDGET_RAISES = [("AGENTS.md", 300, 400, "an earlier, '
            'already-spent raise")]\n',
        ),
        (
            "raised context budget recording a different jump than it made",
            "tools/context_budget.py",
            'BUDGETS = {"AGENTS.md": 4000}\n'
            'BUDGET_RAISES = [("AGENTS.md", 400, 450, "understated")]\n',
        ),
        (
            "removed context budget",
            "tools/context_budget.py",
            "BUDGETS = {}\n",
        ),
        (
            "lowered diff-coverage floor",
            "Makefile",
            "DIFF_BASE ?= origin/master\nDIFF_COVERAGE_MIN ?= 0\n",
        ),
        (
            "deleted diff-coverage floor",
            "Makefile",
            "DIFF_BASE ?= origin/master\n",
        ),
    ],
)
def test_ratchet_rejects_a_weakened_threshold(
    tmp_path, monkeypatch, label, path, tampered
) -> None:
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    (repo / path).write_text(tampered, encoding="utf-8")
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 1, f"ratchet allowed a {label}"


def test_ratchet_allows_a_raised_floor(tmp_path, monkeypatch) -> None:
    """Improving a threshold must never be blocked, or the ratchet stops turning."""
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    (repo / "tools/coverage_gate.py").write_text(
        'FLOORS = {"tagalong/domain.py": 95.0}\nNEW_FILE_FLOOR = 80.0\n',
        encoding="utf-8",
    )
    (repo / "Makefile").write_text(
        "DIFF_BASE ?= origin/master\nDIFF_COVERAGE_MIN ?= 100\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 0


def test_ratchet_allows_a_mutated_module_to_be_renamed(tmp_path, monkeypatch) -> None:
    """Renaming a package is not narrowing the mutmut scope.

    The comparison is stringwise, so every path changes at once and the base
    branch still spells the old prefix. The same allowance the coverage floors
    already have applies: a threshold may be dropped when its file is gone.
    """
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    (repo / "renamed").mkdir()
    (repo / "tagalong/domain.py").rename(repo / "renamed/domain.py")
    (repo / "tagalong").rmdir()
    (repo / "pyproject.toml").write_text(
        "\n[tool.coverage.report]\nfail_under = 46\n\n[tool.mutmut]\n"
        'source_paths = ["renamed/domain.py"]\n',
        encoding="utf-8",
    )
    (repo / "tools/coverage_gate.py").write_text(
        'FLOORS = {"renamed/domain.py": 81.0}\nNEW_FILE_FLOOR = 60.0\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 0


def test_ratchet_still_rejects_dropping_a_module_that_is_still_there(
    tmp_path, monkeypatch
) -> None:
    """The guard must not become a way to quietly stop mutating live code."""
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    (repo / "pyproject.toml").write_text(
        "\n[tool.coverage.report]\nfail_under = 46\n\n[tool.mutmut]\n"
        "source_paths = []\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 1


@pytest.mark.parametrize(
    ("label", "base_makefile"),
    [
        ("the base has no floor to compare against", "DIFF_BASE ?= origin/master\n"),
        ("the base has no Makefile at all", None),
    ],
)
def test_ratchet_allows_a_threshold_that_did_not_exist_before(
    tmp_path, monkeypatch, label, base_makefile
) -> None:
    """Introducing a floor must not be mistaken for weakening one."""
    gate = _load_gate("ratchet_gate")
    base_files = dict(BASE_FILES)
    if base_makefile is None:
        del base_files["Makefile"]
    else:
        base_files["Makefile"] = base_makefile
    repo = _fake_repo(tmp_path, base_files)
    (repo / "Makefile").write_text(
        "DIFF_BASE ?= origin/master\nDIFF_COVERAGE_MIN ?= 90\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 0, f"ratchet blocked {label}"


def test_ratchet_allows_a_budget_raise_that_records_its_reason(
    tmp_path, monkeypatch
) -> None:
    """The cap must be raisable, or a deliberate decision has nowhere to go."""
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    (repo / "tools/context_budget.py").write_text(
        'BUDGETS = {"AGENTS.md": 533}\n'
        "BUDGET_RAISES = [\n"
        '    ("AGENTS.md", 300, 400, "an earlier, already-spent raise"),\n'
        '    ("AGENTS.md", 400, 533, "owner asked for headroom"),\n'
        "]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "HEAD"]) == 0


def test_ratchet_fails_loudly_when_the_base_ref_is_missing(
    tmp_path, monkeypatch
) -> None:
    """A ratchet that cannot compare must not report success."""
    gate = _load_gate("ratchet_gate")
    repo = _fake_repo(tmp_path, BASE_FILES)
    monkeypatch.chdir(repo)

    assert gate.main(["ratchet_gate.py", "origin/does-not-exist"]) == 1


# --------------------------------------------------------------------------
# Scanner configuration
# --------------------------------------------------------------------------


def test_semgrep_scans_test_code() -> None:
    """Semgrep's built-in ignore list excludes tests/; the repo must override it.

    Without this file the test-integrity rule below is inert. It was.
    """
    semgrepignore = ROOT / ".semgrepignore"
    assert semgrepignore.exists(), "deleting .semgrepignore silently unscans tests/"

    excluded = {
        line.strip().strip("/")
        for line in semgrepignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "tests" not in excluded


def test_semgrep_forbids_faking_the_unit_under_test() -> None:
    rules = (ROOT / "semgrep.yml").read_text(encoding="utf-8")
    assert "python-test-fakes-the-unit-under-test" in rules
    assert "tagalong" in rules
    assert "domain" in rules


def test_mutmut_mutates_files_that_exist() -> None:
    """A typo in source_paths yields zero mutants and a vacuously perfect run."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source_paths = config["tool"]["mutmut"]["source_paths"]

    assert source_paths
    for path in source_paths:
        assert (ROOT / path).is_file(), f"mutmut mutates nothing: {path} is missing"


def test_mutmut_runs_the_reset_contract() -> None:
    """A new Codex-session path is meaningful only if mutation testing runs it."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    selected = config["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]

    assert "tests/test_new_session.py" in selected
