.DEFAULT_GOAL := ci

SEMGREP_IMAGE := semgrep/semgrep@sha256:bdf7013b2c3634a487671158da77c554f531742326b543a9464d2adf6c433ac8
PYTHON_SOURCES := tagalong.py tagalong

# Parallelize independent gate recipes by default. Hosted CI already splits the
# same groups across jobs; this is the local equivalent. Set `JOBS=N`
# (`JOBS=1` for serial logs) to override this default consistently across the
# older GNU Make shipped by macOS and current Linux Make. Prefer nproc on Linux
# and fall back to POSIX getconf (available on macOS).
CPU_CORES := $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)
ifeq ($(CPU_CORES),)
$(error unable to detect CPU count; pass -jN explicitly.)
endif
# Locally, leave a fifth of the machine to everything else: a gate run that
# takes every core makes the desktop it is running on unusable. Floored so this
# never rounds up past the budget, and clamped to 1 so a single-core box still
# runs. A hosted runner is dedicated and small (4 vCPU), where giving up a core
# buys nothing and costs the mutation lane wall-clock -- so CI takes all of
# them. `CI` is set by GitHub Actions and by every other CI worth the name;
# export CI=1 to get the hosted behaviour locally.
ifeq ($(CI),)
CI_JOBS := $(shell jobs=$$(( $(CPU_CORES) * 4 / 5 )); test "$$jobs" -ge 1 || jobs=1; echo "$$jobs")
else
CI_JOBS := $(CPU_CORES)
endif
JOBS ?= $(CI_JOBS)
MAKEFLAGS += -j$(JOBS)

# Keep the local gate and its hosted lanes defined from the same lists. The
# orchestration gate rejects a dropped target or a hosted lane that stops
# invoking one of these groups.
VERIFY_QUICK := format-check lint types test-integrity context-budget worker-threads ratchet shellcheck workflows orchestration catalog electron-actions
VERIFY_COVERAGE := test-coverage
VERIFY_MUTATION := mutation
VERIFY_ELECTRON := electron-typecheck electron-lint electron-format-check electron-coverage
VERIFY_SECURITY := security-static

# Lines this change touches must be tested even where the file's own floor is
# still low. Overridable so a stacked branch can compare against its base.
#
# Held above the project's own coverage so new code cannot be worse than what
# is already here; below 100 because main() is deliberately untested wiring.
# When it fires, the answer is usually to extract the logic out of main, not
# to waive the check. `make ratchet` refuses to let it fall.
DIFF_BASE ?= origin/master
DIFF_COVERAGE_MIN ?= 90

# Thresholds are compared against this ref so a lowered floor fails the build
# instead of relying on a reviewer noticing the diff.
RATCHET_BASE ?= origin/master

.PHONY: format format-check lint types test test-coverage macos-test-coverage diff-coverage verify-regression mutation test-integrity context-budget worker-threads ratchet semgrep security-static secrets security shellcheck workflows orchestration catalog electron-actions electron-actions-write electron-install electron-typecheck electron-lint electron-format-check electron-test electron-coverage verify-quick verify-coverage verify-mutation verify-electron verify-security verify ci ci-hosted ci-macos hooks hook-check smoke-real start start-tui start-ui start-ui-tui

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

types:
	uv run ty check

# -n on the command line wins over the pyproject addopts default, so the job
# budget governs the gate while a bare `uv run pytest` keeps that default.
test:
	uv run pytest -n $(CI_JOBS)

test-coverage:
	uv run pytest -n $(CI_JOBS) --cov --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=json:coverage.json
	uv run python tools/coverage_gate.py

# The Semgrep planted-violation tests execute the digest-pinned Linux
# container and stay in the Linux coverage lane. This lane still measures all
# application code while leaving container policy to verify-security.
macos-test-coverage:
	uv run pytest -n $(CI_JOBS) -m "not container_security" --cov --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=json:coverage.json
	uv run python tools/coverage_gate.py

diff-coverage:
	uv run diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFF_COVERAGE_MIN) --show-uncovered

verify-regression:
	@test -n "$(TEST)" || { echo "usage: make verify-regression TEST=tests/test_x.py::test_y"; exit 2; }
	tools/verify_regression.sh "$(TEST)"

# Without --max-children mutmut forks os.cpu_count() children, ignoring the
# budget entirely. Its own pytest runs -n0 (see pyproject), so these children
# are the whole of this gate's parallelism.
mutation:
	uv run mutmut run --max-children $(CI_JOBS)
	uv run mutmut export-cicd-stats
	uv run python tools/mutation_gate.py

semgrep:
	mkdir -p reports
	docker run --rm --network none --env SEMGREP_ENABLE_VERSION_CHECK=0 --env SEMGREP_SEND_METRICS=off --volume "$(CURDIR):/src:ro" --volume "$(CURDIR)/reports:/reports" --workdir /src "$(SEMGREP_IMAGE)" semgrep scan --config semgrep.yml --error --metrics=off --json-output /reports/semgrep.json

security-static: semgrep
	mkdir -p reports
	uv run bandit --recursive --configfile pyproject.toml --format json --output reports/bandit.json --exit-zero $(PYTHON_SOURCES)
	uv run bandit --recursive --configfile pyproject.toml --severity-level medium --confidence-level medium $(PYTHON_SOURCES)
	# Audit the locked dependency set rather than the environment. Since the
	# project became installable, `uv sync` puts it in the venv as an editable
	# install, and an environment audit fails on it either way: --strict treats
	# a distribution missing from PyPI as an error, and --skip-editable turns
	# that into a skip, which --strict also refuses. --no-emit-project leaves
	# it out at the source. What gets audited is then exactly what CI installs.
	uv export --all-groups --no-emit-project --no-hashes --quiet -o reports/requirements.txt
	# uv-managed CPython on macOS copies the interpreter into pip-audit's temp
	# venv with @rpath relative to that venv. Without the real libpython on the
	# dyld fallback path, ensurepip SIGABRTs and the audit never runs.
	DYLD_FALLBACK_LIBRARY_PATH="$(shell dirname $$(dirname $$(readlink $(CURDIR)/.venv/bin/python)))/lib" \
		uv run pip-audit --strict -r reports/requirements.txt

secrets:
	gitleaks detect --source . --log-opts="--all"

test-integrity:
	uv run python tools/test_integrity.py

context-budget:
	uv run python tools/context_budget.py

worker-threads:
	uv run python tools/worker_gate.py

ratchet:
	uv run python tools/ratchet_gate.py $(RATCHET_BASE)

shellcheck:
	bash -n fix-codex-sandbox.sh
	bash -n tools/verify_regression.sh
	bash -n tools/start.sh

workflows:
	@workflow_file="$$(find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -print -quit)"; \
		test -n "$$workflow_file" || { echo "error: .github/workflows holds no YAML workflows to lint."; exit 1; }; \
		find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -exec uv run actionlint {} +

orchestration:
	uv run python tools/orchestration_gate.py

catalog:
	uv run python tools/catalog_gate.py

# Regenerate-and-diff: committed actions.ts must match CATALOG (needs uv).
electron-actions:
	uv run python tools/electron_actions_gate.py

electron-actions-write:
	uv run python tools/electron_actions_gate.py --write

# Bun toolchain for electron/. Skip the Electron binary download in CI and
# local gates — types still install; unit tests never launch a window.
electron-install:
	cd electron && ELECTRON_SKIP_BINARY_DOWNLOAD=1 bun install --frozen-lockfile

electron-typecheck: electron-install
	cd electron && bun run typecheck

electron-lint: electron-install
	cd electron && bun run lint

electron-format-check: electron-install
	cd electron && bun run format-check

electron-test: electron-install
	cd electron && bun test

# Per-file floors under electron/src (bun lcov + tools/coverage_gate.ts).
# Replaces plain electron-test in VERIFY_ELECTRON — the suite still runs.
electron-coverage: electron-install
	cd electron && bun run test-coverage
	cd electron && bun run coverage-gate

verify-quick: $(VERIFY_QUICK)

verify-coverage: $(VERIFY_COVERAGE)

verify-mutation: $(VERIFY_MUTATION)

verify-electron: $(VERIFY_ELECTRON)

verify-security: $(VERIFY_SECURITY)

security: security-static secrets

verify: verify-quick verify-coverage verify-mutation verify-electron

ci: verify security

ci-hosted: verify verify-security

ci-macos: verify-quick macos-test-coverage verify-electron

hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

hook-check:
	uv run pre-commit run --all-files
	uv run pre-commit run --hook-stage pre-push --all-files

# Explicitly opt in: this records from the default microphone, reaches local
# audio services and Codex, and may download Piper/Edge voice data.
smoke-real:
	uv run pytest smoke_tests --no-cov -q

# Local launchers, in no gate: they open a real window and a real microphone.
# Each is one recipe rather than a target with prerequisites -- the -j default
# above would otherwise start Electron before the session owns a socket.
#
# DEV=1 (default) gives Electron hot reload: renderer/ soft-reloads, src/
# rebuilds and restarts. DEV=0 uses the plain launch path. Python changes need
# a session restart either way.
DEV ?= 1

# The everyday one. A single prerequisite, so the -j default has nothing to
# reorder; the default goal stays `ci`, so a bare `make` is still the gate.
start: start-ui

start-tui:
	DEV=$(DEV) tools/start.sh tui

start-ui:
	DEV=$(DEV) tools/start.sh ui

start-ui-tui:
	DEV=$(DEV) tools/start.sh ui-tui
