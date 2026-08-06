.DEFAULT_GOAL := ci

SEMGREP_IMAGE := semgrep/semgrep@sha256:bdf7013b2c3634a487671158da77c554f531742326b543a9464d2adf6c433ac8
PYTHON_SOURCES := tagalong.py tagalong

# Parallelize independent gate recipes by default. Hosted CI already splits the
# same groups across jobs; this is the local equivalent. A command-line `-jN`
# (including `-j1` for serial logs) overrides this default. Linux-only via
# nproc — other platforms must pass -jN explicitly rather than get a silent
# wrong guess.
CI_JOBS := $(shell nproc 2>/dev/null)
ifeq ($(CI_JOBS),)
$(error nproc unavailable; TagAlong's parallel make defaults expect Linux. Pass -jN explicitly.)
endif
MAKEFLAGS += -j$(CI_JOBS)

# Keep the local gate and its hosted lanes defined from the same lists. The
# orchestration gate rejects a dropped target or a hosted lane that stops
# invoking one of these groups.
VERIFY_QUICK := format-check lint types test-integrity context-budget worker-threads ratchet shellcheck workflows orchestration catalog
VERIFY_COVERAGE := test-coverage
VERIFY_MUTATION := mutation
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

.PHONY: format format-check lint types test test-coverage diff-coverage verify-regression mutation test-integrity context-budget worker-threads ratchet semgrep security-static secrets security shellcheck workflows orchestration catalog verify-quick verify-coverage verify-mutation verify-security verify ci ci-hosted hooks hook-check smoke-real

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

types:
	uv run ty check

test:
	uv run pytest

test-coverage:
	uv run pytest --cov --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=json:coverage.json
	uv run python tools/coverage_gate.py

diff-coverage:
	uv run diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFF_COVERAGE_MIN) --show-uncovered

verify-regression:
	@test -n "$(TEST)" || { echo "usage: make verify-regression TEST=tests/test_x.py::test_y"; exit 2; }
	tools/verify_regression.sh "$(TEST)"

mutation:
	uv run mutmut run
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

workflows:
	@workflow_file="$$(find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -print -quit)"; \
		test -n "$$workflow_file" || { echo "error: .github/workflows holds no YAML workflows to lint."; exit 1; }; \
		find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -exec uv run actionlint {} +

orchestration:
	uv run python tools/orchestration_gate.py

catalog:
	uv run python tools/catalog_gate.py

verify-quick: $(VERIFY_QUICK)

verify-coverage: $(VERIFY_COVERAGE)

verify-mutation: $(VERIFY_MUTATION)

verify-security: $(VERIFY_SECURITY)

security: security-static secrets

verify: verify-quick verify-coverage verify-mutation

ci: verify security

ci-hosted: verify verify-security

hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

hook-check:
	uv run pre-commit run --all-files
	uv run pre-commit run --hook-stage pre-push --all-files

# Explicitly opt in: this records from the default microphone, reaches local
# audio services and Codex, and may download Piper/Edge voice data.
smoke-real:
	uv run pytest smoke_tests --no-cov -q
