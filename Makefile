.DEFAULT_GOAL := ci

.PHONY: format format-check lint types test test-coverage security-static secrets security shellcheck workflows verify ci ci-hosted hooks hook-check

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
	uv run pytest --cov --cov-report=term-missing --cov-report=xml:coverage.xml

security-static:
	mkdir -p reports
	uv run bandit --configfile pyproject.toml --format json --output reports/bandit.json --exit-zero voice-codex.py voice-codex-tui.py
	uv run bandit --configfile pyproject.toml --severity-level medium --confidence-level medium voice-codex.py voice-codex-tui.py
	uv run pip-audit --strict

secrets:
	gitleaks detect --source . --log-opts="--all"

shellcheck:
	bash -n fix-codex-sandbox.sh

workflows:
	uv run actionlint .github/workflows/ci.yml

security: security-static secrets

verify: format-check lint types test-coverage shellcheck workflows

ci: verify security

ci-hosted: verify security-static

hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

hook-check:
	uv run pre-commit run --all-files
	uv run pre-commit run --hook-stage pre-push --all-files
