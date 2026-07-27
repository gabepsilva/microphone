#!/usr/bin/env bash
# Prove a regression test actually detects the bug it claims to fix.
#
# An agent that writes the fix and the test in one pass produces a test that
# passes against whatever it just built, including a wrong implementation.
# The only evidence that a test detects the defect is watching it fail without
# the fix. This reverts the source change, requires the test to FAIL, restores
# the change, and requires it to PASS.
#
# Usage: make verify-regression TEST=tests/test_domain.py::test_echo_is_cut
#
# The test file itself is never stashed: reverting it would revert the test
# under examination.

set -euo pipefail

TEST_SELECTION="${1:?usage: verify_regression.sh <pytest-selection>}"
SOURCE_PATHS=(voice_codex voice-codex.py voice-codex-tui.py)

if git diff --quiet -- "${SOURCE_PATHS[@]}"; then
    echo "error: no uncommitted change under ${SOURCE_PATHS[*]}."
    echo "This check compares the working tree against HEAD, so the fix must"
    echo "still be uncommitted. For a committed fix, check out its parent."
    exit 2
fi

STASH_MESSAGE="verify-regression $$"
restore() {
    local stash_ref
    stash_ref="$(git stash list --format='%gd %gs' | awk -v m="$STASH_MESSAGE" '$0 ~ m {print $1; exit}')"
    if [ -n "$stash_ref" ]; then
        echo "==> restoring the fix ($stash_ref)"
        git stash pop "$stash_ref" >/dev/null
    fi
}
trap restore EXIT

echo "==> stashing the fix, keeping tests in place"
git stash push --quiet --message "$STASH_MESSAGE" -- "${SOURCE_PATHS[@]}"

echo "==> running $TEST_SELECTION WITHOUT the fix (must fail)"
if uv run pytest "$TEST_SELECTION" --no-cov -q; then
    echo
    echo "error: the test passed without the fix, so it does not detect the bug."
    echo "Assert on the behavior that was actually wrong."
    exit 1
fi
echo "==> good: the test failed without the fix"

restore
trap - EXIT

echo "==> running $TEST_SELECTION WITH the fix (must pass)"
uv run pytest "$TEST_SELECTION" --no-cov -q

echo
echo "regression verified: the test fails without the fix and passes with it."
