#!/usr/bin/env python3
"""Hold every background thread to the shutdown contract the app relies on.

Four classes here own a worker thread — TTS synthesis, the Them capture
reader, the transcript flush timer, and the Codex request loop. Each follows
the same unwritten rule: the thread is a daemon, and every join that waits on
it passes a timeout. Nothing enforced that. It survived because all four sat
in one file, so whoever added the fifth had the other four on screen.

That is the part a split breaks. Once the workers live in separate modules,
the convention is no longer visible from where the next one gets written, and
the failure it prevents is the worst kind to debug: a non-daemon thread or a
bare ``join()`` does not raise, does not fail a test, and does not lose
coverage. It hangs the process at exit, sometimes.

So the convention moves out of the code's layout and into a check:

  * ``threading.Thread(...)`` and ``threading.Timer(...)`` must be daemons,
    either by ``daemon=True`` at the call or by assigning ``.daemon = True``
    to the same target. Timer takes no daemon keyword, which is why both
    spellings are accepted.
  * ``join()`` on anything this file recognizes as a thread must pass a
    timeout. Shutdown must stay bounded even when a worker is wedged.

Thread identity is tracked per top-level class or function by the source text
of the assignment target, so ``" ".join(parts)`` is never mistaken for waiting
on a worker: only expressions assigned from a Thread or Timer call are
checked.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path("voice_codex")

THREAD_FACTORIES = {"Thread", "Timer"}


def _is_thread_call(node: ast.expr) -> bool:
    """Recognize ``threading.Thread(...)``, ``Timer(...)``, and their kin."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in THREAD_FACTORIES
    return isinstance(func, ast.Name) and func.id in THREAD_FACTORIES


def _has_daemon_keyword(call: ast.Call) -> bool:
    return any(
        keyword.arg == "daemon"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def _daemon_targets(scope: ast.AST) -> set[str]:
    """Collect every ``X.daemon = True`` target within one scope."""
    targets: set[str] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and node.value.value is True):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "daemon":
                targets.add(ast.unparse(target.value))
    return targets


def _thread_targets(scope: ast.AST, failures: list[str], path: Path) -> set[str]:
    """Check daemon status and return the expressions holding a thread."""
    daemonized = _daemon_targets(scope)
    threads: set[str] = set()

    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not _is_thread_call(node.value):
            continue
        call = node.value
        assert isinstance(call, ast.Call)
        for target in node.targets:
            name = ast.unparse(target)
            threads.add(name)
            if _has_daemon_keyword(call) or name in daemonized:
                continue
            failures.append(
                f"{path}:{node.lineno}: {name} is a thread that is not a daemon. "
                f"Pass daemon=True, or set {name}.daemon = True, so a wedged "
                f"worker cannot keep the process alive after the interface quits."
            )

    for node in ast.walk(scope):
        if isinstance(node, ast.Call):
            _check_inline_thread(node, failures, path)

    return threads


def _check_inline_thread(call: ast.Call, failures: list[str], path: Path) -> None:
    """A thread constructed and started in one expression is never joined.

    Nothing holds a reference to it, so daemon status is the only thing that
    keeps it from outliving the interface it was started for.
    """
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "start":
        return
    inner = call.func.value
    if not _is_thread_call(inner):
        return
    assert isinstance(inner, ast.Call)
    if not _has_daemon_keyword(inner):
        failures.append(
            f"{path}:{call.lineno}: an inline thread is started without "
            f"daemon=True. It is never joined, so only daemon status stops it "
            f"from outliving the process."
        )


def _check_joins(
    scope: ast.AST, threads: set[str], failures: list[str], path: Path
) -> None:
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "join":
            continue
        receiver = ast.unparse(func.value)
        if receiver not in threads:
            continue
        has_timeout = any(keyword.arg == "timeout" for keyword in node.keywords)
        if not has_timeout and not node.args:
            failures.append(
                f"{path}:{node.lineno}: {receiver}.join() waits without a "
                f"timeout. Shutdown must stay bounded even when the worker is "
                f"blocked on audio, a subprocess, or the network."
            )


def _check_scope(scope: ast.AST, failures: list[str], path: Path) -> None:
    threads = _thread_targets(scope, failures, path)
    _check_joins(scope, threads, failures, path)


def check_source(source: str, path: Path) -> list[str]:
    """Report every worker-thread convention violation in one module."""
    failures: list[str] = []
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            _check_scope(node, failures, path)

    module_level = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(
                node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            )
        ],
        type_ignores=[],
    )
    _check_scope(module_level, failures, path)
    return failures


def main() -> int:
    sources = sorted(PACKAGE.rglob("*.py"))
    if not sources:
        print(f"error: {PACKAGE} holds no modules to check.")
        return 1

    failures: list[str] = []
    for path in sources:
        failures.extend(check_source(path.read_text(encoding="utf-8"), path))

    for failure in failures:
        print(f"error: {failure}")

    if failures:
        print(f"\n{len(failures)} worker-thread convention failure(s).")
        return 1

    print(f"worker threads: {len(sources)} modules follow the shutdown contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
