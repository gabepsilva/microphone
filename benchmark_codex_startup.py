#!/usr/bin/env python3
"""Measure Codex process startup repeatedly, without waiting for a model answer."""

from __future__ import annotations

import argparse
import json
import selectors
import statistics
import subprocess
import time
from pathlib import Path


def read_json_line(proc: subprocess.Popen[str], timeout: float) -> tuple[dict, float]:
    assert proc.stdout
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    ready = selector.select(timeout)
    selector.close()
    if not ready:
        raise TimeoutError("timed out waiting for Codex output")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"Codex exited before producing JSON (status {proc.poll()})")
    return json.loads(line), time.perf_counter()


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def app_server_startup(codex: str, cwd: Path, timeout: float) -> tuple[float, float]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        [codex, "app-server", "--listen", "stdio://"],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdin
        proc.stdin.write(
            json.dumps(
                {
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "clientInfo": {
                            "name": "codex_startup_benchmark",
                            "title": "Codex startup benchmark",
                            "version": "1.0.0",
                        }
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        initialized_seconds: float | None = None
        while initialized_seconds is None:
            message, received = read_json_line(proc, timeout)
            if message.get("id") == 1:
                initialized_seconds = received - started
        proc.stdin.write(json.dumps({"method": "initialized", "params": {}}) + "\n")
        proc.stdin.write(
            json.dumps(
                {
                    "method": "thread/start",
                    "id": 2,
                    "params": {
                        "cwd": str(cwd),
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "ephemeral": True,
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        while True:
            message, received = read_json_line(proc, timeout)
            if message.get("id") == 2:
                return initialized_seconds, received - started
    finally:
        terminate(proc)


def cli_startup(codex: str, cwd: Path, timeout: float) -> tuple[float, float]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        [
            codex,
            "-a",
            "never",
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-C",
            str(cwd),
            "Print the date and time now.",
        ],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        first_event, received = read_json_line(proc, timeout)
        first_event_seconds = received - started
        if first_event.get("type") == "thread.started":
            return first_event_seconds, first_event_seconds
        while True:
            event, received = read_json_line(proc, timeout)
            if event.get("type") == "thread.started":
                return first_event_seconds, received - started
    finally:
        terminate(proc)


def summary(values: list[float]) -> str:
    return (
        f"min={min(values):.3f}s max={max(values):.3f}s "
        f"mean={statistics.mean(values):.3f}s median={statistics.median(values):.3f}s "
        f"stdev={statistics.stdev(values) if len(values) > 1 else 0:.3f}s "
        f"range={(max(values) - min(values)):.3f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    app_initialize_values: list[float] = []
    app_thread_values: list[float] = []
    cli_first_values: list[float] = []
    cli_thread_values: list[float] = []
    cwd = args.cwd.resolve()
    for index in range(args.runs):
        app_initialize, app_thread = app_server_startup(args.codex, cwd, args.timeout)
        cli_first, cli_thread = cli_startup(args.codex, cwd, args.timeout)
        app_initialize_values.append(app_initialize)
        app_thread_values.append(app_thread)
        cli_first_values.append(cli_first)
        cli_thread_values.append(cli_thread)
        print(
            f"{index + 1:2d}/{args.runs}: app initialize={app_initialize:.3f}s, "
            f"app thread/start={app_thread:.3f}s, "
            f"CLI first JSON={cli_first:.3f}s, CLI thread.started={cli_thread:.3f}s",
            flush=True,
        )

    print("\nStartup summary")
    print(f"app-server to initialize:   {summary(app_initialize_values)}")
    print(f"app-server to thread/start: {summary(app_thread_values)}")
    print(f"CLI to first JSON event:   {summary(cli_first_values)}")
    print(f"CLI to thread.started:    {summary(cli_thread_values)}")
    print(
        "CLI/app-server thread-ready mean ratio: "
        f"{statistics.mean(cli_thread_values) / statistics.mean(app_thread_values):.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
