#!/usr/bin/env python3
"""Compare a persistent Codex app-server with fresh `codex exec` processes."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Print the date and time now."


@dataclass
class Sample:
    first_answer_seconds: float | None
    completed_seconds: float
    answer: str


class AppServer:
    def __init__(self, codex: str, cwd: Path, timeout: float) -> None:
        self.timeout = timeout
        self.messages: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self.stderr_lines: list[str] = []
        started = time.perf_counter()
        self.proc = subprocess.Popen(
            [codex, "app-server", "--listen", "stdio://"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout and self.proc.stderr
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.send(
            {
                "method": "initialize",
                "id": 0,
                "params": {
                    "clientInfo": {
                        "name": "codex_latency_benchmark",
                        "title": "Codex latency benchmark",
                        "version": "1.0.0",
                    }
                },
            }
        )
        self._wait_for(lambda msg: msg.get("id") == 0)
        self.send({"method": "initialized", "params": {}})
        self.startup_seconds = time.perf_counter() - started
        self.next_id = 1

    def _read_stdout(self) -> None:
        assert self.proc.stdout
        try:
            for line in self.proc.stdout:
                if line.strip():
                    self.messages.put(json.loads(line))
        except BaseException as exc:
            self.messages.put(exc)

    def _read_stderr(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip())

    def send(self, message: dict[str, Any]) -> None:
        if not self.proc.stdin:
            raise RuntimeError("app-server stdin is closed")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, predicate: Any) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stderr = "\n".join(self.stderr_lines[-20:])
                raise TimeoutError(f"app-server timed out\n{stderr}")
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("app-server timed out") from exc
            if isinstance(message, BaseException):
                raise RuntimeError("failed to read app-server output") from message
            if "error" in message and message.get("id") is not None:
                raise RuntimeError(f"app-server error: {message['error']}")
            if predicate(message):
                return message

    def run(self, prompt: str, cwd: Path) -> Sample:
        request_id = self.next_id
        self.next_id += 1
        started = time.perf_counter()
        self.send(
            {
                "method": "thread/start",
                "id": request_id,
                "params": {
                    "cwd": str(cwd),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                },
            }
        )
        response = self._wait_for(lambda msg: msg.get("id") == request_id)
        thread_id = response["result"]["thread"]["id"]

        turn_request_id = self.next_id
        self.next_id += 1
        self.send(
            {
                "method": "turn/start",
                "id": turn_request_id,
                "params": {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                },
            }
        )

        first_answer: float | None = None
        answer_parts: list[str] = []
        turn_id: str | None = None
        while True:
            message = self._wait_for(lambda _msg: True)
            now = time.perf_counter()
            if message.get("id") == turn_request_id:
                turn_id = message["result"]["turn"]["id"]
                continue
            method = message.get("method")
            params = message.get("params", {})
            event_turn_id = params.get("turnId") or params.get("turn", {}).get("id")
            if turn_id and event_turn_id and event_turn_id != turn_id:
                continue
            if method == "item/agentMessage/delta":
                if first_answer is None:
                    first_answer = now - started
                answer_parts.append(params.get("delta", ""))
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and not answer_parts:
                    text = item.get("text", "")
                    if text:
                        first_answer = first_answer or now - started
                        answer_parts.append(text)
            elif method == "turn/completed":
                status = params.get("turn", {}).get("status")
                if status not in (None, "completed"):
                    raise RuntimeError(f"app-server turn ended with status {status}")
                return Sample(first_answer, now - started, "".join(answer_parts).strip())

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def run_cli(codex: str, prompt: str, cwd: Path, timeout: float) -> Sample:
    command = [
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
        prompt,
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout
    first_answer: float | None = None
    answer = ""
    events: list[dict[str, Any]] = []
    try:
        for line in proc.stdout:
            if not line.strip():
                continue
            event = json.loads(line)
            events.append(event)
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    first_answer = first_answer or time.perf_counter() - started
                    answer = item.get("text", "")
        proc.wait(timeout=max(0.1, timeout - (time.perf_counter() - started)))
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    completed = time.perf_counter() - started
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.returncode:
        raise RuntimeError(
            f"codex exec exited with status {proc.returncode}\n{stderr}\n"
            f"last events: {events[-3:]}"
        )
    return Sample(first_answer, completed, answer.strip())


def mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def display_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="trials per method (default: 3)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--codex", default="codex", help="Codex executable (default: codex)")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per request")
    parser.add_argument("--json-output", type=Path, help="optionally save raw results as JSON")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    server_samples: list[Sample] = []
    cli_samples: list[Sample] = []
    server = AppServer(args.codex, args.cwd.resolve(), args.timeout)
    print(f"Persistent app-server initialized in {server.startup_seconds:.3f}s", flush=True)
    try:
        for index in range(args.runs):
            print(f"Run {index + 1}/{args.runs}: persistent server...", flush=True)
            server_sample = server.run(args.prompt, args.cwd.resolve())
            server_samples.append(server_sample)
            print(
                f"  first answer {display_seconds(server_sample.first_answer_seconds)}, "
                f"complete {server_sample.completed_seconds:.3f}s — {server_sample.answer!r}",
                flush=True,
            )

            print(f"Run {index + 1}/{args.runs}: fresh CLI...", flush=True)
            cli_sample = run_cli(args.codex, args.prompt, args.cwd.resolve(), args.timeout)
            cli_samples.append(cli_sample)
            print(
                f"  first answer {display_seconds(cli_sample.first_answer_seconds)}, "
                f"complete {cli_sample.completed_seconds:.3f}s — {cli_sample.answer!r}",
                flush=True,
            )
    finally:
        server.close()

    server_first = mean([sample.first_answer_seconds for sample in server_samples])
    cli_first = mean([sample.first_answer_seconds for sample in cli_samples])
    server_complete = mean([sample.completed_seconds for sample in server_samples])
    cli_complete = mean([sample.completed_seconds for sample in cli_samples])
    assert server_complete is not None and cli_complete is not None

    print("\nMean results")
    print(f"  persistent server: first answer {display_seconds(server_first)}, complete {server_complete:.3f}s")
    print(f"  fresh CLI:         first answer {display_seconds(cli_first)}, complete {cli_complete:.3f}s")
    print(f"  completion difference (CLI - server): {cli_complete - server_complete:+.3f}s")
    if cli_complete:
        print(f"  persistent server completion speedup: {cli_complete / server_complete:.2f}x")

    result = {
        "prompt": args.prompt,
        "runs": args.runs,
        "codex": args.codex,
        "cwd": str(args.cwd.resolve()),
        "app_server_startup_seconds": server.startup_seconds,
        "persistent_server": [asdict(sample) for sample in server_samples],
        "fresh_cli": [asdict(sample) for sample in cli_samples],
        "means": {
            "persistent_first_answer_seconds": server_first,
            "persistent_completed_seconds": server_complete,
            "cli_first_answer_seconds": cli_first,
            "cli_completed_seconds": cli_complete,
            "completion_difference_seconds": cli_complete - server_complete,
            "completion_speedup": cli_complete / server_complete,
        },
    }
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nRaw results saved to {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
