#!/usr/bin/env python3
"""Always-listening Moonshine-to-Codex conversation."""

import argparse
import os
import queue
import sys
import threading
import time

import sounddevice as sd
from moonshine_voice import MicTranscriber, get_model_for_language
from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import TranscriptEventListener
from openai_codex import ApprovalMode, Codex, Sandbox
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    CommandExecutionOutputDeltaNotification,
    CommandExecutionThreadItem,
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallThreadItem,
    ReasoningEffort,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)


def input_devices():
    return [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


def choose_device():
    devices = input_devices()
    if not devices:
        raise RuntimeError("No audio input devices were found.")

    print("Available audio input devices:")
    for number, (index, device) in enumerate(devices, start=1):
        print(
            f"  {number:2d}) {device['name']} "
            f"(device {index}, {int(device['default_samplerate'])} Hz)"
        )
    print()

    while True:
        answer = input(f"Select a microphone (1-{len(devices)}): ").strip()
        try:
            selected = int(answer)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(devices):
            return devices[selected - 1]
        print(f"Please enter a number from 1 to {len(devices)}.")


class ConversationListener(TranscriptEventListener):
    def __init__(self, confidence_threshold, turn_silence, submit):
        self.confidence_threshold = confidence_threshold
        self.turn_silence = turn_silence
        self.submit = submit
        self.lock = threading.Lock()
        self.pending = []
        self.timer = None

    def _text(self, line):
        if line.words:
            return " ".join(
                word.word.strip()
                for word in line.words
                if word.confidence >= self.confidence_threshold
            ).strip()
        return line.text.strip()

    def _flush(self):
        with self.lock:
            text = " ".join(self.pending).strip()
            self.pending.clear()
            self.timer = None
        if text:
            self.submit(text)

    def _cancel_timer(self):
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None

    def on_line_started(self, event):
        # Speech has resumed. Keep all completed lines buffered and wait for
        # this new line to finish before considering the turn complete.
        self._cancel_timer()

    def on_line_text_changed(self, event):
        # Partial text means the user is actively continuing the same turn.
        self._cancel_timer()

    def on_line_completed(self, event):
        text = self._text(event.line)
        if not text:
            return
        with self.lock:
            self.pending.append(text)
            if self.timer is not None:
                self.timer.cancel()
            self.timer = threading.Timer(self.turn_silence, self._flush)
            self.timer.daemon = True
            self.timer.start()

    def close(self):
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None


class CodexConversation:
    def __init__(self, sandbox, model, reasoning_effort):
        self.sandbox = (
            Sandbox.workspace_write
            if sandbox == "workspace-write"
            else Sandbox.read_only
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.requests = queue.Queue(maxsize=1)
        self.accepting = True
        self.shutdown_requested = threading.Event()
        self.active_turn = None
        self.codex = Codex()
        self.thread = self.codex.thread_start(
            model=self.model,
            sandbox=self.sandbox,
            approval_mode=ApprovalMode.deny_all,
            cwd=os.getcwd(),
        )
        print(
            f"Codex App Server ready. Conversation thread: {self.thread.id}",
            flush=True,
        )
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def submit(self, text):
        if not self.accepting or self.shutdown_requested.is_set():
            return
        self.accepting = False
        try:
            self.requests.put_nowait(text)
        except queue.Full:
            self.accepting = True

    def _run_codex(self, text):
        print("\n--- Codex is working ---", flush=True)
        try:
            self.active_turn = self.thread.turn(
                text,
                effort=ReasoningEffort(self.reasoning_effort),
                sandbox=self.sandbox,
                approval_mode=ApprovalMode.deny_all,
            )
            self._stream_turn(self.active_turn)
        except Exception as error:
            print(f"\nCodex error: {error}", file=sys.stderr, flush=True)
        finally:
            self.active_turn = None
            print("--- Codex finished ---", flush=True)

    @staticmethod
    def _item_root(item):
        return item.root if hasattr(item, "root") else item

    def _stream_turn(self, turn):
        agent_message_open = False
        last_usage = None

        for event in turn.stream():
            payload = event.payload

            if isinstance(payload, ItemStartedNotification):
                item = self._item_root(payload.item)
                if isinstance(item, AgentMessageThreadItem):
                    if not agent_message_open:
                        print("\nCodex: ", end="", flush=True)
                        agent_message_open = True
                elif isinstance(item, CommandExecutionThreadItem):
                    if agent_message_open:
                        print()
                        agent_message_open = False
                    print(f"\n$ {item.command}", flush=True)
                elif isinstance(item, McpToolCallThreadItem):
                    if agent_message_open:
                        print()
                        agent_message_open = False
                    print(f"\nTool: {item.server}.{item.tool}", flush=True)
                continue

            if isinstance(payload, AgentMessageDeltaNotification):
                print(payload.delta, end="", flush=True)
                agent_message_open = True
                continue

            if isinstance(payload, CommandExecutionOutputDeltaNotification):
                print(payload.delta, end="", flush=True)
                continue

            if isinstance(payload, ItemCompletedNotification):
                item = self._item_root(payload.item)
                if isinstance(item, AgentMessageThreadItem) and agent_message_open:
                    print()
                    agent_message_open = False
                elif isinstance(item, CommandExecutionThreadItem):
                    print(f"[command exit: {item.exit_code}]", flush=True)
                elif isinstance(item, McpToolCallThreadItem):
                    print(f"[tool status: {item.status}]", flush=True)
                continue

            if isinstance(payload, ThreadTokenUsageUpdatedNotification):
                last_usage = payload.token_usage.last
                continue

            if isinstance(payload, ErrorNotification):
                print(
                    f"\nCodex runtime error: {payload.error.message}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if isinstance(payload, TurnCompletedNotification):
                if agent_message_open:
                    print()
                if last_usage is not None:
                    print(
                        f"[tokens: {last_usage.total_tokens}]",
                        flush=True,
                    )

    def _worker(self):
        while not self.shutdown_requested.is_set():
            try:
                text = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                return
            self._run_codex(text)
            self.accepting = True

    def close(self):
        self.shutdown_requested.set()
        if self.active_turn is not None:
            try:
                self.active_turn.interrupt()
            except Exception:
                pass
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=3)
        self.codex.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("tiny-streaming", "small-streaming", "medium-streaming"),
        default="medium-streaming",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument(
        "--turn-silence",
        type=float,
        default=3.0,
        help="Quiet seconds before sending a turn to Codex (default: 3.0)",
    )
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write"),
        default="read-only",
        help=(
            "Codex file policy: read-only permits inspection commands but "
            "denies file edits; workspace-write permits edits "
            "(default: read-only)"
        ),
    )
    parser.add_argument(
        "--codex-model",
        default="gpt-5.6-luna",
        help="Codex model (default: gpt-5.6-luna)",
    )
    parser.add_argument(
        "--codex-reasoning",
        choices=("low", "medium", "high"),
        default="low",
        help="Codex reasoning effort (default: low)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    if args.turn_silence <= 0:
        parser.error("--turn-silence must be greater than 0")

    device_index, device = choose_device()
    model_arch = getattr(ModelArch, args.model.replace("-", "_").upper())

    print(f"\nUsing: {device['name']}", file=sys.stderr)
    print(f"Loading Moonshine {args.model} model...", file=sys.stderr)
    model_path, downloaded_arch = get_model_for_language(
        wanted_language=args.language,
        wanted_model_arch=model_arch,
    )

    conversation = CodexConversation(
        args.sandbox,
        args.codex_model,
        args.codex_reasoning,
    )
    listener = ConversationListener(
        args.confidence,
        args.turn_silence,
        conversation.submit,
    )
    transcriber = MicTranscriber(
        model_path=model_path,
        model_arch=downloaded_arch,
        update_interval=0.5,
        device=device_index,
        samplerate=16000,
        channels=1,
    )
    transcriber.add_listener(listener)

    print("\nListening continuously.", flush=True)
    print("Speak normally; a completed utterance is sent to Codex.", flush=True)
    print("Press Ctrl+C to stop. Spoken text is not displayed.", flush=True)
    try:
        transcriber.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        listener.close()
        transcriber.stop()
        transcriber.close()
        conversation.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
