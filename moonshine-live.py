#!/usr/bin/env python3
"""Small interactive Moonshine streaming transcription test."""

import argparse
import sys
import time

import sounddevice as sd
from moonshine_voice import MicTranscriber, get_model_for_language
from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import TranscriptEventListener, TranscriptLine


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


class TerminalListener(TranscriptEventListener):
    """Render Moonshine's revised partial line on one terminal line."""

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.last_length = 0

    def _show(self, line: TranscriptLine):
        if line.words:
            text = " ".join(
                word.word.strip()
                for word in line.words
                if word.confidence >= self.confidence_threshold
            ).strip()
        else:
            text = line.text.strip()
        print(f"\r{text:<{self.last_length}}", end="", flush=True)
        self.last_length = len(text)

    def on_line_started(self, event):
        self.last_length = 0

    def on_line_text_changed(self, event):
        self._show(event.line)

    def on_line_completed(self, event):
        self._show(event.line)
        print()
        self.last_length = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("tiny-streaming", "small-streaming", "medium-streaming"),
        default="medium-streaming",
        help="Moonshine streaming model (default: medium-streaming)",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Minimum word confidence to display (default: 0.60)",
    )
    args = parser.parse_args()

    device_index, device = choose_device()
    model_arch = getattr(ModelArch, args.model.replace("-", "_").upper())

    print(f"\nUsing: {device['name']}", file=sys.stderr)
    print(f"Loading Moonshine {args.model} model...", file=sys.stderr)
    model_path, downloaded_arch = get_model_for_language(
        wanted_language=args.language,
        wanted_model_arch=model_arch,
    )

    transcriber = MicTranscriber(
        model_path=model_path,
        model_arch=downloaded_arch,
        update_interval=0.5,
        device=device_index,
        samplerate=16000,
        channels=1,
    )
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    transcriber.add_listener(TerminalListener(args.confidence))

    print("\nListening. Speak into the selected microphone; press Ctrl+C to stop.")
    try:
        transcriber.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        transcriber.stop()
        transcriber.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
