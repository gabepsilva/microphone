#!/usr/bin/env python3
"""Discover audio devices and ask the startup questions.

Every function here runs before any audio device is opened, so a rejected
answer re-prompts rather than unwinding a half-built runtime.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import threading

from .domain import resolve_response_policy


def input_devices():
    try:
        import sounddevice as sd
    except OSError as error:
        raise RuntimeError(
            "Audio device discovery requires the PortAudio system library."
        ) from error
    return [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


def prompt_until(prompt, resolve, retry):
    """Read answers until ``resolve`` accepts one, re-prompting on rejection.

    A rejected answer must not end startup: every startup question is asked
    before any audio device is opened, so there is nothing to unwind, and the
    person answering is at the keyboard.
    """
    while True:
        answer = input(prompt).strip()
        try:
            return resolve(answer)
        except (KeyError, ValueError):
            print(retry)


def prompt_number(prompt, low, high, retry):
    """Read a number within a closed range, re-prompting until one arrives."""

    def resolve(answer):
        selected = int(answer)
        if not low <= selected <= high:
            raise ValueError(answer)
        return selected

    return prompt_until(prompt, resolve, retry)


def select_microphone(devices, requested):
    """Find a requested microphone by device index or exact name."""
    requested_text = str(requested)
    for index, device in devices:
        if requested_text in (str(index), device["name"]):
            return index, device
    raise RuntimeError(
        f"Microphone {requested!r} was not found. "
        "Remove it from the startup config to select interactively."
    )


def choose_microphone(requested=None):
    devices = input_devices()
    if not devices:
        raise RuntimeError("No audio input devices were found.")

    if requested is not None:
        return select_microphone(devices, requested)

    print("Available audio input devices:")
    for number, (index, device) in enumerate(devices, start=1):
        print(
            f"  {number:2d}) {device['name']} "
            f"(device {index}, {int(device['default_samplerate'])} Hz)"
        )
    print()

    selected = prompt_number(
        f"Select a microphone (1-{len(devices)}): ",
        1,
        len(devices),
        f"Please enter a number from 1 to {len(devices)}.",
    )
    return devices[selected - 1]


def audio_outputs():
    """Return PulseAudio/PipeWire sinks and their corresponding monitors."""
    if shutil.which("pactl") is None:
        return []
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            check=True,
            capture_output=True,
            text=True,
        )
        sinks = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []

    outputs = []
    for sink in sinks:
        name = sink.get("name")
        monitor = sink.get("monitor_source")
        if not name or not monitor:
            continue
        description = sink.get("description")
        if not description:
            description = sink.get("properties", {}).get("device.description")
        outputs.append(
            {
                "name": name,
                "monitor": monitor,
                "description": description or name,
            }
        )
    return outputs


ISOLATED_OUTPUT = "isolated"
ISOLATED_ALIASES = ("isolated", "virtual")


def find_audio_output(outputs, requested):
    """Match a requested output against its sink name, monitor, or description."""
    for output in outputs:
        if requested in (output["name"], output["monitor"], output["description"]):
            return output
    return None


def select_them_output(outputs, requested, require_isolation=False):
    """Resolve a requested Them output without prompting."""
    if requested.lower() == "none":
        return None
    if requested.lower() in ISOLATED_ALIASES:
        return {ISOLATED_OUTPUT: True}
    output = find_audio_output(outputs, requested)
    if output is None:
        raise RuntimeError(
            f"Audio output {requested!r} was not found. "
            "Use --them-output isolated, --them-output none, or select one "
            "interactively."
        )
    if require_isolation:
        raise RuntimeError(
            "Edge TTS cannot be used with a direct Them monitor. "
            "Use --them-output isolated or --them-output none."
        )
    return output


def choose_them_output(requested=None, require_isolation=False):
    """Choose an optional playback sink whose monitor is transcribed as Them."""
    outputs = audio_outputs()

    if requested is not None:
        return select_them_output(outputs, requested, require_isolation)

    print("\nAudio output to transcribe as Them:")
    print("   0) None")
    print("   1) Create isolated Voice Codex Meeting output (recommended)")
    if require_isolation:
        # A direct monitor would transcribe Codex's own speech back as Them.
        outputs = []
        print("      Direct output monitors are hidden while Edge TTS is enabled.")
    else:
        for number, output in enumerate(outputs, start=2):
            print(f"  {number:2d}) {output['description']}")

    print()
    selected = prompt_number(
        f"Select an audio output (0-{len(outputs) + 1}): ",
        0,
        len(outputs) + 1,
        f"Please enter a number from 0 to {len(outputs) + 1}.",
    )
    if selected == 0:
        return None
    if selected == 1:
        return {ISOLATED_OUTPUT: True}
    return outputs[selected - 2]


def select_playback_output(outputs, requested):
    """Resolve a requested playback output without prompting."""
    output = find_audio_output(outputs, requested)
    if output is None:
        raise RuntimeError(f"Playback output {requested!r} was not found.")
    return output


def choose_playback_output(requested=None):
    """Choose the physical output where meeting audio and TTS are heard."""
    outputs = audio_outputs()
    if not outputs:
        raise RuntimeError("No PulseAudio/PipeWire audio outputs were found.")

    if requested is not None:
        return select_playback_output(outputs, requested)

    print("\nPhysical output for meeting audio and Codex TTS:")
    for number, output in enumerate(outputs, start=1):
        print(f"  {number:2d}) {output['description']}")
    print()
    selected = prompt_number(
        f"Select a playback output (1-{len(outputs)}): ",
        1,
        len(outputs),
        f"Please enter a number from 1 to {len(outputs)}.",
    )
    return outputs[selected - 1]


class VirtualMeetingOutput:
    """Isolate meeting playback in a monitored sink and loop it to headphones."""

    DESCRIPTION = "Voice Codex Meeting"

    def __init__(self, playback_output):
        if shutil.which("pactl") is None:
            raise RuntimeError("Isolated meeting audio requires pactl.")
        self.playback_output = playback_output
        self.sink_name = f"voice_codex_meeting_{os.getpid()}"
        self.sink_module = None
        self.loopback_module = None
        self.close_lock = threading.Lock()
        self.closed = False
        try:
            self.sink_module = self._load_module(
                "module-null-sink",
                f"sink_name={self.sink_name}",
                (f"sink_properties=\"device.description='{self.DESCRIPTION}'\""),
            )
            self.loopback_module = self._load_module(
                "module-loopback",
                f"source={self.sink_name}.monitor",
                f"sink={self.playback_output['name']}",
                "source_dont_move=true",
                "sink_dont_move=true",
                "latency_msec=30",
            )
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _load_module(module, *arguments):
        try:
            result = subprocess.run(
                ["pactl", "load-module", module, *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip()
            raise RuntimeError(
                f"Could not load {module}: {detail or 'unknown pactl error'}"
            ) from error
        try:
            return int(result.stdout.strip())
        except ValueError as error:
            raise RuntimeError(
                f"pactl returned an invalid module ID for {module}."
            ) from error

    @property
    def transcript_output(self):
        return {
            "name": self.sink_name,
            "monitor": f"{self.sink_name}.monitor",
            "description": self.DESCRIPTION,
        }

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            for module_id in (self.loopback_module, self.sink_module):
                if module_id is None:
                    continue
                subprocess.run(
                    ["pactl", "unload-module", str(module_id)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            self.loopback_module = None
            self.sink_module = None
            self.closed = True


def choose_codex_after(requested=None):
    """Return the transcript speakers whose completed turns trigger Codex."""
    if requested is not None:
        policy = resolve_response_policy(requested)
        return policy.label, policy.speakers

    print("\nCodex should respond after:")
    print("   1) Them")
    print("   2) User Voice and Them")
    print("   3) User Voice")
    print("   4) Codex will be quiet for voice")
    print()
    policy = prompt_until(
        "Select a response policy (1-4): ",
        resolve_response_policy,
        "Please enter a number from 1 to 4.",
    )
    return policy.label, policy.speakers


TTS_ANSWERS = {
    "1": False,
    "no": False,
    "n": False,
    "2": True,
    "yes": True,
    "y": True,
}


def choose_tts(requested=None):
    """Choose whether Codex responses are also spoken."""
    if requested is not None:
        return requested == "on"

    print("\nSpeak Codex responses with Edge TTS?")
    print("   1) No")
    print("   2) Yes")
    print()
    return prompt_until(
        "Select audio output (1-2): ",
        TTS_ANSWERS.__getitem__,
        "Please enter 1 or 2.",
    )
