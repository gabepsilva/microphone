#!/usr/bin/env python3
"""Discover audio devices and ask the startup questions.

Every function here runs before any audio device is opened, so a rejected
answer re-prompts rather than unwinding a half-built runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from .domain import RESPONSE_POLICIES, resolve_response_policy
from .streams import application_streams, applications, graph


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


def print_menu(title, descriptions, first_number=1, note=None):
    """Print a numbered menu of choices, followed by a blank line."""
    print(title)
    for number, description in enumerate(descriptions, start=first_number):
        print(f"  {number:2d}) {description}")
    if note is not None:
        print(note)
    print()


def choose_from_menu(title, options, subject, first_number=1, note=None):
    """Offer numbered ``(description, value)`` options and return the value.

    The range in the prompt, the range in the retry message, and the range the
    answer is checked against are all the length of the menu, so a chooser
    cannot offer an entry it then refuses. ``first_number`` exists because the
    Them menu counts a "None" entry from zero.
    """
    print_menu(title, [description for description, _ in options], first_number, note)
    last = first_number + len(options) - 1
    selected = prompt_number(
        f"Select {subject} ({first_number}-{last}): ",
        first_number,
        last,
        f"Please enter a number from {first_number} to {last}.",
    )
    return options[selected - first_number][1]


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

    return choose_from_menu(
        "Available audio input devices:",
        [
            (
                f"{device['name']} "
                f"(device {index}, {int(device['default_samplerate'])} Hz)",
                (index, device),
            )
            for index, device in devices
        ],
        "a microphone",
    )


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


def find_audio_output(outputs, requested):
    """Match a requested output against its sink name, monitor, or description."""
    for output in outputs:
        if requested in (output["name"], output["monitor"], output["description"]):
            return output
    return None


def select_tts_output(outputs, requested):
    """Resolve a requested speech output without prompting."""
    output = find_audio_output(outputs, requested)
    if output is None:
        raise RuntimeError(f"Speech output {requested!r} was not found.")
    return output


def choose_tts_output(requested=None):
    """Name the sink Codex speaks through, or nothing to use the system default.

    This one never prompts. Where the assistant's voice comes out stopped being
    a question the session has to settle once application streams replaced the
    monitored sink: the answer no longer affects what gets transcribed, only
    which speakers a person hears it from, and every desktop already has a way
    to move one application's audio.
    """
    if requested is None:
        return None
    return select_tts_output(audio_outputs(), requested)


def stream_label(stream):
    """Describe one application the way the startup menu offers it."""
    state = "playing" if stream.playing else "idle"
    if stream.title and stream.title != stream.application:
        return f"{stream.application} — {stream.title} ({state})"
    return f"{stream.application} ({state})"


NO_THEM_STREAM = "none"


def select_them_stream(requested):
    """Resolve a requested application without prompting.

    A named application is taken at its word rather than checked against what
    is playing. It has to be: an application only appears in the graph once it
    opens audio, so a saved session that starts before the meeting does would
    otherwise refuse the very name it saved a moment ago. An application that
    never appears simply never gets linked, and the Them channel stays quiet.
    """
    if requested.lower() == NO_THEM_STREAM:
        return None
    return requested


def choose_them_stream(requested=None):
    """Choose the application whose audio is transcribed as Them."""
    if requested is not None:
        return select_them_stream(requested)

    streams = applications(application_streams(graph()))
    note = None
    if not streams:
        note = (
            "      No application is playing audio yet. Start the meeting and\n"
            "      run again, or pass --them-stream with the application name."
        )
    return choose_from_menu(
        "\nApplication to transcribe as Them:",
        [
            ("None", None),
            *((stream_label(stream), stream.application) for stream in streams),
        ],
        "an application",
        first_number=0,
        note=note,
    )


def choose_codex_after(requested=None):
    """Return the response policy whose speakers trigger a Codex reply."""
    if requested is not None:
        return resolve_response_policy(requested)

    policies = list(RESPONSE_POLICIES.values())
    print_menu("\nCodex should respond after:", [p.label for p in policies])
    return prompt_until(
        f"Select a response policy (1-{len(policies)}): ",
        resolve_response_policy,
        f"Please enter a number from 1 to {len(policies)}.",
    )


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
