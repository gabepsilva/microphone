#!/usr/bin/env python3
"""Resolve every startup choice before the runtime is wired together.

Command line, then startup config file, then the interactive prompts: this
module answers what the session will be, and nothing here opens an audio
device or contacts Codex.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass

from .catalog import populate_codex_model_catalog
from .choosers import (
    ISOLATED_OUTPUT,
    VirtualMeetingOutput,
    choose_codex_after,
    choose_microphone,
    choose_playback_output,
    choose_them_output,
    choose_tts,
)
from .config import load_startup_config
from .domain import POLICY_NAMES, ResponsePolicy

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice.yaml",
)


def build_parser():
    """Build the command-line parser for the Voice Codex entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "--load-config",
        dest="config",
        metavar="YAML",
        default=DEFAULT_CONFIG_FILE,
        help=f"Load startup prompt choices from a YAML file (default: {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument(
        "--save-config",
        metavar="YAML",
        help="Save resolved startup prompt choices to a YAML file",
    )
    parser.add_argument(
        "--microphone",
        help="Microphone device index or exact name; prompts when omitted",
    )
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
        choices=("read-only", "workspace-write", "full-access"),
        default="full-access",
        help=(
            "Codex command access: full-access runs commands on the host "
            "without sandbox restrictions or approval prompts "
            "(default: full-access)"
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
    parser.add_argument(
        "--codex-fast",
        action="store_true",
        help=("Request Codex Fast mode for lower latency; consumes more credits"),
    )
    parser.add_argument(
        "--them-output",
        help=(
            "PulseAudio/PipeWire output name to transcribe as Them, "
            "'isolated', or 'none'; prompts when omitted"
        ),
    )
    parser.add_argument(
        "--playback-output",
        help=("Physical output used with --them-output isolated; prompts when omitted"),
    )
    parser.add_argument(
        "--codex-after",
        choices=POLICY_NAMES,
        help="Which completed transcript turns trigger Codex; prompts when omitted",
    )
    parser.add_argument(
        "--tts",
        choices=("on", "off"),
        help="Speak Codex responses with Edge TTS; prompts when omitted",
    )
    parser.add_argument(
        "--tts-voice",
        default="en-US-AndrewNeural",
        help="Edge TTS voice (default: en-US-AndrewNeural)",
    )
    return parser


def _apply_startup_config(parser, args):
    """Fill options the command line left unset from the startup config file."""
    if args.config is None:
        return
    try:
        loaded_settings = load_startup_config(args.config)
    except RuntimeError as error:
        parser.error(str(error))
    for key, value in loaded_settings.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    print(f"Loaded startup config: {args.config}", file=sys.stderr)


def _validate_startup_args(parser, args):
    """Reject values argparse cannot constrain, including config-file values."""
    if args.tts not in (None, "on", "off"):
        parser.error("startup config 'tts' must be 'on' or 'off'")
    if args.codex_after is not None and args.codex_after not in POLICY_NAMES:
        allowed = ", ".join(repr(name) for name in POLICY_NAMES)
        parser.error(f"startup config 'codex_after' must be one of {allowed}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    if args.turn_silence <= 0:
        parser.error("--turn-silence must be greater than 0")


def parse_startup_args(argv=None):
    """Parse argv, merge the startup config beneath it, and validate the result."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_startup_config(parser, args)
    _validate_startup_args(parser, args)
    return parser, args


@dataclass(frozen=True)
class StartupSelection:
    """The interactive choices resolved before the runtime is wired together."""

    device_index: int
    device: dict
    tts_enabled: bool
    them_output: dict | None
    them_output_setting: str
    playback_output: dict | None
    policy: ResponsePolicy


def them_output_name(them_output):
    """Name the Them output the way a saved startup config records it."""
    if them_output is None:
        return "none"
    if them_output.get(ISOLATED_OUTPUT):
        return ISOLATED_OUTPUT
    return them_output["name"]


def startup_settings(selection):
    """Build the flat mapping saved by ``--save-config``."""
    return {
        "microphone": selection.device["name"],
        "tts": "on" if selection.tts_enabled else "off",
        "them_output": selection.them_output_setting,
        "playback_output": (
            selection.playback_output["name"]
            if selection.playback_output is not None
            else None
        ),
        "codex_after": selection.policy.name,
    }


def print_startup_summary(args, selection, stream=sys.stderr):
    """Report the resolved startup choices before the slow model load begins."""
    them_output = selection.them_output
    playback_output = selection.playback_output
    print(f"\nUser microphone: {selection.device['name']}", file=stream)
    if them_output is None:
        print("Them audio output: None", file=stream)
    else:
        print(f"Them audio output: {them_output['description']}", file=stream)
    print(f"Codex response policy: {selection.policy.label}", file=stream)
    print(f"Voice turn silence: {args.turn_silence:.1f}s", file=stream)
    print(f"Codex speed: {'Fast' if args.codex_fast else 'Standard'}", file=stream)
    print(f"Codex command access: {args.sandbox}", file=stream)
    print(
        f"Codex audio: "
        f"{'Edge TTS (' + args.tts_voice + ')' if selection.tts_enabled else 'Off'}",
        file=stream,
    )
    if playback_output is not None:
        print(
            f"Meeting and TTS playback: {playback_output['description']}", file=stream
        )
    elif selection.tts_enabled and them_output is not None:
        print(
            "Warning: a non-isolated Them monitor may transcribe Codex TTS.",
            file=stream,
        )
    print(f"Loading Moonshine {args.model} model...", file=stream)


def resolve_startup_selection(args):
    """Run the interactive choosers and capture what they resolved to."""
    device_index, device = choose_microphone(args.microphone)
    tts_enabled = choose_tts(args.tts)
    them_output = choose_them_output(args.them_output, require_isolation=tts_enabled)
    them_output_setting = them_output_name(them_output)
    virtual_meeting = None
    playback_output = None
    if them_output is not None and them_output.get(ISOLATED_OUTPUT):
        playback_output = choose_playback_output(args.playback_output)
        virtual_meeting = VirtualMeetingOutput(playback_output)
        them_output = virtual_meeting.transcript_output
        print(
            "\nCreated isolated meeting output: Voice Codex Meeting",
            file=sys.stderr,
        )
        print(
            "Set Zoom or the meeting app's speaker to Voice Codex Meeting.",
            file=sys.stderr,
        )
    policy = choose_codex_after(args.codex_after)
    return (
        StartupSelection(
            device_index=device_index,
            device=device,
            tts_enabled=tts_enabled,
            them_output=them_output,
            them_output_setting=them_output_setting,
            playback_output=playback_output,
            policy=policy,
        ),
        virtual_meeting,
    )


def build_session_state(args, selection):
    """Build the sidebar's view of the resolved startup choices."""
    from .tui import SessionState

    return SessionState(
        policy=selection.policy.name,
        tts_enabled=selection.tts_enabled,
        tts_voice=args.tts_voice,
        turn_silence=args.turn_silence,
        confidence=args.confidence,
        language=args.language,
        moonshine=args.model,
        codex_model=args.codex_model,
        codex_effort=args.codex_reasoning,
        codex_tier="fast" if args.codex_fast else "standard",
        codex_sandbox=args.sandbox,
    )


def run_session(tui, channels, conversation, virtual_meeting):
    """Run the interface until it quits, then tear every channel down in order.

    Transcribers stop before their listeners close so a listener cannot be fed
    a partial turn after it has flushed, and every transcriber is closed only
    once no listener can still be called back.
    """
    try:
        for transcriber, _ in channels:
            transcriber.start()
        threading.Thread(
            target=populate_codex_model_catalog,
            args=(tui,),
            name="CodexModelCatalog",
            daemon=True,
        ).start()
        tui.run()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        for transcriber, _ in channels:
            transcriber.stop()
        for _, listener in channels:
            listener.close()
        for transcriber, _ in channels:
            transcriber.close()
        conversation.close()
        if virtual_meeting is not None:
            virtual_meeting.close()
