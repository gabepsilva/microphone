#!/usr/bin/env python3
"""Resolve startup configuration before the runtime is wired together.

Command line, then startup config file, then built-in defaults: this module
answers what the session will be, and nothing here opens an audio device or
contacts Codex.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from .catalog import CodexModelOption
from .choosers import (
    NO_AUDIO_STREAM,
    choose_audio_stream,
    choose_taga_after,
    choose_tts_output,
    input_devices,
    select_microphone,
)
from .config import load_startup_config
from .domain import POLICY_NAMES, ResponsePolicy, TurnSilence
from .speech import DEFAULT_PROVIDER, PROVIDER_LABELS, PROVIDERS, default_voice

DEFAULT_TURN_SILENCE = 3.0
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
DEFAULT_CODEX_EFFORT = "low"
DEFAULT_AUDIO_STREAM = NO_AUDIO_STREAM
DEFAULT_TAGA_AFTER = "both"

# Pre-firing overlaps Codex's thinking with the silence a finished turn is
# already waiting out, which is the difference between the first word landing
# as the window closes and landing most of a second after it. The cost is the
# occasional wasted turn, when a speaker who paused turns out not to have
# finished — which is what --no-codex-prefire is for.
DEFAULT_CODEX_PREFIRE = True

# A spoken conversation is waiting on the first word out loud, and the service
# tier is the one part of that wait this program does not otherwise control:
# nothing here can start speaking before Codex has produced a token. It costs
# more credits, which is what --no-codex-fast is for.
DEFAULT_CODEX_FAST = True

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tagalong.yaml",
)


def build_parser():
    """Build the command-line parser for the TagAlong entry point."""
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
        help=(
            "Microphone device index or exact name; the first available input "
            "when omitted"
        ),
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
        help=f"Quiet seconds before sending a turn to Taga (default: {DEFAULT_TURN_SILENCE})",
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
        help=f"Codex model (default: {DEFAULT_CODEX_MODEL})",
    )
    parser.add_argument(
        "--codex-reasoning",
        help=f"Codex reasoning effort (default: {DEFAULT_CODEX_EFFORT})",
    )
    parser.add_argument(
        "--codex-fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Request Codex Fast mode for lower latency; consumes more credits "
            f"(default: {'on' if DEFAULT_CODEX_FAST else 'off'}). "
            "--no-codex-fast asks for the standard tier"
        ),
    )
    parser.add_argument(
        "--codex-prefire",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Start answering before the turn-silence window closes, so the "
            f"reply begins as it ends (default: "
            f"{'on' if DEFAULT_CODEX_PREFIRE else 'off'}). "
            "--no-codex-prefire waits out the full window first"
        ),
    )
    parser.add_argument(
        "--audio-stream",
        help=(
            "Application whose audio is transcribed as Audio, or 'none'; "
            f"default: {DEFAULT_AUDIO_STREAM}"
        ),
    )
    parser.add_argument(
        "--tts-output",
        help="Sink Taga speaks through; the system default when omitted",
    )
    parser.add_argument(
        "--taga-after",
        choices=POLICY_NAMES,
        help=(
            "Which completed transcript turns trigger Taga "
            f"(default: {DEFAULT_TAGA_AFTER})"
        ),
    )
    parser.add_argument(
        "--tts-provider",
        choices=PROVIDERS,
        help=(f"Speech synthesizer for Taga's responses (default: {DEFAULT_PROVIDER})"),
    )
    parser.add_argument(
        "--tts-voice",
        help="Voice name; defaults to the chosen provider's own default voice",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run the session without Textual; keep the Unix socket open so "
            "Electron (or another client) can attach. Exclusive flock is "
            "unchanged — this is a start mode, not a detachable daemon (#102)."
        ),
    )
    return parser


def _apply_startup_config(parser, args):
    """Fill options the command line left unset from the startup config file.

    ``--config`` always names a file, defaulting to ``tagalong.yaml`` beside the
    package. A missing file is an empty configuration layer so a first run can
    start from built-in defaults; every other read failure remains fatal.
    """
    try:
        loaded_settings = load_startup_config(args.config, missing_ok=True)
    except RuntimeError as error:
        parser.error(str(error))
    for key, value in loaded_settings.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if loaded_settings:
        print(f"Loaded startup config: {args.config}", file=sys.stderr)
    else:
        print(
            f"Startup config is empty or unavailable; using defaults: {args.config}",
            file=sys.stderr,
        )


def _resolve_defaults(args):
    """Fill in what neither the command line nor the config file supplied.

    These cannot be argparse defaults. Anything argparse has already filled in
    is not None, and the config file only fills what is still None — so an
    argparse default would silently outrank the saved value it is meant to
    stand in for. The voice has a second reason: its default depends on the
    provider, which the config file may only have supplied a moment ago.
    """
    for option, fallback in (
        ("audio_stream", DEFAULT_AUDIO_STREAM),
        ("taga_after", DEFAULT_TAGA_AFTER),
        ("tts_provider", DEFAULT_PROVIDER),
        ("turn_silence", DEFAULT_TURN_SILENCE),
        ("codex_model", DEFAULT_CODEX_MODEL),
        ("codex_reasoning", DEFAULT_CODEX_EFFORT),
        ("codex_fast", DEFAULT_CODEX_FAST),
        ("codex_prefire", DEFAULT_CODEX_PREFIRE),
    ):
        if getattr(args, option) is None:
            setattr(args, option, fallback)
    if args.tts_voice is None:
        args.tts_voice = default_voice(args.tts_provider)


def _validate_startup_args(parser, args):
    """Reject values argparse cannot constrain, including config-file values."""
    if args.tts_provider is not None and args.tts_provider not in PROVIDERS:
        allowed = ", ".join(repr(name) for name in PROVIDERS)
        parser.error(f"startup config 'tts_provider' must be one of {allowed}")
    if args.codex_fast is not None and not isinstance(args.codex_fast, bool):
        parser.error("startup config 'codex_fast' must be true or false")
    if args.codex_prefire is not None and not isinstance(args.codex_prefire, bool):
        parser.error("startup config 'codex_prefire' must be true or false")
    if args.codex_reasoning is not None and (
        not isinstance(args.codex_reasoning, str) or not args.codex_reasoning
    ):
        parser.error("startup config 'codex_reasoning' must be a non-empty string")
    # bool is a subclass of int, so a YAML `true` would otherwise pass as a
    # one-second window rather than being rejected as the mistake it is.
    if args.turn_silence is not None and (
        isinstance(args.turn_silence, bool)
        or not isinstance(args.turn_silence, int | float)
    ):
        parser.error("startup config 'turn_silence' must be a number")
    if args.taga_after is not None and args.taga_after not in POLICY_NAMES:
        allowed = ", ".join(repr(name) for name in POLICY_NAMES)
        parser.error(f"startup config 'taga_after' must be one of {allowed}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    if args.turn_silence is not None and not (
        TurnSilence.MINIMUM <= args.turn_silence <= TurnSilence.MAXIMUM
    ):
        parser.error(
            f"--turn-silence must be between {TurnSilence.MINIMUM} and "
            f"{TurnSilence.MAXIMUM} seconds"
        )


def parse_startup_args(argv=None):
    """Parse argv, merge the startup config beneath it, and validate the result."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_startup_config(parser, args)
    _validate_startup_args(parser, args)
    _resolve_defaults(args)
    return parser, args


def validate_codex_reasoning(parser, args, models: list[CodexModelOption]) -> None:
    """Reject an effort the selected catalog model does not support."""
    selected = next((model for model in models if model.slug == args.codex_model), None)
    if selected is None or args.codex_reasoning in selected.efforts:
        return
    allowed = ", ".join(repr(effort) for effort in selected.efforts)
    parser.error(
        f"startup config 'codex_reasoning' for model {args.codex_model!r} "
        f"must be one of {allowed}"
    )


@dataclass(frozen=True)
class StartupSelection:
    """The choices resolved before the runtime is wired together."""

    device_index: int | None
    device: dict | None
    tts_provider: str
    audio_stream: str | None
    tts_output: dict | None
    policy: ResponsePolicy


def startup_settings(selection, args):
    """Build the flat mapping the config file holds."""
    return {
        "microphone": (
            selection.device["name"]
            if selection.device is not None
            else args.microphone
        ),
        "tts_provider": selection.tts_provider,
        "audio_stream": (
            NO_AUDIO_STREAM
            if selection.audio_stream is None
            else selection.audio_stream
        ),
        "tts_output": (
            selection.tts_output["name"] if selection.tts_output is not None else None
        ),
        "taga_after": selection.policy.name,
        "turn_silence": args.turn_silence,
        "codex_model": args.codex_model,
        "codex_reasoning": args.codex_reasoning,
        "codex_fast": args.codex_fast,
        "codex_prefire": args.codex_prefire,
    }


def print_startup_summary(args, selection, stream=sys.stderr):
    """Report the resolved startup choices before the slow model load begins."""
    audio_stream = selection.audio_stream
    tts_output = selection.tts_output
    microphone = (
        selection.device["name"] if selection.device is not None else "None yet"
    )
    print(f"\nVoice microphone: {microphone}", file=stream)
    print(f"Audio application: {audio_stream or 'None'}", file=stream)
    print(f"Taga response policy: {selection.policy.label}", file=stream)
    print(f"Voice turn silence: {args.turn_silence:.1f}s", file=stream)
    print(f"Codex speed: {'Fast' if args.codex_fast else 'Standard'}", file=stream)
    print(f"Codex command access: {args.sandbox}", file=stream)
    engine = PROVIDER_LABELS[selection.tts_provider]
    print(f"Taga audio: {engine} ({args.tts_voice})", file=stream)
    if tts_output is not None:
        print(f"Taga speaks through: {tts_output['description']}", file=stream)
    print(f"Loading Moonshine {args.model} model...", file=stream)


def resolve_startup_selection(args):
    """Resolve startup choices without requiring a device or terminal prompt."""
    try:
        devices = input_devices()
    except RuntimeError as error:
        print(f"Microphone discovery unavailable: {error}", file=sys.stderr)
        devices = []

    device_index = None
    device = None
    if args.microphone is None and devices:
        device_index, device = devices[0]
    elif args.microphone is not None:
        try:
            device_index, device = select_microphone(devices, args.microphone)
        except RuntimeError:
            print(
                f"Microphone {args.microphone!r} is not available yet; "
                "select one from the sidebar after startup.",
                file=sys.stderr,
            )
    audio_stream = choose_audio_stream(args.audio_stream)
    policy = choose_taga_after(args.taga_after)
    return StartupSelection(
        device_index=device_index,
        device=device,
        tts_provider=args.tts_provider,
        audio_stream=audio_stream,
        tts_output=choose_tts_output(args.tts_output),
        policy=policy,
    )


def build_session_state(args, selection, models: list[CodexModelOption] | None = None):
    """Build the sidebar's view of the resolved startup choices."""
    from .tui import SessionState

    models = [] if models is None else models
    model_choices = [(model.label, model.slug) for model in models]
    efforts_by_model = {model.slug: list(model.efforts) for model in models}
    default_effort_by_model = {model.slug: model.default_effort for model in models}
    return SessionState(
        microphone=(
            selection.device["name"]
            if selection.device is not None
            else args.microphone
        ),
        policy=selection.policy.name,
        tts_provider=selection.tts_provider,
        tts_voice=args.tts_voice,
        turn_silence=args.turn_silence,
        confidence=args.confidence,
        language=args.language,
        moonshine=args.model,
        codex_model=args.codex_model,
        codex_effort=args.codex_reasoning,
        codex_models=model_choices or [(args.codex_model, args.codex_model)],
        codex_efforts=efforts_by_model.get(args.codex_model, [args.codex_reasoning]),
        codex_efforts_by_model=efforts_by_model,
        codex_default_effort_by_model=default_effort_by_model,
        codex_tier="fast" if args.codex_fast else "standard",
        codex_sandbox=args.sandbox,
    )


def run_session(tui, channels, conversation, audio=None, microphone=None):
    """Run the interface until it quits, then tear every channel down in order.

    Transcribers stop before their listeners close so a listener cannot be fed
    a partial turn after it has flushed, and every transcriber is closed only
    once no listener can still be called back.

    The far end closes first and closes itself, because it may not exist, may
    have been built minutes into the session, and owns the only reference to
    what it built.
    """
    try:
        for transcriber, _ in channels:
            transcriber.start()
        tui.run()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        if audio is not None:
            audio.close()
        if microphone is not None:
            microphone.close()
        for transcriber, _ in channels:
            transcriber.stop()
        for _, listener in channels:
            listener.close()
        for transcriber, _ in channels:
            transcriber.close()
        conversation.close()
