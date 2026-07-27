#!/usr/bin/env python3
"""Wire the always-listening User/Them/Codex conversation together.

This module is the composition root and nothing else. The runtime parts it
assembles are independent of one another — none of them imports any of the
others:

  * ``capture``  — microphone and sink-monitor audio into Moonshine
  * ``listener`` — transcription events into completed turns
  * ``codex``    — the Codex thread and its streamed turns
  * ``speech``   — which synthesizer speaks Codex responses, and the two
                   engines behind it in ``tts`` and ``piper_tts``

``startup`` resolves what the session will be before any of that is built,
drawing on ``choosers`` for the interactive questions and ``catalog`` for the
model list. It is the only module besides this one that imports more than its
own concern, and it holds nothing the running session needs.

Keep it that way: an import between two of the four above is the first sign
that a boundary is in the wrong place.
"""

from __future__ import annotations

import sys

from moonshine_voice import get_model_for_language
from moonshine_voice.moonshine_api import ModelArch

from .capture import (
    CaptureSettings,
    PulseMonitorTranscriber,
    SoundActivityReporter,
    metered_mic_transcriber,
)
from .codex import CodexConversation, CodexSettings
from .config import StartupConfigFile, save_startup_config
from .domain import SpeakerGate, TurnSilence, TurnSilenceClock
from .listener import TranscriptSubmitter, tts_switch
from .speech import SwitchableSpeech, provider_switch
from .startup import (
    build_session_state,
    parse_startup_args,
    print_startup_summary,
    resolve_startup_selection,
    run_session,
    startup_settings,
)


def build_speech(selection, args, playback_output):
    """Build the session's speech, or nothing when the session is silent."""
    if not selection.tts_enabled:
        return None
    return SwitchableSpeech.start(
        selection.tts_provider,
        args.tts_voice,
        output_sink=(playback_output["name"] if playback_output is not None else None),
    )


def remembering(hook, config, key, encode=lambda value: value):
    """Wrap a sidebar hook so an accepted change is written to the config file.

    Only an accepted change is recorded. A hook that refuses — a model the
    session cannot switch to, a speech engine it does not have — has changed
    nothing, and saving it would make the next session start somewhere this
    one never went.
    """

    def apply(value):
        accepted = hook(value)
        if accepted is not False:
            config.record(key, encode(value))
        return accepted

    return apply


def attach_conversation_hooks(tui, conversation, tts, config, turn_silence):
    """Point the interface's controls at the conversation, its speech, and disk."""
    tui.hooks.on_user_text = lambda text: conversation.ingest(
        "User Text", text, respond=True
    )
    tui.hooks.on_interrupt = conversation.interrupt
    tui.hooks.on_codex_model = remembering(
        conversation.request_model, config, "codex_model"
    )
    tui.hooks.on_codex_effort = remembering(
        conversation.request_reasoning_effort, config, "codex_reasoning"
    )
    tui.hooks.on_tts = remembering(
        tts_switch(tts), config, "tts", encode=lambda on: "on" if on else "off"
    )
    tui.hooks.on_tts_provider = remembering(
        provider_switch(tts), config, "tts_provider"
    )
    tui.hooks.on_turn_silence = remembering_turn_silence(turn_silence, config)


def remembering_turn_silence(turn_silence, config):
    """Adopt a typed window and record the value actually applied.

    The applied value is what is saved, not the typed one: the two differ
    whenever the setting clamps, and the file has to describe the session that
    ran rather than the request that produced it.
    """

    def apply(seconds):
        applied = turn_silence.set(seconds)
        config.record("turn_silence", applied)
        return applied

    return apply


def main():
    parser, args = parse_startup_args()
    selection, virtual_meeting = resolve_startup_selection(args)
    them_output = selection.them_output
    playback_output = selection.playback_output

    settings = startup_settings(selection, args)
    if args.save_config is not None:
        try:
            save_startup_config(args.save_config, settings)
        except RuntimeError as error:
            parser.error(str(error))
        print(f"Saved startup config: {args.save_config}", file=sys.stderr)
    # Sidebar changes are written back to the file the session started from,
    # so the next run opens where this one left off.
    config = StartupConfigFile(args.config, settings)
    model_arch = getattr(ModelArch, args.model.replace("-", "_").upper())

    print_startup_summary(args, selection)
    model_path, downloaded_arch = get_model_for_language(
        wanted_language=args.language,
        wanted_model_arch=model_arch,
    )

    available_speakers = {"User Voice"}
    if them_output is not None:
        available_speakers.add("Them")
    gate = SpeakerGate(selection.policy.speakers, available_speakers)

    from .tui import VoiceCodexTUI

    turn_silence = TurnSilence(args.turn_silence)
    countdown = TurnSilenceClock(turn_silence)
    tts = build_speech(selection, args, playback_output)
    tui = VoiceCodexTUI(
        build_session_state(args, selection),
        countdown=countdown,
        speech=tts,
        on_policy=remembering(gate.set_policy, config, "codex_after"),
    )
    if virtual_meeting is not None:
        tui.hooks.on_quit = virtual_meeting.close
    transcript_display = tui
    conversation = CodexConversation(
        CodexSettings(
            sandbox=args.sandbox,
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning,
            service_tier="fast" if args.codex_fast else None,
        ),
        transcript_display,
        tts,
    )

    attach_conversation_hooks(tui, conversation, tts, config, turn_silence)
    submitter = TranscriptSubmitter(conversation, gate, tts)

    user_listener = submitter.channel(
        args.confidence,
        turn_silence,
        "User Voice",
        transcript_display,
        countdown=countdown,
    )
    user_transcriber = metered_mic_transcriber(
        model_path=model_path,
        model_arch=downloaded_arch,
        update_interval=0.25,
        device=selection.device_index,
        samplerate=16000,
        channels=1,
        level_reporter=SoundActivityReporter(tui, "mic"),
    )
    user_transcriber.add_listener(user_listener)

    tui.hooks.on_mute = user_listener.set_muted
    tui.set_audio("mic", device=selection.device["name"])
    tui.set_codex(thread=conversation.thread.id)
    if playback_output is not None:
        tui.set_output(playback_output["description"])

    them_listener = None
    them_transcriber = None
    if them_output is not None:
        them_listener = submitter.channel(
            args.confidence,
            turn_silence,
            "Them",
            transcript_display,
            countdown=countdown,
        )
        them_transcriber = PulseMonitorTranscriber(
            model_path=model_path,
            model_arch=downloaded_arch,
            monitor=them_output["monitor"],
            capture=CaptureSettings(update_interval=0.25),
            level_reporter=SoundActivityReporter(tui, "them"),
        )
        them_transcriber.add_listener(them_listener)
        tui.hooks.on_them_mute = them_listener.set_muted
        tui.set_audio("them", device=them_output["description"])
    channels = [(user_transcriber, user_listener)]
    if them_transcriber is not None:
        channels.append((them_transcriber, them_listener))
    run_session(tui, channels, conversation, virtual_meeting)


def run_entrypoint():
    """Run the application, reporting failures without a traceback.

    Both compatibility scripts call this so the shutdown behavior they
    advertise cannot drift apart.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
