#!/usr/bin/env python3
"""Boundary between the runtime and whichever synthesizer is speaking.

Two engines implement :class:`SpeechEngine`, and they are not variations on
one design. Edge sends each sentence to Microsoft and waits on the network, so
it prefetches ahead and trims the silence the service prepends. Piper runs the
model in this process about thirty times faster than the audio plays, so it
needs no prefetch, no stagger, and no external trimmer at all. Neither shape
is right for the other, which is why this file describes what they owe their
callers and not how either one works.

The runtime holds a :class:`SwitchableSpeech`, never an engine directly. That
is what lets the sidebar change providers mid-session: the conversation, the
transcript submitter, and the interface toggle all keep the same object while
the engine underneath it is replaced.
"""

from __future__ import annotations

import sys
import threading
from typing import Protocol

EDGE = "edge"
PIPER = "piper"

DEFAULT_PROVIDER = PIPER

# Piper's medium models are the reason local synthesis is the default: they
# reach the first word in roughly a quarter the time Edge needs, because no
# part of the answer has to cross the network. Its high-quality models give
# that entire margin back, so the default is medium on purpose.
DEFAULT_VOICES = {
    EDGE: "en-US-AndrewNeural",
    PIPER: "en_US-lessac-medium",
}

PROVIDER_LABELS = {
    PIPER: "Piper (local)",
    EDGE: "Edge (cloud)",
}

PROVIDERS = tuple(PROVIDER_LABELS)

# Silence is not a third engine — nothing here builds one — but the interface
# offers it beside the two that are, because "which voice answers me" and
# "should one answer at all" are the same question to whoever is asking. The
# name lives here so the sidebar and its tests agree on one spelling.
NO_VOICE = "none"
NO_VOICE_LABEL = "No voice reply"


class SpeechEngine(Protocol):
    """Speak Taga's sentences, and stop when the user starts talking."""

    def begin_turn(self) -> None: ...

    def speak(self, text: str) -> None: ...

    def interrupt(self) -> None: ...

    def set_enabled(self, enabled: bool) -> None: ...

    def is_likely_echo(self, text: str) -> bool: ...

    def is_speaking(self) -> bool: ...

    def close(self) -> None: ...


def default_voice(provider: str) -> str:
    """Name the voice a provider uses when the session did not choose one."""
    return DEFAULT_VOICES[provider]


def build_speech_engine(provider, voice=None, output_sink=None) -> SpeechEngine:
    """Build the engine for one provider, with that provider's own pipeline.

    The engine modules are imported here rather than at the top of the file so
    that selecting one provider never pays for the other's dependencies —
    Piper loads onnxruntime, and a session on Edge should not.
    """
    if provider not in PROVIDER_LABELS:
        allowed = ", ".join(repr(name) for name in PROVIDERS)
        raise RuntimeError(f"Unknown speech provider {provider!r}; expected {allowed}.")
    if voice is None:
        voice = default_voice(provider)
    if provider == EDGE:
        from .tts import EdgeSentenceTTS

        return EdgeSentenceTTS(voice, output_sink=output_sink)
    from .piper_tts import PiperSentenceTTS

    return PiperSentenceTTS(voice, output_sink=output_sink)


def provider_switch(speech):
    """Build the interface's speech-provider switch.

    It reports whether the session can switch at all, so a session started
    without speech says so instead of appearing to change an engine that does
    not exist. The new provider speaks with its own default voice: a voice
    name belongs to the engine that defines it, and Edge's would mean nothing
    to Piper.
    """

    def switch(provider):
        if speech is None:
            return False
        return speech.set_provider(provider)

    return switch


class SwitchableSpeech:
    """One speech object whose provider can be replaced while it runs.

    Building an engine is slow enough to matter — Piper loads a model — so a
    switch happens on its own thread and the caller is told immediately
    whether it was accepted. Until the new engine is ready the old one keeps
    answering, which is why every method here delegates through ``engine``
    under the lock rather than caching it.
    """

    def __init__(
        self,
        provider,
        engine,
        output_sink=None,
        build=build_speech_engine,
        stream=None,
    ):
        self.provider = provider
        self.engine = engine
        # Remembered so a switch plays through the same device the session
        # chose at startup. The caller that picked it is long gone by then.
        self.output_sink = output_sink
        self.build = build
        self.stream = stream
        self.lock = threading.Lock()
        self.closed = False
        self.switch = None

    @classmethod
    def start(cls, provider, voice=None, output_sink=None, build=build_speech_engine):
        """Build the session's first engine and wrap it."""
        return cls(
            provider,
            build(provider, voice, output_sink),
            output_sink=output_sink,
            build=build,
        )

    def _current(self):
        with self.lock:
            return self.engine

    def begin_turn(self):
        self._current().begin_turn()

    def speak(self, text):
        self._current().speak(text)

    def interrupt(self):
        self._current().interrupt()

    def set_enabled(self, enabled):
        self._current().set_enabled(enabled)

    def is_likely_echo(self, text):
        return self._current().is_likely_echo(text)

    def is_speaking(self):
        return self._current().is_speaking()

    def set_provider(self, provider, voice=None):
        """Start replacing the engine; report whether the switch was started.

        A switch that is already running wins, and a request for the provider
        already in use is not a switch at all.
        """
        with self.lock:
            if self.closed or provider == self.provider:
                return False
            if self.switch is not None and self.switch.is_alive():
                return False
            self.switch = threading.Thread(
                target=self._switch_to,
                args=(provider, voice, self.output_sink),
                name="SpeechProviderSwitch",
                daemon=True,
            )
            self.switch.start()
        return True

    def _switch_to(self, provider, voice, output_sink):
        """Build the new engine, then retire the old one it replaces."""
        try:
            engine = self.build(provider, voice, output_sink)
        except Exception as error:
            print(
                f"\nCould not switch speech to {provider}: {error}",
                file=sys.stderr if self.stream is None else self.stream,
                flush=True,
            )
            return
        with self.lock:
            # A session that closed while the model loaded gets no speech: the
            # engine just built is shut down instead of installed, and nothing
            # about the facade moves — a half-applied switch would leave the
            # sidebar naming a provider that never started.
            installing = not self.closed
            if installing:
                retired, self.engine, self.provider = self.engine, engine, provider
        if not installing:
            engine.close()
            return
        retired.close()

    def close(self):
        with self.lock:
            self.closed = True
            engine, switch = self.engine, self.switch
        engine.close()
        if switch is not None:
            switch.join(timeout=10)
