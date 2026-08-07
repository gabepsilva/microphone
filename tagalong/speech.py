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

    def wait_ready(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


def default_voice(provider: str) -> str:
    """Name the voice a provider uses when the session did not choose one."""
    return DEFAULT_VOICES[provider]


def _engine_voice(engine, provider, voice=None):
    """Prefer the voice the engine records; fall back to what the caller asked."""
    recorded = getattr(engine, "voice", None)
    if recorded is not None:
        return recorded
    if voice is not None:
        return voice
    return default_voice(provider)


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
        # The voice the installed engine was built with. Provider switches and
        # voice-only switches both update it once the new engine is ready —
        # never when the thread merely starts — so clients do not advertise a
        # name that never loaded.
        self.voice = _engine_voice(engine, provider)
        # Whether replies are spoken belongs to the session, not to whichever
        # engine happens to be installed: a muted session that switches
        # providers is still muted, and a freshly built engine speaks unless
        # it is told otherwise.
        self.enabled = True

    @classmethod
    def start(cls, provider, voice=None, output_sink=None, build=build_speech_engine):
        """Build the session's first engine and wrap it."""
        engine = build(provider, voice, output_sink)
        speech = cls(
            provider,
            engine,
            output_sink=output_sink,
            build=build,
        )
        # Thread the caller's voice through even when a fake builder omits
        # ``engine.voice`` — otherwise the facade reports the provider default
        # while the engine was asked for something else.
        speech.voice = _engine_voice(engine, provider, voice)
        return speech

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
        with self.lock:
            self.enabled = enabled
            engine = self.engine
        engine.set_enabled(enabled)

    def is_likely_echo(self, text):
        return self._current().is_likely_echo(text)

    def is_speaking(self):
        return self._current().is_speaking()

    def wait_ready(self, timeout: float | None = None) -> None:
        """Forward readiness to the installed engine."""
        self._current().wait_ready(timeout)

    @property
    def switching(self) -> bool:
        """True while a provider or voice switch thread is still running."""
        switch = self.switch
        return switch is not None and switch.is_alive()

    def set_provider(self, provider, voice=None, *, on_applied=None, on_failed=None):
        """Start replacing the engine; report whether the switch was started.

        A switch that is already running wins, and a request for the provider
        already in use is not a switch at all. Without a voice the new
        provider speaks with its own default: a voice name belongs to the
        engine that defines it, and Edge's would mean nothing to Piper.
        """
        with self.lock:
            if self.closed or provider == self.provider:
                return False
            if self.switch is not None and self.switch.is_alive():
                return False
            self.switch = threading.Thread(
                target=self._switch_to,
                args=(provider, voice, self.output_sink, on_applied, on_failed),
                name="SpeechProviderSwitch",
                daemon=True,
            )
            self.switch.start()
        return True

    def set_voice(self, voice, *, on_applied=None, on_failed=None):
        """Start rebuilding the current provider with a different voice.

        Same-provider voice changes cannot ride on :meth:`set_provider`, which
        refuses when the provider is unchanged. Failure leaves the prior engine
        installed; ``on_applied`` / ``on_failed`` fire after readiness settles.
        """
        with self.lock:
            if self.closed or voice == self.voice:
                return False
            if self.switch is not None and self.switch.is_alive():
                return False
            self.switch = threading.Thread(
                target=self._switch_to,
                args=(self.provider, voice, self.output_sink, on_applied, on_failed),
                name="SpeechVoiceSwitch",
                daemon=True,
            )
            self.switch.start()
        return True

    def _switch_to(self, provider, voice, output_sink, on_applied=None, on_failed=None):
        """Build the new engine, wait until it is ready, then retire the old one."""
        engine = None
        try:
            engine = self.build(provider, voice, output_sink)
            engine.wait_ready()
        except Exception as error:
            if engine is not None:
                engine.close()
            target = (
                f"{provider} voice {voice!r}" if voice is not None else f"{provider}"
            )
            print(
                f"\nCould not switch speech to {target}: {error}",
                file=sys.stderr if self.stream is None else self.stream,
                flush=True,
            )
            if on_failed is not None:
                on_failed(str(error))
            return
        applied_voice = _engine_voice(engine, provider, voice)
        with self.lock:
            # A session that closed while the model loaded gets no speech: the
            # engine just built is shut down instead of installed, and nothing
            # about the facade moves — a half-applied switch would leave the
            # sidebar naming a provider that never started.
            installing = not self.closed
            if installing:
                # Muted before the switch means muted after it. The new engine
                # is silenced under the same lock that installs it, so no
                # sentence can reach it in between.
                if not self.enabled:
                    engine.set_enabled(False)
                retired, self.engine, self.provider = self.engine, engine, provider
                self.voice = applied_voice
        if not installing:
            engine.close()
            if on_failed is not None:
                on_failed("speech session closed during switch")
            return
        retired.close()
        if on_applied is not None:
            on_applied(applied_voice)

    def close(self):
        with self.lock:
            self.closed = True
            engine, switch = self.engine, self.switch
        engine.close()
        if switch is not None:
            switch.join(timeout=10)
