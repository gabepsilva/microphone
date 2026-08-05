"""The canonical state every client renders, and how a handler changes it.

State is immutable and replaced wholesale. A snapshot handed to a client is
therefore a value it can hold for as long as it likes, and no client can
mutate what another one is reading — which is the failure a shared mutable
session object invites the moment a second frontend exists.

Two fields carry a desired value and an effective one, and they are exactly
the two the session applies asynchronously: choosing a microphone or a far end
starts work on another thread that can succeed, fail, or be overtaken. Nothing
else here is split, because nothing else has an independent effective reading
to report — a mute the session records before any channel is open is desired
state that the channel replays when it opens, and inventing an "effective
mute" to sit beside it would be a field with no source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ..speech import DEFAULT_PROVIDER


@dataclass(frozen=True)
class Selection:
    """What was asked for, and what the session actually reached."""

    desired: str | None = None
    effective: str | None = None


@dataclass(frozen=True)
class AppState:
    """Everything a client needs to draw the session, and nothing it draws with."""

    microphone: Selection = field(default_factory=Selection)
    microphone_muted: bool = False
    audio_stream: Selection = field(default_factory=Selection)
    audio_stream_muted: bool = False
    # Answering both the room and the far end, the same default the interface
    # starts with. A test pins it against the policy catalog so a rename there
    # cannot leave this holding a policy that no longer exists.
    response_policy: str = "both"
    tts_enabled: bool = True
    tts_provider: str = DEFAULT_PROVIDER
    codex_model: str = ""
    codex_reasoning: str = ""
    turn_silence: float = 3.0

    def changed_from(self, older: AppState) -> dict[str, object]:
        """Return the fields that differ, for a ``state.changed`` payload.

        Publishing the difference rather than the whole state keeps an event
        readable in a log and makes an unexpected change visible instead of
        buried in twelve unchanged fields.
        """
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(older, name)
        }


def with_desired(state: AppState, name: str, value: str | None) -> AppState:
    """Record what a selection field was asked for, leaving what it reached."""
    selection: Selection = getattr(state, name)
    return replace(state, **{name: replace(selection, desired=value)})


def with_effective(state: AppState, name: str, value: str | None) -> AppState:
    """Record what a selection field reached, leaving what it was asked for."""
    selection: Selection = getattr(state, name)
    return replace(state, **{name: replace(selection, effective=value)})


@dataclass(frozen=True)
class Effect:
    """What a handler did: the new state, and whether the work is finished.

    A handler that finished names the effective value it settled on, which is
    what the caller is answered with — the requested value is never assumed,
    because a setting that coerces would then report a number the session is
    not running.

    A handler that started slow work instead supplies ``settle``, the function
    that will fold the eventual effective value into the state. Keeping that
    here rather than in the controller is what lets the controller stay
    ignorant of which field an action moves — and since it stays ignorant, the
    value a reconciler reports arrives as ``object``. The handler that started
    the work is the one place that knows what type it will be, so narrowing it
    is that handler's job.
    """

    state: AppState
    effective: object = None
    settle: Callable[[AppState, object], AppState] | None = None

    @classmethod
    def applied(cls, state: AppState, effective: object = None) -> Effect:
        """The work is done; *effective* is what the session now holds."""
        return cls(state, effective)

    @classmethod
    def pending(
        cls, state: AppState, settle: Callable[[AppState, object], AppState]
    ) -> Effect:
        """Desired state is recorded; a reconciler will report the outcome."""
        return cls(state, settle=settle)
