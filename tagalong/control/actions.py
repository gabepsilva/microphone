"""The static catalog of things a client may ask the session to do.

One operation is registered once and every client reaches it the same way, so
the Textual sidebar, a future Electron window, and an agent cannot drift into
three different meanings of "mute". Slash commands are not in this catalog:
they are human syntax over these operations, and giving them an entry of their
own would re-import string parsing as an external interface.

The catalog is *static* for a protocol version. Whether the actor holding a
connection may run an operation is authorization (:mod:`.actors`), and whether
the session can honour it right now is applicability, decided by the handler
at dispatch time. Neither removes an entry from this list. That separation is
what lets a generated agent tool schema stay stable while a microphone comes
and goes: muting with no open channel records desired state that the channel
replays when it opens, which is a very different answer from "no such action".

Validation here is deliberately only about shape — types, nullability, and
whether a value is one of a fixed set. Bounds are not repeated: the session
already clamps its turn-silence window, and a second copy of those bounds in a
schema is a copy that can disagree with the one that runs. A request outside
those bounds is accepted, coerced, and answered with the effective value.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..domain import POLICY_NAMES
from ..piper_voices import PIPER_VOICE_IDS
from ..speech import PROVIDERS

# Bumped when an existing action's meaning or payload changes incompatibly.
# Adding an action does not need a bump: a client that has never heard of it
# does not call it.
PROTOCOL_VERSION = 1


class Scope(StrEnum):
    """The authority an operation needs, coarse enough for a human to grant."""

    CONVERSE = "converse"
    SESSION = "session"
    AUDIO = "audio"
    SETTINGS = "settings"
    TRANSCRIPT = "transcript"


class Kind(StrEnum):
    """The shape of one payload value.

    ``TEXT`` and ``NAME`` are both strings and are still worth separating:
    text is prose the session will carry around, a name identifies something
    the session offers, and only a name is ever nullable — a selection of
    ``None`` means "nothing selected", which is not the same as empty prose.
    """

    TEXT = "text"
    NAME = "name"
    FLAG = "flag"
    NUMBER = "number"
    IDS = "ids"
    DATA = "data"


class InvalidPayload(ValueError):
    """The request does not match its action's schema."""


def _check_text(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidPayload("expected text")
    return value


def _check_name(value: object) -> str:
    name = _check_text(value)
    if not name:
        raise InvalidPayload("expected a name, not an empty string")
    return name


def _check_flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidPayload("expected true or false")
    return value


def _check_number(value: object) -> float:
    # bool is an int in Python, and `muted=True` arriving where seconds are
    # expected should read as the mistake it is rather than as 1.0.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidPayload("expected a number")
    if not math.isfinite(value):
        raise InvalidPayload("expected a finite number")
    return float(value)


def _check_ids(value: object) -> tuple[str, ...]:
    # A string is itself a sequence of strings, so passing one where a list of
    # identifiers belongs would otherwise be silently accepted as its letters.
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise InvalidPayload("expected a list of identifiers")
    checked: list[str] = []
    for position, item in enumerate(value):
        try:
            checked.append(_check_name(item))
        except InvalidPayload as error:
            raise InvalidPayload(f"identifier {position}: {error}") from error
    return tuple(checked)


def _check_data(value: object) -> bytes:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str):
        # JSON-RPC and MCP carry binary as base64; in-process callers pass bytes.
        try:
            return base64.b64decode(value, validate=True)
        except binascii.Error as error:
            raise InvalidPayload("expected binary data") from error
    raise InvalidPayload("expected binary data")


_CHECKS = {
    Kind.TEXT: _check_text,
    Kind.NAME: _check_name,
    Kind.FLAG: _check_flag,
    Kind.NUMBER: _check_number,
    Kind.IDS: _check_ids,
    Kind.DATA: _check_data,
}


@dataclass(frozen=True)
class Parameter:
    """One payload value: its shape, whether it is required, and its choices."""

    name: str
    kind: Kind
    required: bool = True
    nullable: bool = False
    choices: tuple[str, ...] = ()
    default: object = None

    def check(self, value: object) -> object:
        """Return *value* normalized to this parameter's kind, or raise.

        Every refusal names this parameter. A caller sending four values wants
        to know which one it got wrong, and "expected text" on its own leaves
        that to guesswork.
        """
        try:
            return self._checked(value)
        except InvalidPayload as error:
            raise InvalidPayload(f"{self.name}: {error}") from error

    def _checked(self, value: object) -> object:
        if value is None:
            if not self.nullable:
                raise InvalidPayload("expected a value, not null")
            return None
        checked = _CHECKS[self.kind](value)
        if self.choices and checked not in self.choices:
            allowed = ", ".join(self.choices)
            raise InvalidPayload(f"expected one of {allowed}")
        return checked


@dataclass(frozen=True)
class ActionSpec:
    """One operation: what it is called, what it needs, and who may call it."""

    id: str
    summary: str
    scope: Scope
    parameters: tuple[Parameter, ...] = ()

    def validate(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Return the normalized payload for this action, or raise.

        Unknown keys are refused rather than dropped. A caller that misspells
        ``respond`` means something by it, and silently sending the message
        with the default is the outcome hardest to notice.

        What comes back is read-only and holds only immutable values — a list
        of attachment ids arrives as a tuple, a whole number of seconds as a
        float, an omitted optional as its default. It is therefore a canonical
        description of the request rather than a view of what the caller still
        owns: two spellings of the same request normalize to the same payload,
        and a caller that goes on editing its own dict cannot change what the
        session understood it to have asked for.
        """
        known = {parameter.name for parameter in self.parameters}
        unexpected = sorted(set(payload) - known)
        if unexpected:
            raise InvalidPayload(f"{self.id}: unexpected: {', '.join(unexpected)}")

        checked: dict[str, object] = {}
        for parameter in self.parameters:
            if parameter.name not in payload:
                if parameter.required:
                    raise InvalidPayload(f"{self.id}: {parameter.name}: required")
                checked[parameter.name] = parameter.default
                continue
            try:
                checked[parameter.name] = parameter.check(payload[parameter.name])
            except InvalidPayload as error:
                raise InvalidPayload(f"{self.id}: {error}") from error
        return MappingProxyType(checked)


# A device or application is chosen from a catalog the session refreshes while
# it runs, and "nothing selected" is one of the choices — so both selections
# are nullable names rather than required ones.
SELECTION = Parameter("name", Kind.NAME, nullable=True)

CATALOG: tuple[ActionSpec, ...] = (
    ActionSpec(
        "message.send",
        "Send a message to Taga as this actor",
        Scope.CONVERSE,
        (
            Parameter("text", Kind.TEXT),
            Parameter("images", Kind.IDS, required=False, default=()),
            Parameter("respond", Kind.FLAG, required=False, default=True),
        ),
    ),
    ActionSpec(
        "attachment.upload",
        "Upload an image and receive an identifier to send with a message",
        Scope.CONVERSE,
        (Parameter("data", Kind.DATA),),
    ),
    ActionSpec(
        "transcript.append",
        "Add context to the transcript without asking for a reply",
        Scope.TRANSCRIPT,
        (Parameter("text", Kind.TEXT),),
    ),
    ActionSpec(
        "session.new", "Start a fresh session and clear the transcript", Scope.SESSION
    ),
    ActionSpec(
        "session.interrupt",
        "Stop the reply in progress",
        Scope.SESSION,
        (
            Parameter(
                "generation", Kind.NUMBER, required=False, nullable=True, default=None
            ),
        ),
    ),
    ActionSpec(
        "session.quit",
        "Shut down the running session",
        Scope.SESSION,
    ),
    ActionSpec("voice.end_turn", "Submit the pending spoken turn now", Scope.SESSION),
    ActionSpec(
        "microphone.select",
        "Listen to a named input device, or to none",
        Scope.AUDIO,
        (SELECTION,),
    ),
    ActionSpec(
        "microphone.set_muted",
        "Stop or resume transcribing the microphone",
        Scope.AUDIO,
        (Parameter("muted", Kind.FLAG),),
    ),
    ActionSpec(
        "audio_stream.select",
        "Listen to a named application as the far end, or to none",
        Scope.AUDIO,
        (SELECTION,),
    ),
    ActionSpec(
        "audio_stream.set_muted",
        "Stop or resume transcribing the far end",
        Scope.AUDIO,
        (Parameter("muted", Kind.FLAG),),
    ),
    ActionSpec(
        "response_policy.set",
        "Choose which speakers Taga answers out loud",
        Scope.SETTINGS,
        (Parameter("policy", Kind.NAME, choices=POLICY_NAMES),),
    ),
    ActionSpec(
        "tts.set_enabled",
        "Turn spoken replies on or off",
        Scope.SETTINGS,
        (Parameter("enabled", Kind.FLAG),),
    ),
    ActionSpec(
        "tts.set_provider",
        "Choose the speech engine",
        Scope.SETTINGS,
        (Parameter("provider", Kind.NAME, choices=PROVIDERS),),
    ),
    ActionSpec(
        "tts.set_voice",
        "Choose the speech voice for the current engine",
        Scope.SETTINGS,
        (Parameter("voice", Kind.NAME, choices=PIPER_VOICE_IDS),),
    ),
    ActionSpec(
        "codex.set_model",
        "Choose the Codex model",
        Scope.SETTINGS,
        (Parameter("model", Kind.NAME),),
    ),
    ActionSpec(
        "codex.set_reasoning",
        "Choose the reasoning effort",
        Scope.SETTINGS,
        (Parameter("effort", Kind.NAME),),
    ),
    ActionSpec(
        "turn_silence.set",
        "Set how long a pause ends a spoken turn",
        Scope.SETTINGS,
        (Parameter("seconds", Kind.NUMBER),),
    ),
    ActionSpec("transcript.save", "Write the transcript to disk", Scope.TRANSCRIPT),
)


@dataclass(frozen=True)
class Capability:
    """One catalog entry paired with whether this actor may invoke it.

    Applicability is deliberately absent. Whether the session can honour the
    action *right now* changes with every device that appears, and answering
    it here would make discovery a snapshot that is stale before it is read.
    """

    action: ActionSpec
    allowed: bool
