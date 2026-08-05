from __future__ import annotations

from typing import cast

import pytest

from tagalong.control.actions import (
    CATALOG,
    PROTOCOL_VERSION,
    ActionSpec,
    Capability,
    InvalidPayload,
    Kind,
    Parameter,
    Scope,
)
from tagalong.control.actors import Actor, agent, local_user
from tagalong.control.policy import authorizes
from tagalong.control.state import AppState
from tagalong.domain import POLICY_NAMES
from tagalong.speech import PROVIDERS


def action(action_id: str) -> ActionSpec:
    """The catalog entry under test, by id."""
    for spec in CATALOG:
        if spec.id == action_id:
            return spec
    raise AssertionError(f"catalog has no {action_id}")


def test_every_catalog_entry_has_a_distinct_id() -> None:
    ids = [spec.id for spec in CATALOG]

    assert len(ids) == len(set(ids))
    assert PROTOCOL_VERSION == 1


def test_slash_commands_are_not_an_action() -> None:
    """Commands are human syntax over these actions, never a second API."""
    assert not [spec for spec in CATALOG if spec.id.startswith("command.")]
    assert "commands.list" not in {spec.id for spec in CATALOG}
    assert action("session.new").scope is Scope.SESSION


def test_an_interrupt_may_name_the_generation_it_targets() -> None:
    """Omitting generation interrupts whatever is current; naming one pins it."""
    assert action("session.interrupt").validate({}) == {"generation": None}
    assert action("session.interrupt").validate({"generation": 3}) == {
        "generation": 3.0
    }


def test_the_catalog_takes_its_choices_from_the_session_that_runs_them() -> None:
    """A schema copy of a fixed set is a copy that can disagree with the set."""
    assert action("response_policy.set").parameters[0].choices == POLICY_NAMES
    assert action("tts.set_provider").parameters[0].choices == PROVIDERS
    assert AppState().response_policy in POLICY_NAMES
    assert AppState().tts_provider in PROVIDERS


def test_a_valid_payload_comes_back_normalized() -> None:
    checked = action("message.send").validate({"text": "hello", "images": ["a1"]})

    assert checked == {"text": "hello", "images": ("a1",), "respond": True}


def test_an_omitted_optional_value_takes_its_documented_default() -> None:
    checked = action("message.send").validate({"text": "hi"})

    assert checked["images"] == ()
    assert checked["respond"] is True


@pytest.mark.parametrize(
    ("action_id", "payload", "message"),
    [
        ("message.send", {}, "message.send: text: required"),
        (
            "message.send",
            {"text": "hi", "respnod": False},
            "message.send: unexpected: respnod",
        ),
        ("message.send", {"text": 3}, "message.send: text: expected text"),
        (
            "message.send",
            {"text": None},
            "message.send: text: expected a value, not null",
        ),
        # A string is itself a sequence of strings: accepted, "abc" would send
        # a three-attachment list of a, b and c.
        (
            "message.send",
            {"text": "hi", "images": "abc"},
            "message.send: images: expected a list of identifiers",
        ),
        (
            "message.send",
            {"text": "hi", "images": [1]},
            "message.send: images: identifier 0: expected text",
        ),
        (
            "message.send",
            {"text": "hi", "images": ["ok", ""]},
            "message.send: images: identifier 1: expected a name, not an empty string",
        ),
        (
            "turn_silence.set",
            {"seconds": "soon"},
            "turn_silence.set: seconds: expected a number",
        ),
        (
            "turn_silence.set",
            {"seconds": float("inf")},
            "turn_silence.set: seconds: expected a finite number",
        ),
        (
            "microphone.set_muted",
            {"muted": 1},
            "microphone.set_muted: muted: expected true or false",
        ),
        (
            "microphone.select",
            {"name": ""},
            "microphone.select: name: expected a name, not an empty string",
        ),
        # A choice outside the fixed set the session offers.
        (
            "response_policy.set",
            {"policy": "sometimes"},
            "response_policy.set: policy: expected one of audio, both, voice, quiet",
        ),
        (
            "attachment.upload",
            {"data": "PNG"},
            "attachment.upload: data: expected binary data",
        ),
    ],
)
def test_a_refusal_says_which_action_which_value_and_what_was_expected(
    action_id, payload, message
) -> None:
    """The caller has to be able to correct the request from the answer alone."""
    with pytest.raises(InvalidPayload) as refusal:
        action(action_id).validate(payload)

    assert str(refusal.value) == message


def test_a_whole_number_of_seconds_is_still_a_number() -> None:
    checked = action("turn_silence.set").validate({"seconds": 3})

    assert checked == {"seconds": 3.0}
    assert isinstance(checked["seconds"], float)


@pytest.mark.parametrize("value", [True, "3", None, float("nan"), float("inf")])
def test_a_turn_silence_window_that_is_not_a_finite_number_is_refused(value) -> None:
    """`muted=True` where seconds belong is a mistake, not 1.0 seconds."""
    with pytest.raises(InvalidPayload, match="seconds"):
        action("turn_silence.set").validate({"seconds": value})


@pytest.mark.parametrize("value", [1, "yes", None])
def test_a_flag_that_is_not_a_boolean_is_refused(value) -> None:
    with pytest.raises(InvalidPayload, match="muted"):
        action("microphone.set_muted").validate({"muted": value})


def test_selecting_nothing_is_a_selection() -> None:
    assert action("microphone.select").validate({"name": None}) == {"name": None}
    assert action("audio_stream.select").validate({"name": None}) == {"name": None}


def test_uploaded_bytes_arrive_as_bytes() -> None:
    checked = action("attachment.upload").validate({"data": bytearray(b"PNG")})

    assert checked == {"data": b"PNG"}


def test_uploaded_base64_arrives_as_bytes() -> None:
    import base64

    checked = action("attachment.upload").validate(
        {"data": base64.b64encode(b"PNG").decode("ascii")}
    )

    assert checked == {"data": b"PNG"}


def test_uploaded_data_rejects_non_binary_shapes() -> None:
    with pytest.raises(InvalidPayload) as int_shape:
        action("attachment.upload").validate({"data": 12})
    assert str(int_shape.value) == "attachment.upload: data: expected binary data"
    with pytest.raises(InvalidPayload) as bad_b64:
        action("attachment.upload").validate({"data": "!!!!"})
    assert str(bad_b64.value) == "attachment.upload: data: expected binary data"


def test_session_quit_is_in_the_catalog() -> None:
    assert action("session.quit").scope is Scope.SESSION
    assert action("session.quit").validate({}) == {}


def test_a_value_may_be_null_only_where_the_action_says_so() -> None:
    required = Parameter("name", Kind.NAME)

    with pytest.raises(InvalidPayload, match="expected a value, not null"):
        required.check(None)


def test_the_person_at_the_session_may_do_everything_in_the_catalog() -> None:
    from tagalong.control.actors import ActorKind

    owner = local_user()

    assert [spec for spec in CATALOG if not authorizes(owner, spec)] == []
    assert owner.id == "local"
    assert owner.kind is ActorKind.HUMAN
    assert local_user("gabriel").id == "gabriel"


def test_an_agent_holds_only_the_scopes_it_was_granted() -> None:
    from tagalong.control.actors import ActorKind

    reader = agent("notes-bot", {Scope.TRANSCRIPT})

    assert reader.kind is ActorKind.AGENT
    assert authorizes(reader, action("transcript.append"))
    assert not authorizes(reader, action("microphone.select"))
    assert not authorizes(reader, action("message.send"))


def test_an_actor_with_no_scopes_may_do_nothing() -> None:
    """An unconsidered actor is the one that must not be the permissive case."""
    nobody = Actor("nobody")

    assert [spec for spec in CATALOG if authorizes(nobody, spec)] == []


def test_a_capability_pairs_a_static_action_with_this_actor_s_authority() -> None:
    entry = Capability(action("message.send"), allowed=False)

    assert entry.action.id == "message.send"
    assert entry.allowed is False


def test_a_validated_payload_is_a_canonical_description_not_a_view() -> None:
    """What a retry has to match cannot be something the caller still owns."""
    images = ["img-a"]

    checked = action("message.send").validate({"text": "hi", "images": images})
    images[0] = "img-b"

    assert checked == {"text": "hi", "images": ("img-a",), "respond": True}
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(dict, checked)["text"] = "bye"
