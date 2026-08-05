from __future__ import annotations

import dataclasses

import pytest

from tagalong.control.state import (
    AppState,
    Effect,
    Selection,
    with_desired,
    with_effective,
)


def test_state_a_client_is_holding_cannot_be_changed_under_it() -> None:
    """Two frontends reading one mutable session object is the failure here."""
    state = AppState()

    # Through a named field because a literal assignment is a type error and a
    # literal setattr is a lint error; the runtime guard is the one a client
    # holding a snapshot actually relies on.
    field = "tts_enabled"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(state, field, False)


def test_recording_what_was_asked_for_leaves_what_was_reached() -> None:
    open_channel = with_effective(AppState(), "microphone", "Yeti")

    asked = with_desired(open_channel, "microphone", "Webcam")

    assert asked.microphone == Selection(desired="Webcam", effective="Yeti")


def test_recording_what_was_reached_leaves_what_was_asked_for() -> None:
    asked = with_desired(AppState(), "audio_stream", "Zoom")

    reached = with_effective(asked, "audio_stream", "Zoom")

    assert reached.audio_stream == Selection(desired="Zoom", effective="Zoom")


def test_a_selection_change_leaves_the_other_selection_alone() -> None:
    state = with_desired(AppState(), "microphone", "Yeti")

    changed = with_desired(state, "audio_stream", "Zoom")

    assert changed.microphone == Selection(desired="Yeti")


def test_only_what_moved_is_reported_as_changed() -> None:
    """Eleven unchanged fields are where an unexpected change hides."""
    before = AppState()
    after = dataclasses.replace(before, tts_enabled=False, codex_model="luna")

    assert after.changed_from(before) == {
        "tts_enabled": False,
        "codex_model": "luna",
    }


def test_state_that_did_not_move_reports_nothing() -> None:
    assert AppState().changed_from(AppState()) == {}


def test_a_finished_effect_names_the_value_the_session_settled_on() -> None:
    effect = Effect.applied(AppState(), 30.0)

    assert effect.effective == 30.0
    assert effect.settle is None


def test_a_pending_effect_carries_the_way_to_fold_in_the_result() -> None:
    def settle(state: AppState, effective: object) -> AppState:
        assert isinstance(effective, str)
        return with_effective(state, "microphone", effective)

    effect = Effect.pending(AppState(), settle)

    assert effect.settle is not None
    assert effect.settle(effect.state, "Yeti").microphone.effective == "Yeti"
