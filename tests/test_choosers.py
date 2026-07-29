"""Device discovery and the interactive startup questions.

Every audio boundary is faked here: PortAudio, ``pactl``, and the terminal.
The prompt loops are driven through the real ``input`` call site so that a
rejected answer is proven to re-prompt rather than end startup.
"""

from __future__ import annotations

import builtins
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from voice_codex.catalog import CodexModelOption, populate_codex_model_catalog
from voice_codex.choosers import (
    audio_outputs,
    choose_codex_after,
    choose_microphone,
    choose_them_stream,
    choose_tts,
    choose_tts_output,
    find_audio_output,
    input_devices,
    prompt_number,
    prompt_until,
    select_microphone,
    select_them_stream,
    select_tts_output,
)
from voice_codex.domain import RESPONSE_POLICIES
from voice_codex.streams import ApplicationStream, stream_label


def answer_with(monkeypatch, answers):
    """Feed the prompt loops a fixed sequence of typed answers."""
    remaining = list(answers)
    asked: list[str] = []

    def fake_input(prompt=""):
        asked.append(prompt)
        return remaining.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    return asked


def fake_sounddevice(monkeypatch, devices=None, error=None):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            if error is not None:
                raise error
            return SimpleNamespace(query_devices=lambda: devices)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class RecordingStatus:
    """A session-status display that keeps what the catalog probe reported."""

    def __init__(self):
        self.notes: list[str] = []
        self.errors: list[str] = []
        self.fields: dict[str, object] = {}
        self.catalog: dict[str, object] = {}

    def note(self, text):
        self.notes.append(text)

    def error(self, message):
        self.errors.append(message)

    def set_codex(self, **fields):
        self.fields.update(fields)

    def set_codex_catalog(self, models, efforts_by_model, default_effort_by_model):
        self.catalog = {
            "models": models,
            "efforts": efforts_by_model,
            "defaults": default_effort_by_model,
        }


def fake_pactl(monkeypatch, stdout="[]", found=True, error=None):
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/pactl" if found else None
    )

    def fake_run(_command, **_kwargs):
        if error is not None:
            raise error
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)


SINKS = [
    {"name": "sink.a", "monitor_source": "sink.a.monitor", "description": "Speakers"},
    {"name": "sink.b", "monitor_source": "sink.b.monitor", "description": "Headset"},
]
OUTPUTS = [
    {"name": "sink.a", "monitor": "sink.a.monitor", "description": "Speakers"},
    {"name": "sink.b", "monitor": "sink.b.monitor", "description": "Headset"},
]
DEVICES = [
    (0, {"name": "Yeti", "default_samplerate": 48000.0}),
    (2, {"name": "Webcam", "default_samplerate": 16000.0}),
]


def test_a_rejected_answer_is_asked_again_instead_of_ending_startup(
    monkeypatch, capsys
) -> None:
    answer_with(monkeypatch, ["nonsense", "", "7"])

    assert prompt_until("pick: ", int, "try again") == 7
    assert capsys.readouterr().out.count("try again") == 2


def test_a_number_outside_the_range_is_asked_again(monkeypatch, capsys) -> None:
    asked = answer_with(monkeypatch, ["0", "9", "not a number", "3"])

    assert prompt_number("pick: ", 1, 4, "1 to 4 please") == 3
    assert capsys.readouterr().out.count("1 to 4 please") == 3
    assert asked == ["pick: "] * 4


@pytest.mark.parametrize("answer", ["1", "4"])
def test_both_ends_of_the_range_are_accepted(monkeypatch, answer) -> None:
    answer_with(monkeypatch, [answer])

    assert prompt_number("pick: ", 1, 4, "retry") == int(answer)


def test_only_devices_with_input_channels_are_offered(monkeypatch) -> None:
    fake_sounddevice(
        monkeypatch,
        devices=[
            {"name": "Yeti", "max_input_channels": 2},
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Webcam", "max_input_channels": 1},
        ],
    )

    assert [name["name"] for _, name in input_devices()] == ["Yeti", "Webcam"]
    assert [index for index, _ in input_devices()] == [0, 2]


def test_a_missing_portaudio_library_is_reported_as_a_runtime_error(
    monkeypatch,
) -> None:
    fake_sounddevice(monkeypatch, error=OSError("libportaudio.so not found"))

    with pytest.raises(RuntimeError, match="PortAudio system library"):
        input_devices()


@pytest.mark.parametrize(
    ("requested", "expected_index"),
    [("Yeti", 0), ("Webcam", 2), ("0", 0), ("2", 2), (2, 2)],
)
def test_a_microphone_is_found_by_index_or_by_exact_name(
    requested, expected_index
) -> None:
    index, _ = select_microphone(DEVICES, requested)

    assert index == expected_index


def test_an_unknown_microphone_names_the_config_that_asked_for_it() -> None:
    with pytest.raises(RuntimeError, match="'Studio Mic' was not found"):
        select_microphone(DEVICES, "Studio Mic")


def test_choosing_a_microphone_lists_every_device_before_asking(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("voice_codex.choosers.input_devices", lambda: DEVICES)
    answer_with(monkeypatch, ["2"])

    assert choose_microphone() == DEVICES[1]
    menu = capsys.readouterr().out
    assert "1) Yeti (device 0, 48000 Hz)" in menu
    assert "2) Webcam (device 2, 16000 Hz)" in menu


def test_choosing_a_microphone_fails_when_the_machine_has_none(monkeypatch) -> None:
    monkeypatch.setattr("voice_codex.choosers.input_devices", list)

    with pytest.raises(RuntimeError, match="No audio input devices"):
        choose_microphone()


def test_audio_outputs_pair_each_sink_with_its_monitor(monkeypatch) -> None:
    fake_pactl(monkeypatch, stdout=json.dumps(SINKS))

    assert audio_outputs() == OUTPUTS


def test_audio_outputs_are_empty_without_pipewire(monkeypatch) -> None:
    fake_pactl(monkeypatch, found=False)

    assert audio_outputs() == []


@pytest.mark.parametrize(
    "error",
    [
        OSError("pactl vanished"),
        subprocess.CalledProcessError(1, "pactl"),
    ],
)
def test_a_failing_pactl_yields_no_outputs_instead_of_raising(
    monkeypatch, error
) -> None:
    fake_pactl(monkeypatch, error=error)

    assert audio_outputs() == []


def test_unparseable_pactl_output_yields_no_outputs(monkeypatch) -> None:
    fake_pactl(monkeypatch, stdout="not json")

    assert audio_outputs() == []


def test_a_sink_without_a_monitor_is_skipped(monkeypatch) -> None:
    fake_pactl(
        monkeypatch,
        stdout=json.dumps(
            [
                {"name": "sink.a", "description": "No monitor"},
                {"monitor_source": "orphan.monitor", "description": "No name"},
                SINKS[0],
            ]
        ),
    )

    assert audio_outputs() == [OUTPUTS[0]]


def test_a_sink_description_falls_back_to_its_properties_then_its_name(
    monkeypatch,
) -> None:
    fake_pactl(
        monkeypatch,
        stdout=json.dumps(
            [
                {
                    "name": "sink.a",
                    "monitor_source": "sink.a.monitor",
                    "properties": {"device.description": "From properties"},
                },
                {"name": "sink.b", "monitor_source": "sink.b.monitor"},
            ]
        ),
    )

    assert [output["description"] for output in audio_outputs()] == [
        "From properties",
        "sink.b",
    ]


@pytest.mark.parametrize("requested", ["sink.b", "sink.b.monitor", "Headset"])
def test_an_output_is_matched_by_sink_monitor_or_description(requested) -> None:
    assert find_audio_output(OUTPUTS, requested) is OUTPUTS[1]


def test_an_unknown_output_matches_nothing() -> None:
    assert find_audio_output(OUTPUTS, "sink.c") is None


@pytest.mark.parametrize("requested", ["none", "NONE", "None"])
def test_no_them_application_is_accepted_in_any_case(requested) -> None:
    assert select_them_stream(requested) is None


def test_a_named_application_is_taken_without_checking_the_graph() -> None:
    """An application that is not playing yet still has to be selectable."""
    assert select_them_stream("ZOOM VoiceEngine") == "ZOOM VoiceEngine"


STREAMS = [
    ApplicationStream(
        node_id=41,
        application="Chromium",
        title="Playback",
        binary="chromium",
        playing=True,
    ),
    ApplicationStream(
        node_id=42,
        application="ZOOM VoiceEngine",
        title="ZOOM VoiceEngine",
        binary="zoom",
        playing=False,
    ),
]


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (STREAMS[0], "Chromium — Playback (playing)"),
        (STREAMS[1], "ZOOM VoiceEngine (idle)"),
    ],
)
def test_a_stream_is_labelled_by_application_title_and_whether_it_plays(
    stream, expected
) -> None:
    assert stream_label(stream) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("0", None), ("1", "Chromium"), ("2", "ZOOM VoiceEngine")],
)
def test_choosing_a_them_application_maps_each_menu_entry(
    monkeypatch, answer, expected
) -> None:
    monkeypatch.setattr("voice_codex.choosers.graph", list)
    monkeypatch.setattr(
        "voice_codex.choosers.offered_applications",
        lambda objects: [(stream_label(s), s.application) for s in STREAMS],
    )
    answer_with(monkeypatch, [answer])

    assert choose_them_stream() == expected


def test_a_silent_machine_says_what_to_do_instead_of_offering_nothing(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("voice_codex.choosers.graph", list)
    monkeypatch.setattr("voice_codex.choosers.offered_applications", lambda objects: [])
    answer_with(monkeypatch, ["0"])

    assert choose_them_stream() is None
    assert "No application is playing audio yet" in capsys.readouterr().out


def test_a_requested_speech_output_is_matched() -> None:
    assert select_tts_output(OUTPUTS, "Speakers") is OUTPUTS[0]


def test_an_unknown_speech_output_is_rejected() -> None:
    with pytest.raises(RuntimeError, match=r"'sink\.c' was not found"):
        select_tts_output(OUTPUTS, "sink.c")


def test_no_requested_speech_output_leaves_the_system_default_alone(
    monkeypatch,
) -> None:
    """Nothing may prompt here: the session no longer has to settle this."""

    def unreachable():
        raise AssertionError("the sinks were listed for a question nobody asked")

    monkeypatch.setattr("voice_codex.choosers.audio_outputs", unreachable)

    assert choose_tts_output() is None


def test_a_requested_speech_output_is_resolved_against_the_sinks(monkeypatch) -> None:
    monkeypatch.setattr("voice_codex.choosers.audio_outputs", lambda: list(OUTPUTS))

    assert choose_tts_output("Headset")["name"] == "sink.b"


@pytest.mark.parametrize(
    ("requested", "label"),
    [
        ("them", "Them"),
        ("both", "User Voice and Them"),
        ("user", "User Voice"),
        ("quiet", "Codex will be quiet for voice"),
    ],
)
def test_a_requested_response_policy_needs_no_prompt(requested, label) -> None:
    assert choose_codex_after(requested).label == label


def test_choosing_a_response_policy_accepts_its_menu_number(
    monkeypatch, capsys
) -> None:
    answer_with(monkeypatch, ["9", "2"])

    policy = choose_codex_after()

    assert (policy.name, policy.label, policy.speakers) == (
        "both",
        "User Voice and Them",
        frozenset({"User Voice", "Them"}),
    )
    assert "Please enter a number from 1 to 4." in capsys.readouterr().out


@pytest.mark.parametrize(("requested", "expected"), [("on", True), ("off", False)])
def test_a_requested_tts_setting_needs_no_prompt(requested, expected) -> None:
    assert choose_tts(requested) is expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("1", False),
        ("no", False),
        ("n", False),
        ("2", True),
        ("yes", True),
        ("y", True),
    ],
)
def test_every_accepted_tts_answer_maps_to_a_setting(
    monkeypatch, answer, expected
) -> None:
    answer_with(monkeypatch, [answer])

    assert choose_tts() is expected


def test_an_unrecognised_tts_answer_is_asked_again(monkeypatch, capsys) -> None:
    answer_with(monkeypatch, ["maybe", "y"])

    assert choose_tts() is True
    assert "Please enter 1 or 2." in capsys.readouterr().out


def test_the_model_catalog_populates_the_sidebar_selectors(monkeypatch) -> None:
    option = CodexModelOption("slug-a", "Model A", ("low", "high"), "high")
    monkeypatch.setattr("voice_codex.catalog.probe_codex_models", lambda: [option])
    recorded = {}

    display = RecordingStatus()
    populate_codex_model_catalog(display)
    recorded = display.catalog

    assert recorded["models"] == [("Model A", "slug-a")]
    assert recorded["efforts"] == {"slug-a": ["low", "high"]}
    assert recorded["defaults"] == {"slug-a": "high"}
    assert display.notes == []


def test_an_unavailable_model_catalog_notes_it_instead_of_failing(monkeypatch) -> None:
    monkeypatch.setattr("voice_codex.catalog.probe_codex_models", list)
    display = RecordingStatus()

    populate_codex_model_catalog(display)

    assert display.notes == [
        "Codex model catalog unavailable; using the configured model"
    ]
    assert display.catalog == {}


# --------------------------------------------------------------------------
# Configured answers skip the prompt
#
# Each chooser resolves a value the command line or the startup config already
# supplied without asking. The resolution itself is covered above through the
# ``select_*`` functions; what these add is that the chooser delegates to them
# and prints no menu, so a fully configured session never blocks on a
# terminal that may not be attached.
# --------------------------------------------------------------------------


def refuse_input(monkeypatch):
    """Make any prompt a test failure rather than a hang."""

    def unexpected_prompt(prompt=""):
        raise AssertionError(f"startup asked {prompt!r} for an answer it was given")

    monkeypatch.setattr(builtins, "input", unexpected_prompt)


def test_a_requested_microphone_skips_the_menu(monkeypatch, capsys) -> None:
    monkeypatch.setattr("voice_codex.choosers.input_devices", lambda: DEVICES)
    refuse_input(monkeypatch)

    assert choose_microphone("Webcam") == DEVICES[1]
    assert capsys.readouterr().out == ""


def test_a_requested_them_application_skips_the_menu(monkeypatch, capsys) -> None:
    def unreachable():
        raise AssertionError("the graph was read for a question nobody asked")

    monkeypatch.setattr("voice_codex.choosers.graph", unreachable)
    refuse_input(monkeypatch)

    assert choose_them_stream("ZOOM VoiceEngine") == "ZOOM VoiceEngine"
    assert capsys.readouterr().out == ""


def test_a_requested_speech_output_skips_the_menu(monkeypatch, capsys) -> None:
    monkeypatch.setattr("voice_codex.choosers.audio_outputs", lambda: list(OUTPUTS))
    refuse_input(monkeypatch)

    assert choose_tts_output("Headset") is OUTPUTS[1]
    assert capsys.readouterr().out == ""


def test_the_startup_menu_offers_every_defined_policy_in_order(
    monkeypatch, capsys
) -> None:
    answer_with(monkeypatch, ["1"])

    choose_codex_after()

    menu = capsys.readouterr().out
    for number, policy in enumerate(RESPONSE_POLICIES.values(), start=1):
        assert f"{number:2d}) {policy.label}" in menu


@pytest.mark.parametrize("name", list(RESPONSE_POLICIES))
def test_every_policy_is_reachable_by_name_and_by_menu_number(name) -> None:
    policy = RESPONSE_POLICIES[name]
    number = str(list(RESPONSE_POLICIES).index(name) + 1)

    # Both spellings a user can supply resolve to the same policy, so the menu
    # numbering cannot drift away from the mapping it is generated from.
    assert choose_codex_after(name) is policy
    assert choose_codex_after(number) is policy
