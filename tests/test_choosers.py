"""Device discovery and the interactive startup questions.

Every audio boundary is faked here: PortAudio, ``pactl``, and the terminal.
The prompt loops are driven through the real ``input`` call site so that a
rejected answer is proven to re-prompt rather than end startup.
"""

from __future__ import annotations

import builtins
import json
import subprocess
from types import SimpleNamespace

import pytest


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


def fake_pactl(monkeypatch, voice, stdout="[]", found=True, error=None):
    monkeypatch.setattr(
        voice.shutil, "which", lambda name: "/usr/bin/pactl" if found else None
    )

    def fake_run(_command, **_kwargs):
        if error is not None:
            raise error
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(voice.subprocess, "run", fake_run)


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
    voice, monkeypatch, capsys
) -> None:
    answer_with(monkeypatch, ["nonsense", "", "7"])

    assert voice.prompt_until("pick: ", int, "try again") == 7
    assert capsys.readouterr().out.count("try again") == 2


def test_a_number_outside_the_range_is_asked_again(voice, monkeypatch, capsys) -> None:
    asked = answer_with(monkeypatch, ["0", "9", "not a number", "3"])

    assert voice.prompt_number("pick: ", 1, 4, "1 to 4 please") == 3
    assert capsys.readouterr().out.count("1 to 4 please") == 3
    assert asked == ["pick: "] * 4


@pytest.mark.parametrize("answer", ["1", "4"])
def test_both_ends_of_the_range_are_accepted(voice, monkeypatch, answer) -> None:
    answer_with(monkeypatch, [answer])

    assert voice.prompt_number("pick: ", 1, 4, "retry") == int(answer)


def test_only_devices_with_input_channels_are_offered(voice, monkeypatch) -> None:
    fake_sounddevice(
        monkeypatch,
        devices=[
            {"name": "Yeti", "max_input_channels": 2},
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Webcam", "max_input_channels": 1},
        ],
    )

    assert [name["name"] for _, name in voice.input_devices()] == ["Yeti", "Webcam"]
    assert [index for index, _ in voice.input_devices()] == [0, 2]


def test_a_missing_portaudio_library_is_reported_as_a_runtime_error(
    voice, monkeypatch
) -> None:
    fake_sounddevice(monkeypatch, error=OSError("libportaudio.so not found"))

    with pytest.raises(RuntimeError, match="PortAudio system library"):
        voice.input_devices()


@pytest.mark.parametrize(
    ("requested", "expected_index"),
    [("Yeti", 0), ("Webcam", 2), ("0", 0), ("2", 2), (2, 2)],
)
def test_a_microphone_is_found_by_index_or_by_exact_name(
    voice, requested, expected_index
) -> None:
    index, _ = voice.select_microphone(DEVICES, requested)

    assert index == expected_index


def test_an_unknown_microphone_names_the_config_that_asked_for_it(voice) -> None:
    with pytest.raises(RuntimeError, match="'Studio Mic' was not found"):
        voice.select_microphone(DEVICES, "Studio Mic")


def test_choosing_a_microphone_lists_every_device_before_asking(
    voice, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(voice, "input_devices", lambda: DEVICES)
    answer_with(monkeypatch, ["2"])

    assert voice.choose_microphone() == DEVICES[1]
    menu = capsys.readouterr().out
    assert "1) Yeti (device 0, 48000 Hz)" in menu
    assert "2) Webcam (device 2, 16000 Hz)" in menu


def test_choosing_a_microphone_fails_when_the_machine_has_none(
    voice, monkeypatch
) -> None:
    monkeypatch.setattr(voice, "input_devices", list)

    with pytest.raises(RuntimeError, match="No audio input devices"):
        voice.choose_microphone()


def test_audio_outputs_pair_each_sink_with_its_monitor(voice, monkeypatch) -> None:
    fake_pactl(monkeypatch, voice, stdout=json.dumps(SINKS))

    assert voice.audio_outputs() == OUTPUTS


def test_audio_outputs_are_empty_without_pipewire(voice, monkeypatch) -> None:
    fake_pactl(monkeypatch, voice, found=False)

    assert voice.audio_outputs() == []


@pytest.mark.parametrize(
    "error",
    [
        OSError("pactl vanished"),
        subprocess.CalledProcessError(1, "pactl"),
    ],
)
def test_a_failing_pactl_yields_no_outputs_instead_of_raising(
    voice, monkeypatch, error
) -> None:
    fake_pactl(monkeypatch, voice, error=error)

    assert voice.audio_outputs() == []


def test_unparseable_pactl_output_yields_no_outputs(voice, monkeypatch) -> None:
    fake_pactl(monkeypatch, voice, stdout="not json")

    assert voice.audio_outputs() == []


def test_a_sink_without_a_monitor_is_skipped(voice, monkeypatch) -> None:
    fake_pactl(
        monkeypatch,
        voice,
        stdout=json.dumps(
            [
                {"name": "sink.a", "description": "No monitor"},
                {"monitor_source": "orphan.monitor", "description": "No name"},
                SINKS[0],
            ]
        ),
    )

    assert voice.audio_outputs() == [OUTPUTS[0]]


def test_a_sink_description_falls_back_to_its_properties_then_its_name(
    voice, monkeypatch
) -> None:
    fake_pactl(
        monkeypatch,
        voice,
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

    assert [output["description"] for output in voice.audio_outputs()] == [
        "From properties",
        "sink.b",
    ]


@pytest.mark.parametrize("requested", ["sink.b", "sink.b.monitor", "Headset"])
def test_an_output_is_matched_by_sink_monitor_or_description(voice, requested) -> None:
    assert voice.find_audio_output(OUTPUTS, requested) is OUTPUTS[1]


def test_an_unknown_output_matches_nothing(voice) -> None:
    assert voice.find_audio_output(OUTPUTS, "sink.c") is None


@pytest.mark.parametrize("requested", ["none", "NONE", "None"])
def test_them_output_none_is_accepted_in_any_case(voice, requested) -> None:
    assert voice.select_them_output(OUTPUTS, requested) is None


@pytest.mark.parametrize("requested", ["isolated", "virtual", "ISOLATED"])
def test_them_output_isolated_requests_a_virtual_sink(voice, requested) -> None:
    assert voice.select_them_output(OUTPUTS, requested) == {"isolated": True}


def test_a_direct_them_monitor_is_returned_when_tts_is_off(voice) -> None:
    assert voice.select_them_output(OUTPUTS, "Headset") is OUTPUTS[1]


def test_a_direct_them_monitor_is_refused_while_tts_is_on(voice) -> None:
    with pytest.raises(RuntimeError, match="Edge TTS cannot be used"):
        voice.select_them_output(OUTPUTS, "Headset", require_isolation=True)


def test_an_unknown_them_output_suggests_the_working_alternatives(voice) -> None:
    with pytest.raises(RuntimeError, match="--them-output isolated"):
        voice.select_them_output(OUTPUTS, "sink.missing")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("0", None), ("1", {"isolated": True}), ("2", OUTPUTS[0]), ("3", OUTPUTS[1])],
)
def test_choosing_a_them_output_maps_each_menu_entry(
    voice, monkeypatch, answer, expected
) -> None:
    monkeypatch.setattr(voice, "audio_outputs", lambda: list(OUTPUTS))
    answer_with(monkeypatch, [answer])

    assert voice.choose_them_output() == expected


def test_direct_monitors_are_hidden_from_the_menu_while_tts_is_on(
    voice, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(voice, "audio_outputs", lambda: list(OUTPUTS))
    answer_with(monkeypatch, ["2", "1"])

    assert voice.choose_them_output(require_isolation=True) == {"isolated": True}
    menu = capsys.readouterr().out
    assert "Direct output monitors are hidden" in menu
    assert "Speakers" not in menu
    assert "Please enter a number from 0 to 1." in menu


def test_a_requested_playback_output_is_matched(voice) -> None:
    assert voice.select_playback_output(OUTPUTS, "Speakers") is OUTPUTS[0]


def test_an_unknown_playback_output_is_rejected(voice) -> None:
    with pytest.raises(RuntimeError, match=r"'sink\.c' was not found"):
        voice.select_playback_output(OUTPUTS, "sink.c")


def test_choosing_a_playback_output_lists_the_sinks(voice, monkeypatch, capsys) -> None:
    monkeypatch.setattr(voice, "audio_outputs", lambda: list(OUTPUTS))
    answer_with(monkeypatch, ["2"])

    assert voice.choose_playback_output() is OUTPUTS[1]
    assert "2) Headset" in capsys.readouterr().out


def test_choosing_a_playback_output_fails_without_any_sink(voice, monkeypatch) -> None:
    monkeypatch.setattr(voice, "audio_outputs", list)

    with pytest.raises(RuntimeError, match="No PulseAudio/PipeWire audio outputs"):
        voice.choose_playback_output()


@pytest.mark.parametrize(
    ("requested", "label"),
    [
        ("them", "Them"),
        ("both", "User Voice and Them"),
        ("user", "User Voice"),
        ("quiet", "Codex will be quiet for voice"),
    ],
)
def test_a_requested_response_policy_needs_no_prompt(voice, requested, label) -> None:
    assert voice.choose_codex_after(requested)[0] == label


def test_choosing_a_response_policy_accepts_its_menu_number(
    voice, monkeypatch, capsys
) -> None:
    answer_with(monkeypatch, ["9", "2"])

    label, speakers = voice.choose_codex_after()

    assert (label, speakers) == (
        "User Voice and Them",
        frozenset({"User Voice", "Them"}),
    )
    assert "Please enter a number from 1 to 4." in capsys.readouterr().out


@pytest.mark.parametrize(("requested", "expected"), [("on", True), ("off", False)])
def test_a_requested_tts_setting_needs_no_prompt(voice, requested, expected) -> None:
    assert voice.choose_tts(requested) is expected


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
    voice, monkeypatch, answer, expected
) -> None:
    answer_with(monkeypatch, [answer])

    assert voice.choose_tts() is expected


def test_an_unrecognised_tts_answer_is_asked_again(voice, monkeypatch, capsys) -> None:
    answer_with(monkeypatch, ["maybe", "y"])

    assert voice.choose_tts() is True
    assert "Please enter 1 or 2." in capsys.readouterr().out


def test_the_model_catalog_populates_the_sidebar_selectors(voice, monkeypatch) -> None:
    option = voice.CodexModelOption("slug-a", "Model A", ("low", "high"), "high")
    monkeypatch.setattr(voice, "probe_codex_models", lambda: [option])
    recorded = {}

    display = SimpleNamespace(
        note=lambda message: recorded.setdefault("note", message),
        set_codex_catalog=lambda models, efforts, defaults: recorded.update(
            models=models, efforts=efforts, defaults=defaults
        ),
    )
    voice.populate_codex_model_catalog(display)

    assert recorded["models"] == [("Model A", "slug-a")]
    assert recorded["efforts"] == {"slug-a": ["low", "high"]}
    assert recorded["defaults"] == {"slug-a": "high"}
    assert "note" not in recorded


def test_an_unavailable_model_catalog_notes_it_instead_of_failing(
    voice, monkeypatch
) -> None:
    monkeypatch.setattr(voice, "probe_codex_models", list)
    notes: list[str] = []

    voice.populate_codex_model_catalog(
        SimpleNamespace(
            note=notes.append,
            set_codex_catalog=lambda *args: pytest.fail("catalog should stay empty"),
        )
    )

    assert notes == ["Codex model catalog unavailable; using the configured model"]
