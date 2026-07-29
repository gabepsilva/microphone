"""Reading the PipeWire graph, and wiring one application into a capture node.

``pw-dump`` and ``pw-link`` are faked at the subprocess boundary. What is not
faked is the reasoning on top of them: which streams a session may transcribe,
which of them are this program's own, and which links a relink pass still owes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest

from voice_codex.streams import (
    ApplicationRefresher,
    ApplicationStream,
    StreamTap,
    application_streams,
    applications,
    graph,
    linked_sources,
    node_ports,
    nodes_named,
    offered_applications,
    parent_process,
    require_pipewire,
    spawned_here,
    stream_label,
)

WAIT_SECONDS = 10


def node(
    node_id,
    media_class="Stream/Output/Audio",
    state="running",
    application=None,
    **props,
):
    """One graph node, shaped the way ``pw-dump`` reports it."""
    if application is not None:
        props["application.name"] = application
    return {
        "id": node_id,
        "type": "PipeWire:Interface:Node",
        "info": {"state": state, "props": {"media.class": media_class, **props}},
    }


def port(port_id, node_id, direction, name="output_FL", channel=None):
    props = {
        "node.id": node_id,
        "port.direction": direction,
        "port.name": name,
    }
    if channel is not None:
        props["audio.channel"] = channel
    return {"id": port_id, "type": "PipeWire:Interface:Port", "info": {"props": props}}


def link(output_node, output_port, input_node, input_port):
    return {
        "id": 900 + output_port,
        "type": "PipeWire:Interface:Link",
        "info": {
            "props": {
                "link.output.node": output_node,
                "link.output.port": output_port,
                "link.input.node": input_node,
                "link.input.port": input_port,
            }
        },
    }


def nobody(_pid):
    """Attribute every stream to some other program."""
    return False


# --------------------------------------------------------------------------
# Reading the graph
# --------------------------------------------------------------------------


def test_the_graph_is_read_from_one_dump() -> None:
    payload = [node(1, application="Chromium")]

    def run(command, **_kwargs):
        assert command == ["pw-dump"]
        return SimpleNamespace(stdout=json.dumps(payload))

    assert graph(run=run) == payload


@pytest.mark.parametrize(
    "failure",
    [
        OSError("pw-dump is gone"),
        subprocess.CalledProcessError(1, ["pw-dump"]),
    ],
)
def test_a_failed_dump_is_an_empty_graph_rather_than_an_error(failure) -> None:
    """The caller polls, so a graph that cannot be read now is read next pass."""

    def run(_command, **_kwargs):
        raise failure

    assert graph(run=run) == []


def test_an_unparsable_dump_is_an_empty_graph() -> None:
    assert graph(run=lambda *_a, **_k: SimpleNamespace(stdout="not json")) == []


def test_a_dump_that_is_not_a_list_is_an_empty_graph() -> None:
    assert graph(run=lambda *_a, **_k: SimpleNamespace(stdout='{"id": 1}')) == []


def test_an_object_without_an_info_block_is_survived() -> None:
    """pw-dump reports a removal as a bare object, and it lands in every read."""
    objects = [{"id": 7}, "junk", node(1, application="Chromium")]

    assert [s.application for s in application_streams(objects, mine=nobody)] == [
        "Chromium"
    ]


# --------------------------------------------------------------------------
# Telling this program's own audio from everyone else's
# --------------------------------------------------------------------------


def test_a_process_reports_the_parent_the_kernel_gives_it(tmp_path) -> None:
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text(
        "Name:\tffplay\nPPid:\t7\n", encoding="utf-8"
    )

    assert parent_process(42, proc=str(tmp_path)) == 7


def test_a_process_that_cannot_be_read_has_no_parent(tmp_path) -> None:
    assert parent_process(42, proc=str(tmp_path)) is None


def test_a_status_without_a_parent_line_has_no_parent(tmp_path) -> None:
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text("Name:\tffplay\n", encoding="utf-8")

    assert parent_process(42, proc=str(tmp_path)) is None


def test_this_process_owns_its_own_stream() -> None:
    assert spawned_here(os.getpid()) is True


def test_a_stream_started_by_this_process_is_its_own() -> None:
    """Codex speaks through a child player, and the graph names the child."""
    parents = {55: 9, 9: os.getpid()}

    assert spawned_here(55, parent=parents.get) is True


def test_an_unrelated_process_is_not_this_program() -> None:
    parents = {55: 9, 9: 1}

    assert spawned_here(55, own_pid=1234, parent=parents.get) is False


def test_a_stream_with_no_process_id_belongs_to_someone_else() -> None:
    assert spawned_here(None) is False


def test_an_ancestry_walk_gives_up_rather_than_looping() -> None:
    """A cycle in the reported parents must not hang the picker."""
    assert spawned_here(1, own_pid=999, parent=lambda _pid: 1, limit=3) is False


# --------------------------------------------------------------------------
# Which streams a session may be pointed at
# --------------------------------------------------------------------------


def test_only_playback_streams_are_offered() -> None:
    objects = [
        node(1, application="Chromium"),
        node(2, media_class="Audio/Sink", application="Headphones"),
        node(3, media_class="Stream/Input/Audio", application="Recorder"),
    ]

    assert [s.node_id for s in application_streams(objects, mine=nobody)] == [1]


def test_this_program_never_offers_its_own_voice() -> None:
    objects = [
        node(1, application_name="ffplay", **{"application.process.id": 55}),
        node(2, application="Chromium", **{"application.process.id": 66}),
    ]

    streams = application_streams(objects, mine=lambda pid: pid == 55)

    assert [s.application for s in streams] == ["Chromium"]


def test_a_stream_falls_back_to_its_binary_when_it_has_no_name() -> None:
    objects = [node(1, **{"application.process.binary": "zoom"})]

    assert [s.application for s in application_streams(objects, mine=nobody)] == [
        "zoom"
    ]


def test_a_nameless_stream_is_skipped_rather_than_offered_blank() -> None:
    assert application_streams([node(1)], mine=nobody) == []


def test_a_stream_without_an_id_is_skipped() -> None:
    objects = [
        {
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Stream/Output/Audio",
                    "application.name": "Chromium",
                }
            },
        }
    ]

    assert application_streams(objects, mine=nobody) == []


def test_a_stream_carries_its_title_and_whether_it_is_playing() -> None:
    objects = [
        node(
            1,
            state="idle",
            application="Chromium",
            **{
                "media.name": "Playback",
                "application.process.binary": "chromium",
            },
        ),
    ]

    (stream,) = application_streams(objects, mine=nobody)

    assert (stream.title, stream.binary, stream.playing) == (
        "Playback",
        "chromium",
        False,
    )


# --------------------------------------------------------------------------
# Collapsing streams into the applications anyone means to choose between
# --------------------------------------------------------------------------


def make_stream(application, node_id=1, playing=False, title="", binary=""):
    return ApplicationStream(
        node_id=node_id,
        application=application,
        title=title,
        binary=binary,
        playing=playing,
    )


def test_an_applications_streams_are_offered_as_one_entry() -> None:
    """A browser opens a node per tab, and the tap follows the name, not the node."""
    streams = [
        make_stream("Chromium", node_id=1),
        make_stream("Chromium", node_id=2),
        make_stream("ZOOM VoiceEngine", node_id=3),
    ]

    assert [s.application for s in applications(streams)] == [
        "Chromium",
        "ZOOM VoiceEngine",
    ]


def test_an_application_counts_as_playing_when_any_of_its_streams_is() -> None:
    streams = [
        make_stream("Chromium", node_id=1, playing=False),
        make_stream("Chromium", node_id=2, playing=True),
    ]

    assert applications(streams)[0].playing is True


def test_the_applications_making_sound_are_offered_first() -> None:
    streams = [
        make_stream("Aardvark", playing=False),
        make_stream("ZOOM VoiceEngine", node_id=2, playing=True),
    ]

    assert [s.application for s in applications(streams)] == [
        "ZOOM VoiceEngine",
        "Aardvark",
    ]


# --------------------------------------------------------------------------
# Ports and links
# --------------------------------------------------------------------------


def test_ports_are_collected_by_node_and_direction() -> None:
    objects = [
        port(10, 1, "out"),
        port(11, 1, "in"),
        port(12, 2, "out"),
        node(1, application="Chromium"),
    ]

    assert node_ports(objects, [1], "out") == [10]
    assert node_ports(objects, [1, 2], "out") == [10, 12]


def test_ports_are_ordered_by_channel_not_by_the_order_the_graph_holds_them() -> None:
    """The graph routinely lists a stream's right channel first."""
    objects = [
        port(70, 1, "out", name="output_FR", channel="FR"),
        port(71, 1, "out", name="output_FL", channel="FL"),
    ]

    assert node_ports(objects, [1], "out") == [71, 70]


def test_a_port_whose_channel_is_unnamed_sorts_after_the_named_ones() -> None:
    objects = [
        port(80, 1, "out", name="output_2", channel="AUX0"),
        port(81, 1, "out", name="output_FL", channel="FL"),
    ]

    assert node_ports(objects, [1], "out") == [81, 80]


def test_a_node_is_found_by_its_exact_name() -> None:
    objects = [
        node(1, **{"node.name": "voice_codex_tap_9"}),
        node(2, **{"node.name": "voice_codex_tap_90"}),
    ]

    assert nodes_named(objects, "voice_codex_tap_9") == [1]


def test_the_links_already_feeding_a_node_are_reported_as_pairs() -> None:
    """One source port legitimately feeds both sides when the source is mono."""
    objects = [link(1, 10, 5, 50), link(1, 10, 5, 51), link(2, 20, 6, 60)]

    assert linked_sources(objects, 5) == {(10, 50), (10, 51)}


# --------------------------------------------------------------------------
# The tap
# --------------------------------------------------------------------------


def test_the_missing_tool_is_named_rather_than_the_toolkit(monkeypatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "pw-link" else "/x"
    )

    with pytest.raises(RuntimeError, match="pw-link is required"):
        require_pipewire()


def test_every_pipewire_tool_present_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    assert require_pipewire() is None


def test_the_recorder_is_told_not_to_connect_itself_to_anything() -> None:
    """Without this the capture node autoconnects and records the microphone."""
    command = StreamTap("Chromium").command(16000)

    assert command[0] == "pw-record"
    assert command[command.index("--target") + 1] == "0"
    assert command[command.index("--rate") + 1] == "16000"
    assert command[command.index("--channels") + 1] == "2"
    assert command[command.index("--format") + 1] == "s16"
    assert command[-1] == "-"


def test_the_capture_node_is_named_for_this_process() -> None:
    tap = StreamTap("Chromium")

    assert tap.node_name == f"voice_codex_tap_{os.getpid()}"
    assert f"node.name = {tap.node_name}" in " ".join(tap.command(16000))


def tap_graph(tap, application="Chromium", links=(), tap_present=True, mono=False):
    """A graph holding one application and, usually, the two-channel tap."""
    objects = [
        node(1, **{"application.name": application}),
        port(10, 1, "out", name="output_FL", channel="FL"),
        node(2, **{"application.name": "Other"}),
        port(20, 2, "out", channel="FL"),
        *links,
    ]
    if not mono:
        objects.insert(2, port(11, 1, "out", name="output_FR", channel="FR"))
    if tap_present:
        objects += [
            node(5, media_class="Stream/Input/Audio", **{"node.name": tap.node_name}),
            port(50, 5, "in", name="input_FL", channel="FL"),
            port(51, 5, "in", name="input_FR", channel="FR"),
        ]
    return objects


def recording_tap(application="Chromium", objects=None, returncode=0):
    """A tap whose pw-link calls are recorded instead of run."""
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    tap = StreamTap(application, run=run, dump=lambda: objects or [])
    return tap, calls


def test_a_stream_is_linked_channel_for_channel() -> None:
    """Crossed channels would swap a stereo far end left for right."""
    tap = StreamTap("Chromium")
    tap, calls = recording_tap(objects=tap_graph(tap))

    assert tap.link() == 2
    assert calls == [["pw-link", "10", "50"], ["pw-link", "11", "51"]]


def test_a_mono_application_feeds_both_sides_of_the_tap() -> None:
    """The caller averages the pair, so one side alone would arrive halved."""
    tap = StreamTap("Chromium")
    tap, calls = recording_tap(objects=tap_graph(tap, mono=True))

    assert tap.link() == 2
    assert calls == [["pw-link", "10", "50"], ["pw-link", "10", "51"]]


def test_only_the_chosen_application_is_linked() -> None:
    tap = StreamTap("Chromium")
    tap, calls = recording_tap(objects=tap_graph(tap))

    tap.link()

    assert [command[1] for command in calls] == ["10", "11"]


def test_a_link_already_made_is_left_alone() -> None:
    """Relinking is a poll; a second copy of a working link is the failure."""
    tap = StreamTap("Chromium")
    tap, calls = recording_tap(objects=tap_graph(tap, links=[link(1, 10, 5, 50)]))

    assert tap.link() == 1
    assert calls == [["pw-link", "11", "51"]]


def test_following_nothing_links_nothing() -> None:
    """A session that has not been pointed anywhere yet is not an error."""
    tap = StreamTap()
    tap, calls = recording_tap(None, objects=tap_graph(tap))

    assert tap.link() == 0
    assert calls == []


def test_switching_applications_cuts_the_links_the_last_one_left() -> None:
    """Otherwise the old application keeps feeding the tap alongside the new."""
    tap = StreamTap("Chromium")
    objects = tap_graph(tap, links=[link(1, 10, 5, 50), link(1, 11, 5, 51)])
    tap, calls = recording_tap("Other", objects=objects)

    assert tap.link() == 4
    assert calls[-2:] == [
        ["pw-link", "--disconnect", "10", "50"],
        ["pw-link", "--disconnect", "11", "51"],
    ]


def test_choosing_nothing_cuts_every_link_the_tap_had() -> None:
    tap = StreamTap("Chromium")
    objects = tap_graph(tap, links=[link(1, 10, 5, 50), link(1, 11, 5, 51)])
    tap, calls = recording_tap(None, objects=objects)

    assert tap.link() == 2
    assert calls == [
        ["pw-link", "--disconnect", "10", "50"],
        ["pw-link", "--disconnect", "11", "51"],
    ]


def test_following_moves_the_name_without_touching_the_recorder() -> None:
    tap = StreamTap("Chromium", dump=list)

    tap.follow("Brave")

    assert (tap.application, tap.node_name) == ("Brave", tap.node_name)


def test_a_half_described_link_is_not_something_to_cut() -> None:
    """pw-dump reports a link mid-teardown with an end already gone."""
    objects = [link(1, 10, 5, 50)]
    objects[0]["info"]["props"]["link.output.port"] = None

    assert linked_sources(objects, 5) == set()


def test_a_stream_with_no_ports_yet_is_skipped_rather_than_half_linked() -> None:
    """A node appears in the graph a moment before its ports do."""
    tap = StreamTap("Chromium")
    objects = [o for o in tap_graph(tap) if o.get("id") not in (10, 11)]
    tap, calls = recording_tap(objects=objects)

    assert tap.link() == 0
    assert calls == []


def test_a_tap_whose_recorder_has_not_appeared_links_nothing() -> None:
    tap = StreamTap("Chromium")
    tap, calls = recording_tap(objects=tap_graph(tap, tap_present=False))

    assert tap.link() == 0
    assert calls == []


def test_an_application_that_is_not_playing_yet_links_nothing() -> None:
    tap = StreamTap("ZOOM VoiceEngine")
    tap, calls = recording_tap("ZOOM VoiceEngine", objects=tap_graph(tap))

    assert tap.link() == 0
    assert calls == []


def test_a_refused_link_is_not_counted() -> None:
    tap = StreamTap("Chromium")
    tap, _ = recording_tap(objects=tap_graph(tap), returncode=1)

    assert tap.link() == 0


def test_a_link_tool_that_cannot_be_run_is_not_counted() -> None:
    tap = StreamTap("Chromium")
    objects = tap_graph(tap)

    def run(_command, **_kwargs):
        raise OSError("pw-link is gone")

    tap = StreamTap("Chromium", run=run, dump=lambda: objects)

    assert tap.link() == 0


def test_following_an_application_relinks_until_it_is_stopped() -> None:
    tap = StreamTap("Chromium")
    objects = tap_graph(tap)
    passes = threading.Event()
    dumps: list[int] = []

    def dump():
        dumps.append(1)
        if len(dumps) >= 2:
            passes.set()
        return objects

    tap = StreamTap(
        "Chromium",
        poll=0.001,
        run=lambda *_a, **_k: SimpleNamespace(returncode=0),
        dump=dump,
    )
    tap.start()
    try:
        assert passes.wait(WAIT_SECONDS)
    finally:
        tap.stop()

    assert tap.watcher is None


def test_starting_a_tap_twice_follows_it_once() -> None:
    tap = StreamTap("Chromium", poll=0.001, dump=list)
    tap.start()
    watcher = tap.watcher
    try:
        tap.start()

        assert tap.watcher is watcher
    finally:
        tap.stop()


def test_stopping_a_tap_that_never_started_does_nothing() -> None:
    tap = StreamTap("Chromium", dump=list)

    tap.stop()

    assert tap.watcher is None


# --------------------------------------------------------------------------
# Keeping the picker's list current
# --------------------------------------------------------------------------


class FakeDisplay:
    """Record what the sidebar would be told to offer."""

    def __init__(self):
        self.offered = []

    def set_them_streams(self, applications):
        self.offered.append(list(applications))


PLAYING = [node(1, application="Brave", state="running", **{"media.name": "Playback"})]


def test_the_offered_list_is_labelled_and_paired_with_what_the_tap_follows() -> None:
    assert offered_applications(PLAYING) == [("Brave — Playback (playing)", "Brave")]


def test_a_long_title_is_cut_rather_than_widening_the_sidebar() -> None:
    """A stream titles itself with a tab heading or an absolute file path."""
    stream = make_stream("mpv", title="/home/someone/very/long/path/to/a/recording.wav")

    label = stream_label(stream)

    assert label == "mpv — /home/someone/very/long/path/to… (idle)"
    assert len(label) < len(stream.title)


def test_a_refresh_tells_the_display_what_is_playing() -> None:
    display = FakeDisplay()
    refresher = ApplicationRefresher(display, dump=lambda: PLAYING)

    assert refresher.refresh() is True
    assert display.offered == [[("Brave — Playback (playing)", "Brave")]]


def test_an_unchanged_list_is_not_reported_again() -> None:
    """Every report repaints the sidebar, and the list is usually the same."""
    display = FakeDisplay()
    refresher = ApplicationRefresher(display, dump=lambda: PLAYING)
    refresher.refresh()

    assert refresher.refresh() is False
    assert len(display.offered) == 1


def test_an_application_that_stopped_playing_is_reported_as_a_change() -> None:
    display = FakeDisplay()
    graphs = [PLAYING, []]
    refresher = ApplicationRefresher(display, dump=lambda: graphs.pop(0))
    refresher.refresh()

    assert refresher.refresh() is True
    assert display.offered[-1] == []


def test_the_refresher_keeps_polling_until_it_is_stopped() -> None:
    display = FakeDisplay()
    passes = threading.Event()
    dumps = []

    def dump():
        dumps.append(1)
        if len(dumps) >= 2:
            passes.set()
        return []

    refresher = ApplicationRefresher(display, poll=0.001, dump=dump)
    refresher.start()
    try:
        assert passes.wait(WAIT_SECONDS)
    finally:
        refresher.stop()

    assert refresher.worker is None


def test_starting_the_refresher_twice_polls_once() -> None:
    refresher = ApplicationRefresher(FakeDisplay(), poll=0.001, dump=list)
    refresher.start()
    worker = refresher.worker
    try:
        refresher.start()

        assert refresher.worker is worker
    finally:
        refresher.stop()


def test_stopping_a_refresher_that_never_started_does_nothing() -> None:
    refresher = ApplicationRefresher(FakeDisplay(), dump=list)

    refresher.stop()

    assert refresher.worker is None
