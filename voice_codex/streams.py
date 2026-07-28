#!/usr/bin/env python3
"""Capture one application's audio instead of a whole output device.

A sink monitor carries everything the speakers play, and on this program's own
machine that includes the speech it just synthesized: tapping the output a
meeting is heard through means transcribing Codex's replies back as the far
end. The way around that used to be a virtual sink the meeting app had to be
pointed at by hand, which every session paid for and every user had to be told
about.

PipeWire makes the detour unnecessary. A playback stream is a node in the graph
with its own output ports, and a port can feed more than one destination, so a
capture node that starts unconnected and is linked to one application receives
that application and nothing else — while the application goes on playing to
the real speakers, unrouted and unaware it is being listened to. Codex's own
playback is simply never linked, so it cannot be transcribed at all. That is
the difference worth the rewrite: the old arrangement filtered the echo out
afterwards and the filter could miss.

Everything here shells out to ``pw-dump``, ``pw-link``, and ``pw-record``
rather than binding libpipewire, because the graph is described far more often
than it is changed, and a subprocess is a boundary the tests can stand in for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass

PW_DUMP = "pw-dump"
PW_LINK = "pw-link"
PW_RECORD = "pw-record"

# The graph's name for one application sending audio to an output, as opposed
# to the output itself, a capture device, or a monitor.
PLAYBACK_STREAM = "Stream/Output/Audio"

PORT_OBJECT = "PipeWire:Interface:Port"
LINK_OBJECT = "PipeWire:Interface:Link"

OUTGOING = "out"
INCOMING = "in"

# How far up the process tree a stream's owner is followed before it is taken
# to be someone else's. Codex's speech plays through a direct child, so one
# step would do; the rest is slack for a player that wraps itself in a shell.
ANCESTRY_LIMIT = 8


def _info(obj):
    """Read one graph object's ``info`` block, whatever the object turns out to be.

    ``pw-dump`` emits removals as bare objects with no ``info`` at all, so
    every read of the graph has to survive one.
    """
    if not isinstance(obj, dict):
        return {}
    info = obj.get("info")
    return info if isinstance(info, dict) else {}


def _properties(obj):
    """Read one graph object's property dictionary."""
    props = _info(obj).get("props")
    return props if isinstance(props, dict) else {}


def graph(run=subprocess.run):
    """Describe the whole PipeWire graph: nodes, ports, and links in one dump.

    One dump answers every question this module asks, and asking them
    separately would be worse than slow — the graph moves between calls, so a
    port list and a link list fetched apart can disagree about which nodes
    exist. A failed or unparsable dump is an empty graph rather than an error:
    the caller polls, and the next pass is a second away.
    """
    try:
        result = run([PW_DUMP], check=True, capture_output=True, text=True)
        objects = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []
    return objects if isinstance(objects, list) else []


def parent_process(pid, proc="/proc"):
    """Name a process's parent, or nothing when it cannot be read."""
    try:
        with open(f"{proc}/{pid}/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def spawned_here(pid, own_pid=None, parent=parent_process, limit=ANCESTRY_LIMIT):
    """Report whether a stream belongs to this program rather than to a user app.

    Codex speaks through a short-lived player process, and the graph names that
    stream after the player. Without this the picker would offer this program's
    own voice as something to transcribe, and would offer a different one every
    sentence.

    Ancestry rather than the process name, because the player is an ordinary
    tool a user may also be running for their own reasons: what distinguishes
    Codex's copy of it is that this process started it.
    """
    if not isinstance(pid, int):
        return False
    own = os.getpid() if own_pid is None else own_pid
    current = pid
    for _ in range(limit):
        if current is None or current <= 0:
            return False
        if current == own:
            return True
        current = parent(current)
    return False


@dataclass(frozen=True)
class ApplicationStream:
    """One application's playback, as the graph currently has it."""

    node_id: int
    application: str
    title: str
    binary: str
    playing: bool


def application_streams(objects, mine=spawned_here):
    """List the playback streams a session could transcribe.

    A stream with no name is skipped rather than shown as a blank line: it
    cannot be described to whoever is choosing, and a config file could not
    record it in a way that would find it again.
    """
    streams = []
    for obj in objects:
        props = _properties(obj)
        if props.get("media.class") != PLAYBACK_STREAM:
            continue
        if mine(props.get("application.process.id")):
            continue
        application = props.get("application.name") or props.get(
            "application.process.binary"
        )
        node_id = obj.get("id")
        if not application or not isinstance(node_id, int):
            continue
        streams.append(
            ApplicationStream(
                node_id=node_id,
                application=str(application),
                title=str(props.get("media.name") or ""),
                binary=str(props.get("application.process.binary") or ""),
                playing=_info(obj).get("state") == "running",
            )
        )
    return streams


def applications(streams):
    """Collapse streams into the applications a session can be pointed at.

    A browser opens a node per tab that plays audio and a meeting app opens one
    per call, so the node list is longer than the list of things anyone means
    to choose between. The tap follows a name, which is what makes collapsing
    them right rather than merely tidier: picking any of an application's
    streams picks all of them, so offering them separately would promise a
    distinction that does not exist.
    """
    collapsed = {}
    for stream in streams:
        found = collapsed.get(stream.application)
        collapsed[stream.application] = ApplicationStream(
            node_id=stream.node_id,
            application=stream.application,
            title=stream.title,
            binary=stream.binary,
            playing=stream.playing or (found is not None and found.playing),
        )
    return sorted(collapsed.values(), key=lambda s: (not s.playing, s.application))


def node_ports(objects, node_ids, direction):
    """Collect the ports of some nodes, in one direction."""
    wanted = set(node_ids)
    ports = []
    for obj in objects:
        if obj.get("type") != PORT_OBJECT:
            continue
        props = _properties(obj)
        if props.get("node.id") in wanted and props.get("port.direction") == direction:
            ports.append(obj.get("id"))
    return ports


def nodes_named(objects, node_name):
    """Find the graph nodes carrying an exact ``node.name``."""
    return [
        obj.get("id")
        for obj in objects
        if _properties(obj).get("node.name") == node_name
    ]


def linked_sources(objects, node_id):
    """The output ports already feeding a node.

    Relinking is a poll, so this is what keeps it from stacking a second copy
    of every link on top of the ones that are working.
    """
    sources = set()
    for obj in objects:
        if obj.get("type") != LINK_OBJECT:
            continue
        props = _properties(obj)
        if props.get("link.input.node") == node_id:
            sources.add(props.get("link.output.port"))
    return sources


def require_pipewire():
    """Fail early, and by name, when the tools the tap is built from are missing."""
    for tool in (PW_RECORD, PW_LINK, PW_DUMP):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"{tool} is required to transcribe an application's audio. "
                "Voice Codex captures PipeWire playback streams directly."
            )


class StreamTap:
    """Feed one application's playback into a capture node of this program's own.

    The tap is rebuilt continuously rather than wired once. A stream node lives
    only as long as the application is playing through it: a meeting app that
    restarts, or a browser that opens a second tab, arrives as a new node with a
    new ID, and a tap linked once would fall silent exactly when the thing it
    was watching came back. So the link pass runs on a timer and is written to
    be idempotent — it adds what is missing and leaves the rest alone — which
    makes the cost of polling one graph dump and nothing else.

    The capture node itself belongs to the ``pw-record`` process the caller
    starts; this class only finds it by name and wires it up. That split is why
    a tap whose recorder has not appeared yet is not an error: it links nothing
    and tries again.
    """

    NODE_PREFIX = "voice_codex_tap_"
    POLL_SECONDS = 1.0

    # pw-record's own default is 100ms, which is added to everything this
    # channel decides — including whether the far end is talking right now.
    LATENCY = "20ms"

    def __init__(self, application, poll=POLL_SECONDS, run=subprocess.run, dump=graph):
        self.application = application
        # The PID is in the name so a session that died without cleaning up
        # cannot be mistaken for this one; the node disappears with the
        # recorder, so nothing has to be swept.
        self.node_name = f"{self.NODE_PREFIX}{os.getpid()}"
        self.poll = poll
        self.run = run
        self.dump = dump
        self.stopping = threading.Event()
        self.watcher = None

    def command(self, samplerate):
        """Build the ``pw-record`` command that streams the tap as raw PCM.

        ``--target 0`` is the whole point. It tells PipeWire not to connect the
        capture node to anything, so it arrives empty and carries only what
        this class links into it. Without it the node autoconnects to the
        default source, and the Them channel quietly records the microphone.
        """
        return [
            PW_RECORD,
            "--target",
            "0",
            "--rate",
            str(samplerate),
            "--channels",
            "1",
            "--format",
            "s16",
            "--latency",
            self.LATENCY,
            "-P",
            f"{{ node.name = {self.node_name} }}",
            "-",
        ]

    def link(self):
        """Link every matching stream that is not linked yet; report how many.

        The count is what tells a first attach from a quiet poll, which is the
        only thing the caller wants to know often enough to be worth returning.
        """
        objects = self.dump()
        taps = nodes_named(objects, self.node_name)
        inputs = node_ports(objects, taps, INCOMING)
        if not taps or not inputs:
            return 0
        sources = [
            stream.node_id
            for stream in application_streams(objects)
            if stream.application == self.application
        ]
        already = linked_sources(objects, taps[0])
        linked = 0
        for port in node_ports(objects, sources, OUTGOING):
            # A stereo application meets a mono tap, so both of its ports land
            # on the one input and PipeWire sums them. That is the downmix this
            # channel wants: speech on one side only must not be halved away.
            if port not in already and self._connect(port, inputs[0]):
                linked += 1
        return linked

    def _connect(self, source_port, tap_port):
        """Join two ports, treating a refusal as something the next pass retries."""
        try:
            result = self.run(
                [PW_LINK, str(source_port), str(tap_port)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        return result.returncode == 0

    def start(self):
        """Begin following the application, linking whatever it is playing now."""
        if self.watcher is not None:
            return
        self.stopping.clear()
        self.watcher = threading.Thread(
            target=self._follow,
            name="StreamTapLinker",
            daemon=True,
        )
        self.watcher.start()

    def _follow(self):
        while True:
            self.link()
            if self.stopping.wait(self.poll):
                return

    def stop(self):
        """Stop following. The links go with the recorder that owned the node."""
        if self.watcher is None:
            return
        self.stopping.set()
        self.watcher.join(timeout=5)
        self.watcher = None
