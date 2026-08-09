"""The MPRIS media port: what it announces and what it never pretends to do.

dbus-fast's session bus is faked at the point where these tests hand the
player a bus to live on; everything the module builds on top of it — the two
interfaces it exports, the properties it promises, the signals it emits, the
Stop method, and the fallbacks when no bus can be claimed — is the code under
test. Whether a real Linux session bus answers the same way is what the
environment guarantees; what this module does with the bus is proved here.
"""

from __future__ import annotations

import io
import threading
import time

import pytest
from dbus_fast.aio.message_bus import MessageBus as AioMessageBus
from dbus_fast.constants import NameFlag, RequestNameReply
from dbus_fast.errors import DBusError
from dbus_fast.service import ServiceInterface

from tagalong import media_controls
from tagalong.media_controls import (
    MEDIA_STATUS_PLAYING,
    MEDIA_STATUS_STOPPED,
    PLAYER_BUS_NAME,
    MediaControlsPort,
    MediaControlsUnavailable,
    MprisMediaControls,
    NullMediaControls,
)

WAIT_SECONDS = 10

ROOT_NAME = "org.mpris.MediaPlayer2"
PLAYER_NAME = "org.mpris.MediaPlayer2.Player"


def wait_until(predicate):
    """Wait for an announcement through the loop rather than for a delay."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class FakeBus(AioMessageBus):
    """Stand in for a dbus-fast session bus, recording what it is asked.

    The real bus serializes messages and calls ``_interface_signal_notify``
    on itself when a service emits; this one keeps the same shape and records
    it, together with the exported interfaces and the name claim.
    """

    def __init__(self, reply=RequestNameReply.PRIMARY_OWNER, connect_error=None):
        self.reply = reply
        self.connect_error = connect_error
        self.claimed: list[str] = []
        self.exported: list[tuple[str, str, object]] = []
        self.signals: list[tuple[str, str, list]] = []
        self.disconnected = False

    async def connect(self):
        if self.connect_error is not None:
            raise self.connect_error

    async def request_name(self, name: str, flags: NameFlag = NameFlag.NONE):  # noqa: ARG002 - the claim is what a real bus would hear
        self.claimed.append(name)
        return self.reply

    def export(self, path, interface):
        ServiceInterface._add_bus(interface, self, _noop_maker)
        self.exported.append((path, interface.name, interface))

    def _interface_signal_notify(  # noqa: PLR0913 - mirrors the bus's fixed six
        self, interface, interface_name, member, signature, body, unix_fds=None
    ):
        del interface, signature, unix_fds
        self.signals.append((interface_name, member, body))

    def disconnect(self):
        self.disconnected = True


def _noop_maker(interface, method):
    """A handler maker for interfaces exported onto the fake bus: nothing."""
    del interface, method

    def handler(message, send_reply):
        del message, send_reply

    return handler


class StopHook:
    def __init__(self):
        self.stops = 0

    def __call__(self):
        self.stops += 1


def launch(bus, stops=None):
    """A real port living on a fake bus, and the bus it wrote to."""
    port = MprisMediaControls(request_stop=stops, bus_factory=lambda: bus)
    return port, bus


def player_of(bus):
    return next(interface for _, name, interface in bus.exported if name == PLAYER_NAME)


def changes(signal):
    """The changed-properties dict of one PropertiesChanged signal."""
    iface, member, body = signal
    if iface != "org.freedesktop.DBus.Properties" or member != "PropertiesChanged":
        return None
    return {prop: variant.value for prop, variant in body[1].items()}


# --------------------------------------------------------------------------
# The null port and the gates that choose it
# --------------------------------------------------------------------------


def test_the_null_port_drops_every_announcement() -> None:
    port = NullMediaControls()
    port.publish(MEDIA_STATUS_PLAYING, "Hello")
    port.publish(MEDIA_STATUS_STOPPED)
    port.close()
    assert isinstance(port, MediaControlsPort)


def test_disabled_build_returns_null_without_a_word(capsys) -> None:
    port = media_controls.build_media_controls(False)
    assert isinstance(port, NullMediaControls)
    assert capsys.readouterr().err == ""


def test_a_platform_without_dbus_fast_falls_back_with_one_line(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(media_controls, "_dbus_fast_available", lambda: False)
    port = media_controls.build_media_controls(True)
    assert isinstance(port, NullMediaControls)
    err = capsys.readouterr().err
    assert err.count("Media controls unavailable") == 1


def test_a_connect_failure_is_reported_as_unavailable() -> None:
    bus = FakeBus(connect_error=RuntimeError("no bus here"))
    with pytest.raises(MediaControlsUnavailable, match="no bus here"):
        MprisMediaControls(bus_factory=lambda: bus)


# --------------------------------------------------------------------------
# What goes onto the bus
# --------------------------------------------------------------------------


def test_the_player_is_exported_under_the_mpris_name() -> None:
    port, bus = launch(FakeBus())

    assert bus.claimed == [PLAYER_BUS_NAME]
    assert {name for _, name, _ in bus.exported} == {ROOT_NAME, PLAYER_NAME}
    assert all(path == "/org/mpris/MediaPlayer2" for path, _, _ in bus.exported)
    port.close()


def test_playing_publishes_the_status_and_the_title() -> None:
    port, bus = launch(FakeBus())
    port.publish(MEDIA_STATUS_PLAYING, "Hello world")

    assert wait_until(lambda: bool(bus.signals)), "no signal reached the bus"
    changed = changes(bus.signals[0])
    assert changed["PlaybackStatus"] == "Playing"
    metadata = changed["Metadata"]
    assert metadata["xesam:title"].value == "Hello world"
    assert str(metadata["mpris:trackid"].value).startswith(
        "/org/mpris/MediaPlayer2/Track/"
    )
    port.close()


def test_a_repeated_title_is_not_a_new_track() -> None:
    port, bus = launch(FakeBus())
    port.publish(MEDIA_STATUS_PLAYING, "Same line")
    port.publish(MEDIA_STATUS_PLAYING, "Same line")

    assert wait_until(lambda: bool(bus.signals))
    time.sleep(0.05)
    assert len(bus.signals) == 1
    port.close()


def test_stopping_publishes_only_the_status_change() -> None:
    port, bus = launch(FakeBus())
    port.publish(MEDIA_STATUS_PLAYING, "Hello")
    assert wait_until(lambda: bool(bus.signals))
    port.publish(MEDIA_STATUS_STOPPED)

    assert wait_until(lambda: len(bus.signals) >= 2)
    changed = changes(bus.signals[1])
    assert "PlaybackStatus" in changed
    assert changed["PlaybackStatus"] == "Stopped"
    assert "Metadata" not in changed
    port.close()


def test_the_title_survives_into_the_property_read() -> None:
    port, bus = launch(FakeBus())
    port.publish(MEDIA_STATUS_PLAYING, "Still saying this")
    assert wait_until(lambda: bool(bus.signals))

    player = player_of(bus)
    metadata = player.Metadata
    assert metadata["xesam:title"].value == "Still saying this"
    port.close()


# --------------------------------------------------------------------------
# The desktop's stop
# --------------------------------------------------------------------------


def test_the_players_stop_method_retires_the_speech() -> None:
    stops = StopHook()
    port, bus = launch(FakeBus(), stops=stops)

    player_of(bus).Stop()

    assert stops.stops == 1
    port.close()


def test_methods_a_stop_only_player_has_not_are_rejected() -> None:
    port, bus = launch(FakeBus())
    player = player_of(bus)

    with pytest.raises(DBusError):
        player.Play()
    with pytest.raises(DBusError):
        player.Seek(1000000)

    port.close()


def test_cannot_pretend_the_properties_it_does_not_have() -> None:
    port, bus = launch(FakeBus())
    player = player_of(bus)

    assert player.CanPlay is False
    assert player.CanPause is False
    assert player.CanSeek is False
    assert player.CanControl is True
    assert player.PlaybackStatus == "Stopped"
    port.close()


# --------------------------------------------------------------------------
# Giving up the name honestly
# --------------------------------------------------------------------------


def test_a_claimed_name_is_reported_and_falls_back() -> None:
    claimed = FakeBus(reply=RequestNameReply.IN_QUEUE)
    with pytest.raises(MediaControlsUnavailable, match="already claimed"):
        MprisMediaControls(bus_factory=lambda: claimed)
    assert claimed.disconnected is True

    out = io.StringIO()
    port = media_controls.build_media_controls(
        True, stream=out, bus_factory=lambda: FakeBus(reply=RequestNameReply.IN_QUEUE)
    )
    assert isinstance(port, NullMediaControls)
    assert "already claimed" in out.getvalue()


def _slow_bus(release):
    """A bus whose connect is parked until the test says otherwise."""

    class SlowFakeBus(FakeBus):
        async def connect(self):
            release.wait(timeout=WAIT_SECONDS)
            await super().connect()

    return SlowFakeBus


def test_a_session_bus_that_answers_late_leaves_no_orphan_claim(
    monkeypatch,
) -> None:
    """Giving up on a slow bus must not leave a name claim behind.

    The constructor raises and the session falls back to silence; the loop
    thread wakes up later anyway, and it must release whatever it reached
    rather than exporting a player nobody publishes to.
    """
    monkeypatch.setattr(MprisMediaControls, "READY_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    bus = _slow_bus(release)()
    with pytest.raises(MediaControlsUnavailable, match="did not answer"):
        MprisMediaControls(bus_factory=lambda: bus)

    release.set()
    assert wait_until(lambda: bus.disconnected)
    assert wait_until(lambda: bus.claimed == [])


def test_a_connect_failure_falls_back_with_one_line() -> None:
    out = io.StringIO()
    port = media_controls.build_media_controls(
        True,
        stream=out,
        bus_factory=lambda: FakeBus(connect_error=RuntimeError("no bus here")),
    )
    assert isinstance(port, NullMediaControls)
    assert "Media controls unavailable" in out.getvalue()


def test_close_releases_the_bus_and_survives_a_second_close() -> None:
    port, bus = launch(FakeBus())
    port.close()
    assert wait_until(lambda: bus.disconnected)

    again, bus2 = launch(FakeBus())
    again.close()
    again.close()
    assert wait_until(lambda: bus2.disconnected)


def test_announcements_after_close_are_dropped() -> None:
    port, bus = launch(FakeBus())
    port.close()
    assert wait_until(lambda: bus.disconnected)
    port.publish(MEDIA_STATUS_PLAYING, "Too late")
    time.sleep(0.05)
    assert bus.signals == []
