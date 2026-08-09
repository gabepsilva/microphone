#!/usr/bin/env python3
"""Publish what Taga is saying to the desktop's media controls, or not.

A session can expose itself as an MPRIS player — the same contract the media
keys, the shell's player widget, and the lock-screen controls follow — so
whichever one of those the room uses can stop a reply mid-sentence. MPRIS is
read and written across the session bus by name; ``dbus-fast`` supplies the
walking part and this module the shape of the player being described.

Two design choices keep the MPRIS surface small, because a player that
promises less can still be useful and stops being trustworthy the moment it
advertises a capability it cannot honor:

  * Stop-only. The widget gets a working stop and no pause, play, or seek, so
    ``CanPlay``, ``CanPause``, and ``CanSeek`` are all false and the few
    methods left behave the way the widget's enabled controls do. A desktop
    that believes the properties needs no lie to render them.
  * Always visible while speaking. The status climbs to "Playing" the moment
    speech is queued, not when the first sample comes out, and falls back to
    "Stopped" only when the last owed sentence has finished — the same span
    ``SpeechActivity`` covers, so the widget does not flicker between
    sentences. The sentence text rides in the Metadata that joins the status,
    and reaches the bus only when the engine has something actually audible.

Layout:

  * :data:`MediaControlsPort` — the narrow surface ``queued_tts`` announces
    through, so the engines depend on a protocol and not on DBus
  * :func:`build_media_controls` — the composition helper behind
    ``--media-controls``
  * :class:`MprisMediaControls` — the named player on the session bus
  * :class:`NullMediaControls` — a port that answers nothing, used when media
    controls are off, when the platform has no session bus, or when the name
    is already claimed

Everything that touches ``dbus_fast`` sits at the bottom of this file, defined
only when the package is importable — it is an optional, Linux-only
dependency, so importing this module must never require it.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import Protocol, runtime_checkable

MEDIA_STATUS_PLAYING = "playing"
MEDIA_STATUS_STOPPED = "stopped"

# The object path below is fixed by the MPRIS spec. The bus name is ours,
# appended to the player prefix so two tagalong instances never collide with
# each other's player.
PLAYER_BUS_NAME = "org.mpris.MediaPlayer2.tagalong"
OBJECT_PATH = "/org/mpris/MediaPlayer2"

_MEDIA_PLAYER2 = "org.mpris.MediaPlayer2"
_MEDIA_PLAYER2_PLAYER = "org.mpris.MediaPlayer2.Player"


@runtime_checkable
class MediaControlsPort(Protocol):
    """Announce to the desktop what this session's speech is doing.

    ``publish`` is called from any thread — the engines speak from their own
    workers — and must return immediately. The statuses are
    :data:`MEDIA_STATUS_PLAYING` and :data:`MEDIA_STATUS_STOPPED`; the title
    is the sentence currently being spoken, when the caller knows one.
    """

    def publish(self, status: str, title: str | None = None) -> None: ...

    def close(self) -> None: ...


class MediaControlsUnavailable(RuntimeError):
    """Raised when no session bus can be claimed. Callers fall back to mute."""


class NullMediaControls:
    """A media port that does nothing, for the sessions that have no desktop.

    Optional feature, optional dependency, optional desktop: wherever MPRIS is
    not wanted or cannot exist, this is what receives the announcements and
    drops them. Being a real object rather than ``None`` is what lets the rest
    of the runtime call ``publish`` and ``close`` without checking anything.
    """

    def publish(self, status: str, title: str | None = None) -> None:
        pass

    def close(self) -> None:
        pass


def _dbus_fast_available() -> bool:
    """Report whether this platform could host an MPRIS player.

    Linux is where the session bus lives; everywhere else this answers False
    without importing anything, because ``dbus-fast`` is Linux-only here.
    """
    if sys.platform != "linux":
        return False
    try:
        import dbus_fast  # noqa: F401 - import probe; the except below is the point
    except ImportError:
        return False
    return True


def build_media_controls(
    enabled: bool, request_stop=None, stream=None, bus_factory=None
) -> MediaControlsPort:
    """Return the session's media port, or a silent one when that is impossible.

    *enabled* is the operator's choice (``--media-controls``). Without it, the
    null port is returned without a word, because the feature was not asked
    for. With it, a session bus and an unclaimed name are required; if either
    is missing, the desktop gets nothing but the user gets one line on
    *stream*, so a missing widget is never mistaken for quiet speech.

    *bus_factory* is how tests hand the port a pretend session bus; sessions
    never need it.
    """
    if not enabled:
        return NullMediaControls()
    try:
        return MprisMediaControls(request_stop=request_stop, bus_factory=bus_factory)
    except MediaControlsUnavailable as error:
        print(
            f"Media controls unavailable: {error}",
            file=sys.stderr if stream is None else stream,
            flush=True,
        )
        return NullMediaControls()


class MprisMediaControls:
    """Own the MPRIS player, served from its own event-loop thread.

    The running session never waits on the bus: ``publish`` is used from any
    engine thread and hops onto the loop here, so transcription, synthesis,
    and playback keep their latencies while one message is being written. The
    same hop carries the desktop's Stop back out as a plain callback.

    Claiming the well-known name is what makes this *a player* on the bus, and
    DO_NOT_QUEUE makes the claim honest: a second tagalong is not hidden in a
    queue waiting to steal the first one's identity the moment it leaves. The
    constructor raises :class:`MediaControlsUnavailable` when the name cannot
    be held, which its caller turns into the null port and one line.
    """

    READY_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        request_stop: Callable[[], None] | None = None,
        *,
        bus_factory=None,
    ):
        if not _dbus_fast_available():
            raise MediaControlsUnavailable(
                "dbus-fast is not installed (Linux-only, required for MPRIS)"
            )
        self._request_stop = request_stop
        self._bus_factory = bus_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bus = None
        self._player = None
        self._outcome = threading.Event()
        self._startup_error: BaseException | None = None
        self._ready = False
        self._closing = False
        threading.Thread(
            target=self._run_loop,
            name="MprisControls",
            daemon=True,
        ).start()
        if not self._outcome.wait(timeout=self.READY_TIMEOUT_SECONDS):
            raise MediaControlsUnavailable(
                f"the session bus did not answer within {self.READY_TIMEOUT_SECONDS}s"
            )
        if self._startup_error is not None:
            raise MediaControlsUnavailable(str(self._startup_error))
        self._ready = True

    # -- the loop thread -----------------------------------------------------

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._establish())
            self._outcome.set()
            loop.run_forever()
        except BaseException as error:
            self._startup_error = error
            self._outcome.set()
        finally:
            loop.close()
            self._loop = None

    async def _establish(self) -> None:
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType, NameFlag, RequestNameReply

        if self._bus_factory is None:
            bus = MessageBus(bus_type=BusType.SESSION)
        else:
            bus = self._bus_factory()
        try:
            await bus.connect()
        except Exception as error:
            raise MediaControlsUnavailable(
                f"could not connect to the session bus: {error}"
            ) from error
        self._bus = bus
        self._player = _Player(self._on_stop)
        bus.export(OBJECT_PATH, _Root())
        bus.export(OBJECT_PATH, self._player)
        reply = await bus.request_name(PLAYER_BUS_NAME, flags=NameFlag.DO_NOT_QUEUE)
        if reply != RequestNameReply.PRIMARY_OWNER:
            raise MediaControlsUnavailable(
                f"{PLAYER_BUS_NAME!r} is already claimed on this session bus"
            )

    def _on_stop(self) -> None:
        """The desktop pressed stop: retire the speech this session owes."""
        if self._request_stop is not None:
            self._request_stop()

    # -- the port ------------------------------------------------------------

    def publish(self, status: str, title: str | None = None) -> None:
        """Hand an announcement to the loop that writes to the bus."""
        loop = self._loop
        player = self._player
        if (
            self._ready
            and loop is not None
            and player is not None
            and loop.is_running()
        ):
            loop.call_soon_threadsafe(player.update, status, title)

    def close(self) -> None:
        """Ask the loop behind this port to hold its silence from now on."""
        loop = self._loop
        if self._ready and not self._closing and loop is not None and loop.is_running():
            self._closing = True
            loop.call_soon_threadsafe(self._shutdown)

    def _shutdown(self) -> None:
        """On the loop thread: release the bus and stop serving messages."""
        bus, self._bus = self._bus, None
        if bus is not None:
            bus.disconnect()
        asyncio.get_running_loop().stop()


def _unsupported(operation: str) -> Exception:
    """The error a stop-only player gives to methods it does not have."""
    from dbus_fast.errors import DBusError

    return DBusError(
        "org.freedesktop.DBus.Error.NotSupported",
        f"TagAlong does not support {operation}",
    )


# The two MPRIS interfaces are defined for real only when dbus-fast exists,
# so the module itself stays importable on platforms that never serve MPRIS.
if _dbus_fast_available():
    from typing import Annotated

    from dbus_fast.annotations import (
        DBusBool,
        DBusDict,
        DBusDouble,
        DBusInt64,
        DBusObjectPath,
        DBusSignature,
        DBusStr,
    )
    from dbus_fast.constants import PropertyAccess
    from dbus_fast.service import ServiceInterface, dbus_method, dbus_property

    class _Root(ServiceInterface):
        """The player's descriptive face: what it is, what it can never do.

        The methods and properties are fixed by the MPRIS spec; almost all of
        them answer the same two values, which is the point — a widget that
        asks "can it do anything?" learns there is nothing but an identity.
        """

        def __init__(self) -> None:
            super().__init__(_MEDIA_PLAYER2)

        @dbus_method()
        def Raise(self):
            raise _unsupported("Raise")

        @dbus_method()
        def Quit(self):
            raise _unsupported("Quit")

        @dbus_property(access=PropertyAccess.READ)
        def CanQuit(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanRaise(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanSetFullscreen(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def Fullscreen(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def HasTrackList(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def Identity(self) -> DBusStr:
            return "TagAlong"

        @dbus_property(access=PropertyAccess.READ)
        def SupportedUriSchemes(self) -> Annotated[list[str], DBusSignature("as")]:
            return []

        @dbus_property(access=PropertyAccess.READ)
        def SupportedMimeTypes(self) -> Annotated[list[str], DBusSignature("as")]:
            return []

    class _Player(ServiceInterface):
        """The playing part of the player, bound to its stop callback.

        Holds only the state the desktop asked for: the status, the sentence
        being said, and how this track is numbered. The announcement flow
        itself lives in :class:`MprisMediaControls`, which calls ``update``
        on its own loop; Stop is the one control this player honors.
        """

        def __init__(self, on_stop) -> None:
            super().__init__(_MEDIA_PLAYER2_PLAYER)
            self._status = "Stopped"
            self._title: str | None = None
            self._track = 0
            self._on_stop = on_stop

        def _metadata(self) -> DBusDict:
            from dbus_fast.signature import Variant

            if self._title is None:
                return {}
            return {
                "xesam:title": Variant("s", self._title),
                "mpris:trackid": Variant("o", f"{OBJECT_PATH}/Track/{self._track}"),
            }

        def update(self, status: str, title: str | None) -> None:
            changes: dict[str, object] = {}
            mpris = "Playing" if status == MEDIA_STATUS_PLAYING else "Stopped"
            if mpris != self._status:
                self._status = mpris
                changes["PlaybackStatus"] = mpris
            if mpris == "Playing" and title and title != self._title:
                self._title = title
                self._track += 1
                changes["Metadata"] = self._metadata()
            if changes:
                self.emit_properties_changed(changes)

        # Stop is the one control this player honors.
        @dbus_method()
        def Stop(self):
            self._on_stop()

        @dbus_method()
        def Play(self):
            raise _unsupported("Play")

        @dbus_method()
        def Pause(self):
            raise _unsupported("Pause")

        @dbus_method()
        def PlayPause(self):
            raise _unsupported("PlayPause")

        @dbus_method()
        def Next(self):
            raise _unsupported("Next")

        @dbus_method()
        def Previous(self):
            raise _unsupported("Previous")

        @dbus_method()
        def Seek(self, offset: DBusInt64):  # noqa: ARG002 - fixed by the MPRIS spec
            raise _unsupported("Seek")

        @dbus_method()
        def SetPosition(self, track_id: DBusObjectPath, position: DBusInt64):  # noqa: ARG002 - fixed by the MPRIS spec
            raise _unsupported("SetPosition")

        @dbus_method()
        def OpenUri(self, uri: DBusStr):  # noqa: ARG002 - fixed by the MPRIS spec
            raise _unsupported("OpenUri")

        @dbus_property(access=PropertyAccess.READ)
        def PlaybackStatus(self) -> DBusStr:
            return self._status

        @dbus_property(access=PropertyAccess.READ)
        def LoopStatus(self) -> DBusStr:
            return "None"

        @dbus_property(access=PropertyAccess.READ)
        def Rate(self) -> DBusDouble:
            return 1.0

        @dbus_property(access=PropertyAccess.READ)
        def Shuffle(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def Metadata(self) -> DBusDict:
            return self._metadata()

        @dbus_property(access=PropertyAccess.READ)
        def Volume(self) -> DBusDouble:
            return 1.0

        @dbus_property(access=PropertyAccess.READ)
        def Position(self) -> DBusInt64:
            return 0

        @dbus_property(access=PropertyAccess.READ)
        def MinimumRate(self) -> DBusDouble:
            return 1.0

        @dbus_property(access=PropertyAccess.READ)
        def MaximumRate(self) -> DBusDouble:
            return 1.0

        @dbus_property(access=PropertyAccess.READ)
        def CanGoNext(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanGoPrevious(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanPlay(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanPause(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanSeek(self) -> DBusBool:
            return False

        @dbus_property(access=PropertyAccess.READ)
        def CanControl(self) -> DBusBool:
            return True
