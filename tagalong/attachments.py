"""Typed-message image attachments.

Images never ride in the prompt as base64. They are saved under the TagAlong
cache, referenced by ``[Image #N]`` tokens in the draft, and attached to the
Codex turn as local files.

This module owns pure token logic, disk storage, and the OS clipboard adapter.
The TUI owns the draft; Codex owns turning stored paths into SDK inputs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

# Product cache root for new data. Other engines may still use older names.
CACHE_ROOT_NAME = "tagalong"
ATTACHMENTS_DIR_NAME = "attachments"

IMAGE_TOKEN_RE = re.compile(r"\[Image #(\d+)\]")

# Clipboard image MIME types we accept, preferred first.
_CLIPBOARD_IMAGE_TYPES: tuple[tuple[str, str], ...] = (
    ("image/png", ".png"),
    ("image/jpeg", ".jpg"),
    ("image/jpg", ".jpg"),
    ("image/webp", ".webp"),
    ("image/gif", ".gif"),
)

# Reject absurd pastes that would thrash the model or the disk.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_GIF_MAGIC = b"GIF8"
_WEBP_MAGIC_PREFIX = b"RIFF"
_WEBP_MAGIC_SUFFIX = b"WEBP"


@dataclass(frozen=True, slots=True)
class ClipboardImage:
    """Raw image bytes taken from the OS clipboard, with a file suffix."""

    data: bytes
    suffix: str


class SystemClipboard(Protocol):
    """Port for reading the operating-system clipboard.

    Textual's own paste replays an in-app copy: ``app.clipboard`` is a string
    the app itself set, never the OS clipboard. Both halves of a real paste
    therefore come through this port, which also lets tests supply a fake
    without patching subprocess or the widget under test.
    """

    def read_image(self) -> ClipboardImage | None:
        """Return a clipboard image, or ``None`` when none is available."""

    def read_text(self) -> str | None:
        """Return clipboard text, or ``None`` when the clipboard holds none."""


def cache_home(
    *,
    xdg_cache_home: str | None = None,
    home: str | None = None,
) -> Path:
    """Return ``$XDG_CACHE_HOME/tagalong`` (or ``~/.cache/tagalong``)."""
    if xdg_cache_home is None:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        root = Path(xdg_cache_home)
    else:
        root = Path(home or os.path.expanduser("~")) / ".cache"
    return root / CACHE_ROOT_NAME


def default_attachments_dir(
    *,
    xdg_cache_home: str | None = None,
    home: str | None = None,
) -> Path:
    """``…/tagalong/attachments`` under the resolved cache home."""
    return cache_home(xdg_cache_home=xdg_cache_home, home=home) / ATTACHMENTS_DIR_NAME


def image_token(number: int) -> str:
    """Render the human-facing marker for attachment ``number`` (1-based)."""
    if number < 1:
        raise ValueError("image numbers are 1-based")
    return f"[Image #{number}]"


def parse_image_numbers(text: str) -> list[int]:
    """Return image numbers mentioned in ``text``, in first-seen order."""
    seen: list[int] = []
    for match in IMAGE_TOKEN_RE.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def looks_like_image(data: bytes) -> bool:
    """True when ``data`` is a recognised, size-bounded image payload."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        return False
    if data.startswith(_PNG_MAGIC) or data.startswith(_JPEG_MAGIC):
        return True
    if data.startswith(_GIF_MAGIC):
        return True
    return (
        data.startswith(_WEBP_MAGIC_PREFIX)
        and len(data) >= 12
        and data[8:12] == _WEBP_MAGIC_SUFFIX
    )


def _normalise_suffix(suffix: str) -> str:
    return suffix if suffix.startswith(".") else f".{suffix}"


@dataclass(slots=True)
class AttachmentStore:
    """Writes clipboard images into a stable on-disk location.

    ``directory`` defaults to the TagAlong attachments cache. Tests inject a
    temporary directory so they never touch the real user cache.
    """

    directory: Path | None = None

    def save(self, image: ClipboardImage) -> Path:
        """Persist ``image`` and return its absolute path."""
        if not looks_like_image(image.data):
            raise ValueError("clipboard data is not a recognised image")
        suffix = _normalise_suffix(image.suffix)
        target = (
            self.directory if self.directory is not None else default_attachments_dir()
        )
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = target / f"{stamp}-{uuid4().hex[:8]}{suffix}"
        path.write_bytes(image.data)
        return path.resolve()


def _run_clipboard(command: list[str], *, timeout: float) -> bytes | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


# A local clipboard helper answers immediately or not at all; the cap only
# keeps a wedged one from freezing the prompt.
_CLIPBOARD_TIMEOUT_SECONDS = 2.0

# osascript hex-encodes the whole image into its own stdout, so its budget
# scales with the paste: a multi-megabyte screenshot takes about a second.
_APPLESCRIPT_TIMEOUT_SECONDS = 5.0

# Asking for the PNG flavor keeps one command for every capture path: macOS
# synthesizes it for anything image-shaped on the pasteboard, including the
# TIFF a screenshot leaves, and fails cleanly when the clipboard holds text.
_APPLESCRIPT_PNG_COMMAND = ("osascript", "-e", "the clipboard as «class PNGf»")

# AppleScript renders raw data as ``«data PNGf<hex>»`` on a single line.
_APPLESCRIPT_PNG_PREFIX = "«data PNGf"
_APPLESCRIPT_DATA_SUFFIX = "»"


def _decode_applescript_png(stdout: bytes) -> bytes | None:
    """Return the PNG bytes inside an AppleScript ``«data PNGf…»`` literal."""
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text.startswith(_APPLESCRIPT_PNG_PREFIX):
        return None
    if not text.endswith(_APPLESCRIPT_DATA_SUFFIX):
        return None
    body = text[len(_APPLESCRIPT_PNG_PREFIX) : -len(_APPLESCRIPT_DATA_SUFFIX)]
    try:
        return bytes.fromhex(body)
    except ValueError:
        return None


def _decode_clipboard_text(data: bytes | None) -> str | None:
    """Return clipboard bytes as text, or ``None`` when there is none to paste."""
    if data is None:
        return None
    text = data.decode("utf-8", errors="replace")
    return text or None


@dataclass(slots=True)
class LinuxImageClipboard:
    """Read the clipboard via Wayland ``wl-paste`` or X11 ``xclip``."""

    def read_image(self) -> ClipboardImage | None:
        for mime, suffix in _CLIPBOARD_IMAGE_TYPES:
            data = _run_clipboard(
                ["wl-paste", "-t", mime, "-n"],
                timeout=_CLIPBOARD_TIMEOUT_SECONDS,
            )
            if data is None:
                data = _run_clipboard(
                    ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
                    timeout=_CLIPBOARD_TIMEOUT_SECONDS,
                )
            if data is not None and looks_like_image(data):
                return ClipboardImage(data=data, suffix=suffix)
        return None

    def read_text(self) -> str | None:
        data = _run_clipboard(
            ["wl-paste", "-n"],
            timeout=_CLIPBOARD_TIMEOUT_SECONDS,
        )
        if data is None:
            data = _run_clipboard(
                ["xclip", "-selection", "clipboard", "-o"],
                timeout=_CLIPBOARD_TIMEOUT_SECONDS,
            )
        return _decode_clipboard_text(data)


@dataclass(slots=True)
class MacImageClipboard:
    """Read the macOS pasteboard through ``osascript`` and ``pbpaste``.

    macOS has no ``wl-paste``/``xclip``, and ``pbpaste`` only speaks text, so
    an image would silently never arrive. AppleScript is the one route to the
    bytes that needs no third-party binary: ``osascript`` ships with the
    system. Text still comes from ``pbpaste``, which is what it is for.
    """

    def read_image(self) -> ClipboardImage | None:
        stdout = _run_clipboard(
            list(_APPLESCRIPT_PNG_COMMAND),
            timeout=_APPLESCRIPT_TIMEOUT_SECONDS,
        )
        if stdout is None:
            return None
        data = _decode_applescript_png(stdout)
        if data is None or not looks_like_image(data):
            return None
        return ClipboardImage(data=data, suffix=".png")

    def read_text(self) -> str | None:
        return _decode_clipboard_text(
            _run_clipboard(["pbpaste"], timeout=_CLIPBOARD_TIMEOUT_SECONDS)
        )


def system_image_clipboard(platform: str | None = None) -> SystemClipboard:
    """Return the clipboard adapter for ``platform`` (this host by default)."""
    if (platform if platform is not None else sys.platform) == "darwin":
        return MacImageClipboard()
    return LinuxImageClipboard()


# Default adapter used by the live app. Tests inject fakes instead.
DEFAULT_IMAGE_CLIPBOARD: SystemClipboard = system_image_clipboard()


@dataclass
class DraftAttachments:
    """Images staged for the current prompt draft, keyed by ``[Image #N]``.

    Numbering is 1-based and stable for the life of the draft. Deleting a
    token from the text drops that image from the submit set; unused files are
    left on disk (cache) rather than deleted mid-edit, so undo stays safe.
    """

    paths: list[str] = field(default_factory=list)

    def add(self, path: str | Path) -> str:
        """Register a saved image and return the token to insert in the prompt."""
        self.paths.append(str(path))
        return image_token(len(self.paths))

    def resolve(self, text: str) -> tuple[str, ...]:
        """Paths for tokens still present in ``text``, in number order.

        Unknown numbers and duplicates after the first mention are ignored so
        a hand-edited draft cannot invent paths or attach the same file twice.
        """
        resolved: list[str] = []
        for number in parse_image_numbers(text):
            index = number - 1
            if 0 <= index < len(self.paths):
                path = self.paths[index]
                if path not in resolved:
                    resolved.append(path)
        return tuple(resolved)

    def clear(self) -> None:
        self.paths.clear()

    def __bool__(self) -> bool:
        return bool(self.paths)
