"""Typed-message image attachments.

Images never ride in the prompt as base64. They are saved under the TagAlong
cache, referenced by opaque ids and ``[Image #N]`` tokens in the draft, and
attached to the Codex turn as local files. External callers upload bytes and
receive ids — they never pass filesystem paths.

This module owns pure token logic, disk storage, the id registry, and the OS
clipboard adapter. The TUI owns the draft; Codex owns turning resolved paths
into SDK inputs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from binascii import Error as BinasciiError
from binascii import unhexlify
from collections.abc import Sequence
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

# macOS pasteboard flavours we ask AppleScript to coerce to, preferred first.
# The four-letter codes are Apple's, not MIME types.
_MAC_CLIPBOARD_TYPES: tuple[tuple[str, str], ...] = (
    ("PNGf", ".png"),
    ("JPEG", ".jpg"),
    ("GIFf", ".gif"),
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


class ImageClipboard(Protocol):
    """Port for reading images from the operating-system clipboard.

    Textual only pastes text. Image paste goes through this port so tests can
    supply a fake without patching subprocess or the widget under test.
    """

    def read_image(self) -> ClipboardImage | None:
        """Return a clipboard image, or ``None`` when none is available."""


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


def image_suffix(data: bytes) -> str:
    """File suffix for a recognised image payload, or raise ``ValueError``."""
    if not looks_like_image(data):
        raise ValueError("data is not a recognised image")
    if data.startswith(_PNG_MAGIC):
        return ".png"
    if data.startswith(_JPEG_MAGIC):
        return ".jpg"
    if data.startswith(_GIF_MAGIC):
        return ".gif"
    return ".webp"


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


def _run_clipboard(command: list[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


@dataclass(slots=True)
class SystemImageClipboard:
    """Read images via Wayland ``wl-paste`` or X11 ``xclip``."""

    def read_image(self) -> ClipboardImage | None:
        for mime, suffix in _CLIPBOARD_IMAGE_TYPES:
            data = _run_clipboard(["wl-paste", "-t", mime, "-n"])
            if data is None:
                data = _run_clipboard(
                    ["xclip", "-selection", "clipboard", "-t", mime, "-o"]
                )
            if data is not None and looks_like_image(data):
                return ClipboardImage(data=data, suffix=suffix)
        return None


def read_primary_selection() -> str | None:
    """Read the X11/Wayland primary selection as text, or ``None`` if empty/missing.

    Prefer ``wl-paste --primary``, then ``xclip -selection primary``, then
    ``xsel --primary``. Reuses :func:`_run_clipboard` so helper failures,
    timeouts, and empty stdout are the same clear ``None`` path AC#5 needs.
    """
    for command in (
        ["wl-paste", "--primary", "--no-newline"],
        ["xclip", "-selection", "primary", "-o"],
        ["xsel", "--primary", "--output"],
    ):
        data = _run_clipboard(command)
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace").strip()
        if text:
            return text
    return None


def _decode_applescript_data(payload: bytes, code: str) -> bytes | None:
    """Decode ``«data PNGf89504…»`` — AppleScript's hex form — into bytes.

    ``osascript`` cannot hand back binary, so it prints raw data as hex. The
    guillemets may be mangled by the output encoding; the four-letter code and
    the hex digits are ASCII, so match on those and ignore the rest.
    """
    text = payload.decode("utf-8", errors="ignore")
    match = re.search(rf"data\s*{code}([0-9A-Fa-f]+)", text)
    if match is None:
        return None
    try:
        return unhexlify(match.group(1))
    except BinasciiError:
        return None


@dataclass(slots=True)
class MacImageClipboard:
    """Read images from the macOS pasteboard.

    ``pngpaste`` is preferred when installed because it returns raw bytes.
    Otherwise AppleScript coerces the pasteboard to an image flavour, which
    keeps a plain macOS install working with no Homebrew package.
    """

    def read_image(self) -> ClipboardImage | None:
        data = _run_clipboard(["pngpaste", "-"])
        if data is not None and looks_like_image(data):
            return ClipboardImage(data=data, suffix=".png")
        for code, suffix in _MAC_CLIPBOARD_TYPES:
            payload = _run_clipboard(
                ["osascript", "-e", f"the clipboard as «class {code}»"]
            )
            if payload is None:
                continue
            decoded = _decode_applescript_data(payload, code)
            if decoded is not None and looks_like_image(decoded):
                return ClipboardImage(data=decoded, suffix=suffix)
        return None


def default_image_clipboard(platform: str | None = None) -> ImageClipboard:
    """Return the clipboard adapter for ``platform`` (defaults to this host)."""
    if (platform if platform is not None else sys.platform) == "darwin":
        return MacImageClipboard()
    return SystemImageClipboard()


# Default adapter used by the live app. Tests inject fakes instead.
DEFAULT_IMAGE_CLIPBOARD: ImageClipboard = default_image_clipboard()


@dataclass
class AttachmentRegistry:
    """Opaque attachment ids over an on-disk store.

    Callers never see filesystem paths. ``upload`` returns an id; ``resolve``
    turns ids back into absolute paths for Codex local-image inputs.
    """

    store: AttachmentStore = field(default_factory=AttachmentStore)
    _paths: dict[str, Path] = field(default_factory=dict)

    def upload(self, data: bytes) -> str:
        """Persist *data* and return an opaque id. Raises ``ValueError`` if rejected."""
        path = self.store.save(ClipboardImage(data=data, suffix=image_suffix(data)))
        attachment_id = uuid4().hex
        self._paths[attachment_id] = path
        return attachment_id

    def resolve(self, ids: Sequence[str]) -> tuple[str, ...]:
        """Absolute paths for *ids*, in order. Unknown ids raise ``KeyError``."""
        resolved: list[str] = []
        for attachment_id in ids:
            path = self._paths.get(attachment_id)
            if path is None:
                raise KeyError(attachment_id)
            resolved.append(str(path))
        return tuple(resolved)


@dataclass
class DraftAttachments:
    """Images staged for the current prompt draft, keyed by ``[Image #N]``.

    Numbering is 1-based and stable for the life of the draft. Values are
    opaque attachment ids, never filesystem paths. Deleting a token from the
    text drops that image from the submit set; unused files are left on disk
    (cache) rather than deleted mid-edit, so undo stays safe.
    """

    ids: list[str] = field(default_factory=list)

    def add(self, attachment_id: str) -> str:
        """Register an uploaded image id and return the token to insert."""
        self.ids.append(attachment_id)
        return image_token(len(self.ids))

    def resolve(self, text: str) -> tuple[str, ...]:
        """Attachment ids for tokens still present in ``text``, in number order.

        Unknown numbers and duplicates after the first mention are ignored so
        a hand-edited draft cannot invent ids or attach the same file twice.
        """
        resolved: list[str] = []
        for number in parse_image_numbers(text):
            index = number - 1
            if 0 <= index < len(self.ids):
                attachment_id = self.ids[index]
                if attachment_id not in resolved:
                    resolved.append(attachment_id)
        return tuple(resolved)

    def clear(self) -> None:
        self.ids.clear()

    def __bool__(self) -> bool:
        return bool(self.ids)
