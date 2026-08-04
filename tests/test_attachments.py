"""Clipboard image attachments, draft tokens, and Codex turn packaging."""

from __future__ import annotations

import asyncio
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from openai_codex import LocalImageInput, TextInput

from tagalong.attachments import (
    MAX_IMAGE_BYTES,
    AttachmentStore,
    ClipboardImage,
    DraftAttachments,
    LinuxImageClipboard,
    MacImageClipboard,
    cache_home,
    default_attachments_dir,
    image_token,
    looks_like_image,
    parse_image_numbers,
    system_image_clipboard,
)
from tagalong.codex import CodexConversation
from tagalong.domain import TranscriptEntry, TranscriptRouter, UserTextMessage
from tagalong.tui import (
    PromptInput,
    PromptPorts,
    SessionState,
    TuiHooks,
    VoiceCodexApp,
)


def tiny_png() -> bytes:
    """A 1x1 PNG the magic-byte checks accept."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


@dataclass
class FakeImageClipboard:
    """Image clipboard that returns a fixed payload (or nothing)."""

    image: ClipboardImage | None = None

    def read_image(self) -> ClipboardImage | None:
        return self.image


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_cache_home_is_under_tagalong(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_home() == tmp_path / "tagalong"
    assert default_attachments_dir() == tmp_path / "tagalong" / "attachments"


def test_cache_home_falls_back_to_dot_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache_home(home=str(tmp_path)) == tmp_path / ".cache" / "tagalong"


def test_image_token_and_parse_round_trip() -> None:
    text = f"see {image_token(1)} and {image_token(2)} again {image_token(1)}"
    assert parse_image_numbers(text) == [1, 2]


def test_looks_like_image_accepts_png_rejects_noise_and_oversize() -> None:
    assert looks_like_image(tiny_png()) is True
    assert looks_like_image(b"not an image") is False
    assert looks_like_image(b"") is False
    huge = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_IMAGE_BYTES + 1)
    assert looks_like_image(huge) is False


def test_attachment_store_writes_under_injected_directory(tmp_path) -> None:
    store = AttachmentStore(directory=tmp_path / "shots")
    path = store.save(ClipboardImage(data=tiny_png(), suffix=".png"))
    assert path.parent == (tmp_path / "shots").resolve()
    assert path.read_bytes() == tiny_png()
    assert path.suffix == ".png"


def test_attachment_store_rejects_non_images(tmp_path) -> None:
    import pytest

    store = AttachmentStore(directory=tmp_path)
    with pytest.raises(ValueError, match="recognised image"):
        store.save(ClipboardImage(data=b"nope", suffix=".png"))


def test_draft_attachments_resolve_only_tokens_still_in_text(tmp_path) -> None:
    draft = DraftAttachments()
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(tiny_png())
    second.write_bytes(tiny_png())
    token_a = draft.add(first)
    token_b = draft.add(second)

    assert token_a == "[Image #1]"
    assert token_b == "[Image #2]"
    assert draft.resolve(f"only {token_b}") == (str(second),)
    assert draft.resolve(f"{token_a} {token_b}") == (str(first), str(second))
    assert draft.resolve("no tokens") == ()
    assert draft.resolve(f"{token_a} {token_a}") == (str(first),)
    assert draft.resolve("[Image #9]") == ()


def test_linux_clipboard_uses_wl_paste(monkeypatch) -> None:
    png = tiny_png()

    def fake_run(command, **_kwargs):
        if command[:2] == ["wl-paste", "-t"] and command[2] == "image/png":
            return type(
                "R",
                (),
                {"returncode": 0, "stdout": png, "stderr": b""},
            )()
        return type("R", (), {"returncode": 1, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("tagalong.attachments.subprocess.run", fake_run)
    assert LinuxImageClipboard().read_image() == ClipboardImage(png, ".png")


def test_linux_clipboard_falls_back_to_xclip(monkeypatch) -> None:
    png = tiny_png()

    def fake_run(command, **_kwargs):
        if command[0] == "wl-paste":
            raise FileNotFoundError
        if (
            command[:3] == ["xclip", "-selection", "clipboard"]
            and "image/png" in command
        ):
            return type(
                "R",
                (),
                {"returncode": 0, "stdout": png, "stderr": b""},
            )()
        return type("R", (), {"returncode": 1, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("tagalong.attachments.subprocess.run", fake_run)
    assert LinuxImageClipboard().read_image() == ClipboardImage(png, ".png")


def test_linux_clipboard_returns_none_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "tagalong.attachments.subprocess.run",
        lambda *_a, **_k: type(
            "R", (), {"returncode": 1, "stdout": b"", "stderr": b""}
        )(),
    )
    assert LinuxImageClipboard().read_image() is None


def applescript_literal(data: bytes) -> bytes:
    """Render ``data`` the way ``osascript`` prints raw clipboard bytes."""
    return f"«data PNGf{data.hex().upper()}»\n".encode()


def fake_osascript(stdout: bytes, *, returncode: int = 0, seen: list | None = None):
    """A ``subprocess.run`` stand-in that answers only the AppleScript read."""

    def run(command, **kwargs):
        if seen is not None:
            seen.append((command, kwargs))
        if command[0] != "osascript":
            return type("R", (), {"returncode": 1, "stdout": b"", "stderr": b""})()
        return type(
            "R", (), {"returncode": returncode, "stdout": stdout, "stderr": b""}
        )()

    return run


def test_mac_clipboard_decodes_the_applescript_png_literal(monkeypatch) -> None:
    png = tiny_png()
    seen: list = []
    monkeypatch.setattr(
        "tagalong.attachments.subprocess.run",
        fake_osascript(applescript_literal(png), seen=seen),
    )

    assert MacImageClipboard().read_image() == ClipboardImage(png, ".png")
    command, kwargs = seen[0]
    # The PNG flavor is what makes one command cover screenshots too, and the
    # budget has to outlast hex-encoding a multi-megabyte paste.
    assert command == ["osascript", "-e", "the clipboard as «class PNGf»"]
    assert kwargs["timeout"] == 5.0


def test_mac_clipboard_is_empty_when_the_clipboard_holds_text(monkeypatch) -> None:
    # osascript exits non-zero with "Can't make some data into the expected
    # type" when the pasteboard has no image flavor.
    monkeypatch.setattr(
        "tagalong.attachments.subprocess.run",
        fake_osascript(b"", returncode=1),
    )
    assert MacImageClipboard().read_image() is None


def test_mac_clipboard_rejects_a_literal_that_is_not_an_image(monkeypatch) -> None:
    monkeypatch.setattr(
        "tagalong.attachments.subprocess.run",
        fake_osascript(applescript_literal(b"not a picture at all")),
    )
    assert MacImageClipboard().read_image() is None


def test_mac_clipboard_rejects_an_oversized_paste(monkeypatch) -> None:
    oversized = tiny_png() + b"\x00" * MAX_IMAGE_BYTES
    monkeypatch.setattr(
        "tagalong.attachments.subprocess.run",
        fake_osascript(applescript_literal(oversized)),
    )
    assert MacImageClipboard().read_image() is None


def test_mac_clipboard_rejects_malformed_applescript_output(monkeypatch) -> None:
    png = tiny_png()
    truncated = applescript_literal(png).replace(b"\xc2\xbb", b"")
    odd_hex = f"«data PNGf{png.hex().upper()}F»\n".encode()
    non_hex = "«data PNGfnot-hex»\n".encode()
    for stdout in (truncated, odd_hex, non_hex, b"the clipboard is empty\n"):
        monkeypatch.setattr(
            "tagalong.attachments.subprocess.run", fake_osascript(stdout)
        )
        assert MacImageClipboard().read_image() is None


def test_system_image_clipboard_follows_the_platform(monkeypatch) -> None:
    assert isinstance(system_image_clipboard("darwin"), MacImageClipboard)
    assert isinstance(system_image_clipboard("linux"), LinuxImageClipboard)

    monkeypatch.setattr("tagalong.attachments.sys.platform", "darwin")
    assert isinstance(system_image_clipboard(), MacImageClipboard)
    monkeypatch.setattr("tagalong.attachments.sys.platform", "linux")
    assert isinstance(system_image_clipboard(), LinuxImageClipboard)


# --------------------------------------------------------------------------
# Domain + Codex packaging
# --------------------------------------------------------------------------


def test_router_carries_images_on_entries() -> None:
    router = TranscriptRouter()
    request = router.ingest(
        "Text",
        "look [Image #1]",
        "T1",
        True,
        images=("/tmp/a.png",),
    )
    assert request is not None
    assert request.entries[0].images == ("/tmp/a.png",)


def test_context_entries_never_embed_filesystem_paths() -> None:
    entry = TranscriptEntry(
        "Text",
        "see [Image #1]",
        "T1",
        images=("/home/secret/a.png",),
    )
    request = type("R", (), {"entries": (entry,)})()
    serialised = CodexConversation.context_entries(request)
    assert serialised == [
        {
            "timestamp": "T1",
            "source": "Text",
            "text": "see [Image #1]",
        }
    ]
    assert "/home/secret" not in str(serialised)


def test_build_turn_input_stays_a_string_without_images() -> None:
    request = TranscriptRouter().ingest("Them", "hi", "T1", True)
    assert CodexConversation.build_turn_input(request) == (
        CodexConversation.build_prompt(request)
    )


def test_build_turn_input_attaches_local_images_in_token_order() -> None:
    router = TranscriptRouter()
    request = router.ingest(
        "Text",
        "look [Image #1]",
        "T1",
        True,
        images=("/cache/tagalong/attachments/a.png",),
    )
    turn_input = CodexConversation.build_turn_input(request)
    assert isinstance(turn_input, list)
    assert isinstance(turn_input[0], TextInput)
    assert isinstance(turn_input[1], LocalImageInput)
    assert turn_input[1].path == "/cache/tagalong/attachments/a.png"
    assert "[Image #1]" in turn_input[0].text
    assert "/cache/tagalong" not in turn_input[0].text


def test_conversation_ingest_queues_images_onto_the_turn(monkeypatch) -> None:
    """ingest carries paths onto the queued request the worker will package."""
    from tagalong.codex import CodexSettings, load_codex_sdk
    from tests.test_conversation import FakeCodex, FakeDisplay

    load_codex_sdk()
    codex = FakeCodex()
    monkeypatch.setattr("tagalong.codex.Codex", lambda: codex)
    # Do not start the real worker; we only inspect the queue.
    monkeypatch.setattr(CodexConversation, "_worker", lambda self: None)
    conversation = CodexConversation(
        CodexSettings(
            sandbox="read-only",
            model="gpt-5.6-luna",
            reasoning_effort="low",
        ),
        FakeDisplay(),
    )
    conversation.ingest(
        "Text",
        "describe [Image #1]",
        respond=True,
        timestamp="T1",
        images=("/tmp/shot.png",),
    )
    queued = conversation.requests.get_nowait()
    assert queued.request.entries[-1].images == ("/tmp/shot.png",)
    turn_input = CodexConversation.build_turn_input(queued.request)
    assert isinstance(turn_input, list)
    assert isinstance(turn_input[0], TextInput)
    assert isinstance(turn_input[1], LocalImageInput)
    assert turn_input[1].path == "/tmp/shot.png"
    conversation.close()


# --------------------------------------------------------------------------
# TUI: prompt owns the draft; clipboard is injected
# --------------------------------------------------------------------------


def _app_with_image_ports(
    tmp_path: Path,
    *,
    on_user_text=None,
    image: ClipboardImage | None = None,
) -> VoiceCodexApp:
    """Build an app whose prompt will see the injected clipboard and store."""
    app = VoiceCodexApp(
        SessionState(),
        TuiHooks(on_user_text=on_user_text),
    )
    app.prompt_ports = PromptPorts(
        clipboard=FakeImageClipboard(image),
        store=AttachmentStore(directory=tmp_path),
    )
    return app


def test_pasting_an_image_inserts_a_token_and_submits_paths(tmp_path) -> None:
    received: list[UserTextMessage] = []
    app = _app_with_image_ports(
        tmp_path,
        on_user_text=received.append,
        image=ClipboardImage(data=tiny_png(), suffix=".png"),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            prompt = app.query_one("#input", PromptInput)
            prompt.focus()
            await pilot.press("ctrl+v")
            assert "[Image #1]" in prompt.value
            assert prompt.draft
            await pilot.press("enter")
            assert not prompt.draft

    asyncio.run(exercise())

    assert len(received) == 1
    message = received[0]
    assert message.text == "[Image #1]"
    assert len(message.images) == 1
    assert Path(message.images[0]).parent == tmp_path.resolve()
    assert Path(message.images[0]).read_bytes() == tiny_png()


def test_clearing_the_prompt_drops_staged_attachments(tmp_path) -> None:
    app = _app_with_image_ports(
        tmp_path,
        image=ClipboardImage(data=tiny_png(), suffix=".png"),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            prompt = app.query_one("#input", PromptInput)
            prompt.focus()
            await pilot.press("ctrl+v")
            assert prompt.draft
            await pilot.press("ctrl+c")
            assert prompt.value == ""
            assert not prompt.draft

    asyncio.run(exercise())


def test_deleting_a_token_before_submit_drops_that_image(tmp_path) -> None:
    received: list[UserTextMessage] = []
    app = _app_with_image_ports(
        tmp_path,
        on_user_text=received.append,
        image=ClipboardImage(data=tiny_png(), suffix=".png"),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            prompt = app.query_one("#input", PromptInput)
            prompt.focus()
            await pilot.press("ctrl+v")
            await pilot.press("ctrl+v")
            assert prompt.value.count("[Image #") == 2
            # Keep only the second token in the draft text.
            prompt.value = "[Image #2]"
            await pilot.press("enter")

    asyncio.run(exercise())

    assert len(received) == 1
    assert received[0].text == "[Image #2]"
    assert len(received[0].images) == 1
    assert received[0].images[0].endswith(".png")


def test_text_paste_still_works_when_clipboard_has_no_image(tmp_path) -> None:
    app = _app_with_image_ports(tmp_path, image=None)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            prompt = app.query_one("#input", PromptInput)
            app.copy_to_clipboard("pasted text")
            prompt.focus()
            await pilot.press("ctrl+v")
            assert prompt.value == "pasted text"

    asyncio.run(exercise())


def test_user_text_message_is_the_hook_payload() -> None:
    message = UserTextMessage(text="hi", images=("/a.png",))
    assert message.text == "hi"
    assert message.images == ("/a.png",)
