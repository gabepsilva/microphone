"""Curated Piper voice catalog and the download flag clients render."""

from __future__ import annotations

from pathlib import Path

from tagalong.piper_voices import (
    PIPER_VOICE_IDS,
    piper_voice_label,
    speech_catalog,
)


def test_the_curated_shortlist_matches_the_product_surface() -> None:
    assert PIPER_VOICE_IDS == (
        "en_US-hfc_female-medium",
        "en_US-lessac-high",
        "en_US-lessac-medium",
        "en_US-sam-medium",
        "en_US-bryce-medium",
        "en_US-libritts_r-medium",
    )
    assert len(PIPER_VOICE_IDS) == 6
    assert len(set(PIPER_VOICE_IDS)) == 6


def test_piper_voice_labels_drop_the_locale_prefix() -> None:
    assert piper_voice_label("en_US-lessac-medium") == "Lessac medium"
    assert piper_voice_label("en_US-hfc_female-medium") == "Hfc female medium"
    assert piper_voice_label("en_US-libritts_r-medium") == "Libritts r medium"


def test_speech_catalog_marks_downloaded_from_model_paths(tmp_path: Path) -> None:
    voice = "en_US-lessac-medium"
    (tmp_path / f"{voice}.onnx").write_bytes(b"model")
    (tmp_path / f"{voice}.onnx.json").write_text("{}", encoding="utf-8")

    rows = speech_catalog(home=tmp_path)
    by_id = {row["id"]: row for row in rows}

    assert by_id[voice] == {
        "id": voice,
        "label": "Lessac medium",
        "downloaded": True,
    }
    assert by_id["en_US-sam-medium"]["downloaded"] is False
    assert [row["id"] for row in rows] == list(PIPER_VOICE_IDS)
