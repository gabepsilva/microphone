"""Curated Piper voices the session offers for mid-session switching.

The full upstream catalog is large and includes multi-speaker models this
runtime never passes a speaker id to. The shortlist is the product surface:
enough speakers to matter, including the few ``low``-only names, without a
live HuggingFace fetch from the UI.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .piper_tts import DEFAULT_MODEL_HOME, model_paths

# Locked shortlist for #124 — every id exists in upstream voices.json with
# ``num_speakers = 1`` (validated in design round 3 / D10).
PIPER_VOICE_IDS: tuple[str, ...] = (
    "en_US-lessac-medium",
    "en_US-lessac-low",
    "en_US-ryan-medium",
    "en_US-amy-medium",
    "en_US-joe-medium",
    "en_US-kristin-medium",
    "en_US-kusal-medium",
    "en_US-danny-low",
    "en_US-kathleen-low",
)


def piper_voice_label(voice_id: str) -> str:
    """Turn ``en_US-lessac-medium`` into a short picker label."""
    rest = voice_id.removeprefix("en_US-").replace("-", " ")
    if not rest:
        return voice_id
    return rest[:1].upper() + rest[1:]


def speech_catalog(
    *,
    voice_ids: Sequence[str] = PIPER_VOICE_IDS,
    home: str | Path = DEFAULT_MODEL_HOME,
) -> list[dict[str, object]]:
    """Return ``{id, label, downloaded}`` rows for the curated Piper list.

    ``downloaded`` is an exists-check on the model pair under the Piper home,
    so a client can label a 63 MB surprise before the operator clicks it.
    """
    rows: list[dict[str, object]] = []
    for voice_id in voice_ids:
        model, config = model_paths(voice_id, home)
        rows.append(
            {
                "id": voice_id,
                "label": piper_voice_label(voice_id),
                "downloaded": model.exists() and config.exists(),
            }
        )
    return rows
