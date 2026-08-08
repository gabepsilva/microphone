"""Curated Piper voices the session offers for mid-session switching.

The full upstream catalog is large and includes multi-speaker models this
runtime does not expose a speaker picker for. The shortlist is the product
surface: a handful of named voices without a live HuggingFace fetch from the
UI. ``en_US-libritts_r-medium`` is the one multi-speaker exception — Piper
synthesizes it with the engine default speaker when none is passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Product shortlist (#124). Every id exists in upstream voices.json.
# Kept free of ``piper_tts`` / numpy so ``control.actions`` (and Edge-only
# sessions) can import the id tuple without loading the local synthesis stack.
PIPER_VOICE_IDS: tuple[str, ...] = (
    "en_US-hfc_female-medium",
    "en_US-lessac-high",
    "en_US-lessac-medium",
    "en_US-sam-medium",
    "en_US-bryce-medium",
    "en_US-libritts_r-medium",
)


def piper_voice_label(voice_id: str) -> str:
    """Turn ``en_US-lessac-medium`` into a short picker label."""
    rest = voice_id.removeprefix("en_US-").replace("_", " ").replace("-", " ")
    if not rest:
        return voice_id
    return rest[:1].upper() + rest[1:]


def speech_catalog(
    *,
    voice_ids: Sequence[str] = PIPER_VOICE_IDS,
    home: str | Path | None = None,
) -> list[dict[str, object]]:
    """Return ``{id, label, downloaded}`` rows for the curated Piper list.

    ``downloaded`` is an exists-check on the model pair under the Piper home,
    so a client can label a large download before the operator clicks it.
    """
    # Import here so listing ids never pays for onnxruntime/numpy — same
    # reason ``speech.py`` defers engine imports (Edge-only sessions).
    from .piper_tts import DEFAULT_MODEL_HOME, model_paths

    model_home = DEFAULT_MODEL_HOME if home is None else home
    rows: list[dict[str, object]] = []
    for voice_id in voice_ids:
        model, config = model_paths(voice_id, model_home)
        rows.append(
            {
                "id": voice_id,
                "label": piper_voice_label(voice_id),
                "downloaded": model.exists() and config.exists(),
            }
        )
    return rows
