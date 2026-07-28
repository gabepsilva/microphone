#!/usr/bin/env python3
"""Read the Codex CLI's model catalog and offer it to the interface."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import cast

from .presentation import SessionStatusSink


@dataclass(frozen=True)
class CodexModelOption:
    """A safe subset of one model entry from the Codex CLI catalog."""

    slug: str
    label: str
    efforts: tuple[str, ...]
    default_effort: str


# The API accepts this and answers fastest at it, but no catalog lists it:
# neither `codex debug models` nor the app server's `model/list` names it for
# any model. So it is offered on top of what the catalog reports rather than
# read out of it. A model that turns out not to take it says so on the first
# turn, and ``codex.FALLBACK_EFFORT`` catches that.
UNLISTED_EFFORT = "none"


def _parse_reasoning_efforts(raw_levels: object) -> list[str]:
    """Collect the usable effort names from one model's reasoning levels."""
    if not isinstance(raw_levels, list):
        return []
    efforts: list[str] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, dict):
            continue
        effort = cast(dict[str, object], raw_level).get("effort")
        if isinstance(effort, str) and effort:
            efforts.append(effort)
    return efforts


def _parse_codex_model(model: dict[str, object]):
    """Parse one catalog entry into (priority, option), or None if unusable."""
    if model.get("visibility") != "list" or model.get("supported_in_api") is False:
        return None
    slug = model.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    label = model.get("display_name")
    if not isinstance(label, str) or not label:
        label = slug
    efforts = _parse_reasoning_efforts(model.get("supported_reasoning_levels"))
    if not efforts:
        return None
    default_effort = model.get("default_reasoning_level")
    if not isinstance(default_effort, str) or default_effort not in efforts:
        default_effort = efforts[0]
    # After the default is settled, so that a catalog entry naming an effort
    # it does not list falls back to one the catalog vouches for rather than
    # to the unlisted one.
    if UNLISTED_EFFORT not in efforts:
        efforts.insert(0, UNLISTED_EFFORT)
    priority = model.get("priority")
    return (
        priority if isinstance(priority, int) else sys.maxsize,
        CodexModelOption(slug, label, tuple(efforts), default_effort),
    )


def _parse_codex_model_catalog(payload: object) -> list[CodexModelOption]:
    """Read the catalog defensively; its shape varies by Codex CLI version."""
    if not isinstance(payload, dict):
        return []
    raw_models = cast(dict[str, object], payload).get("models")
    if not isinstance(raw_models, list):
        return []

    options: list[tuple[int, CodexModelOption]] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue
        parsed = _parse_codex_model(cast(dict[str, object], raw_model))
        if parsed is not None:
            options.append(parsed)
    return [
        option
        for _, option in sorted(options, key=lambda item: (item[0], item[1].label))
    ]


def probe_codex_models() -> list[CodexModelOption]:
    """Read the local CLI catalog, falling back to its bundled copy."""
    for command in (
        ["codex", "debug", "models"],
        ["codex", "debug", "models", "--bundled"],
    ):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            options = _parse_codex_model_catalog(json.loads(result.stdout))
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ):
            continue
        if options:
            return options
    return []


def populate_codex_model_catalog(transcript_display: SessionStatusSink) -> None:
    """Populate TUI selectors without delaying audio or interface startup."""
    options = probe_codex_models()
    if not options:
        transcript_display.note(
            "Codex model catalog unavailable; using the configured model"
        )
        return
    transcript_display.set_codex_catalog(
        [(option.label, option.slug) for option in options],
        {option.slug: list(option.efforts) for option in options},
        {option.slug: option.default_effort for option in options},
    )
