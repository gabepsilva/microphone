"""Shared imports for the Voice Codex package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def voice():
    from voice_codex import cli

    return cli


@pytest.fixture(scope="session")
def tui():
    from voice_codex import tui as tui_module

    return tui_module
