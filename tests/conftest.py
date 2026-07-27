"""Shared fixtures for the Voice Codex package.

Every other test module imports the one module it exercises directly, so the
import at the top of a test file names its subject. The interface is the
exception: Textual is slow to import and most of the suite never touches it,
so it stays behind a session fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tui():
    from voice_codex import tui as tui_module

    return tui_module
