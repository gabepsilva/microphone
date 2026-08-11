"""Shared fixtures for the TagAlong package.

Every other test module imports the one module it exercises directly, so the
import at the top of a test file names its subject. The interface is the
exception: Textual is slow to import and most of the suite never touches it,
so it stays behind a session fixture.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path():
    """Use a short, private path that fits macOS ``sockaddr_un.sun_path``."""
    path = Path(tempfile.mkdtemp(prefix="tl-test-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture(scope="session")
def tui():
    from tagalong import tui as tui_module

    return tui_module
