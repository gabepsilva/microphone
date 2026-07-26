"""Shared loaders for executable scripts with hyphens in their filenames."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, filename: str) -> ModuleType:
    """Load a CLI script without running its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def voice() -> ModuleType:
    return load_script("voice_codex_script", "voice-codex.py")


@pytest.fixture(scope="session")
def tui() -> ModuleType:
    return load_script("voice_codex_tui_script", "voice-codex-tui.py")
