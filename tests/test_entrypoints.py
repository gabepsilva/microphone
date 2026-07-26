from __future__ import annotations

import runpy
from pathlib import Path


def test_tui_entrypoint_launches_the_configured_application(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("voice_codex.cli.main", lambda: called.append(True))

    entrypoint = Path(__file__).resolve().parents[1] / "voice-codex-tui.py"
    runpy.run_path(str(entrypoint), run_name="__main__")

    assert called == [True]
