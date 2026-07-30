from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path


def test_cli_import_does_not_load_the_microphone_adapter(monkeypatch) -> None:
    import moonshine_voice

    original_getattr = moonshine_voice.__getattr__

    def prohibit_microphone_adapter(name: str):
        if name == "MicTranscriber":
            raise AssertionError("CLI import must not load microphone capture")
        return original_getattr(name)

    monkeypatch.setattr(moonshine_voice, "__getattr__", prohibit_microphone_adapter)
    sys.modules.pop("tagalong.cli", None)

    importlib.import_module("tagalong.cli")


def test_entrypoint_launches_the_configured_application(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("tagalong.cli.main", lambda: called.append(True))

    entrypoint = Path(__file__).resolve().parents[1] / "tagalong.py"
    runpy.run_path(str(entrypoint), run_name="__main__")

    assert called == [True]
