#!/usr/bin/env python3
"""Compatibility entry point for the full Voice Codex Textual application.

This launches the same runtime as ``voice-codex.py``: it loads the startup
configuration, starts User Voice transcription, and starts Them transcription
when an audio output is configured.
"""

from __future__ import annotations

from voice_codex.cli import run_entrypoint

if __name__ == "__main__":
    run_entrypoint()
