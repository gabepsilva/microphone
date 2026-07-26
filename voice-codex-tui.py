#!/usr/bin/env python3
"""Compatibility entry point for the full Voice Codex Textual application.

This launches the same runtime as ``voice-codex.py``: it loads the startup
configuration, starts User Voice transcription, and starts Them transcription
when an audio output is configured.
"""

from __future__ import annotations

import sys

from voice_codex.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
