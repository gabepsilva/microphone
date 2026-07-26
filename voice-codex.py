#!/usr/bin/env python3
"""Compatibility entry point for the Voice Codex application."""

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
