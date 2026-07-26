# Voice Codex

Voice Codex is a Linux desktop assistant that continuously transcribes a
microphone and an optional PipeWire/PulseAudio output, then presents the
conversation to Codex. `voice-codex-tui.py` provides a separate Textual UI
surface.

## Reproducible setup

The project requires CPython 3.12 and [uv](https://docs.astral.sh/uv/). The
exact Python dependencies live in `uv.lock`; do not use an unlocked install for
development or CI.

```bash
uv sync --locked --all-groups
cp voice.example.yaml voice.yaml
make hooks
```

`voice.yaml` holds local audio-device choices and is deliberately ignored by
Git. The example keeps all choices interactive, so it is safe to copy on a new
machine.

Some runtime features also require operating-system tools: PipeWire/PulseAudio
(`pactl` and `parec`) for output capture, and `ffmpeg`/`ffplay` for spoken
responses. They are runtime integrations, not Python packages.

## Quality gate

Run the complete local gate before committing:

```bash
make ci
```

This requires Gitleaks 8.30.0 or a compatible version on `PATH`. The gate
checks formatting, linting, types, tests and branch coverage, shell syntax,
static Python security findings, dependency vulnerabilities, and secrets in the
working tree and Git history. See [QUALITY.md](QUALITY.md) for the precise
contract and [AGENTS.md](AGENTS.md) for rules that apply to AI-assisted work.

`make hooks` installs a fast `make verify` gate before every commit and the
complete `make ci` gate before every push. Validate the installed hooks without
creating a commit via `make hook-check`.

GitHub Actions runs the equivalent locked checks in a clean environment and a
separate immutable Gitleaks action. Protect `master` by requiring both checks
before merging.
