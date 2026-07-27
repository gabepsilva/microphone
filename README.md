# Voice Codex

Voice Codex is a Linux desktop assistant that continuously transcribes a
microphone and an optional PipeWire/PulseAudio output, then presents the
conversation to Codex through its Textual interface.

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

The file is also where a session records itself: every setting the sidebar can
change — response policy, speech on/off, speech engine, turn silence, Codex
model and reasoning effort — is written back as you change it, so the next run
starts where the last one left off. Command-line options still override it.
Muting is deliberately not saved; it is how a session is being used at a
moment, not how it is configured.

Some runtime features also require operating-system tools: PipeWire/PulseAudio
(`pactl` and `parec`) for output capture, and `ffmpeg`/`ffplay` for spoken
responses. They are runtime integrations, not Python packages.

## Speech

Codex answers out loud through one of two engines, chosen with `--tts-provider`
or from the sidebar while the session runs:

- `piper` (default) synthesizes on this machine. It reaches the first word in
  roughly a quarter the time Edge needs, because no part of the answer crosses
  the network, and it keeps working offline. Its voice model downloads once, to
  `~/.cache/voice-codex/piper`.
- `edge` uses Microsoft's online voices.

Each provider speaks with its own default voice; `--tts-voice` overrides it for
the provider the session starts with.

The sidebar's engine picker offers a third answer, `No voice reply`, which
silences the session exactly as `Ctrl-T` does — the two are one setting, so
either control moves the other. The engine is remembered while the voice is
off, and choosing one again turns it back on. A session started with
`--tts off` has no engine at all and stays on `No voice reply`.

## Running and presentation

```bash
uv run voice-codex.py
```

`uv run voice-codex-tui.py` is an equivalent compatibility command. Both
commands load `voice.yaml`, initialize User Voice transcription, and also
transcribe Them when `them_output` is configured.

The Textual TUI is the only presentation surface. It is still under active
development; model selection remains a startup option because the Codex SDK
binds it when the conversation thread is created.

The executable scripts are deliberately small compatibility entry points. The
importable `voice_codex` package separates pure configuration and transcript
domain logic from presentation and runtime integrations, so hardware and SDK
boundaries can be tested with deterministic fakes.

## Turn silence

The pause that ends a spoken turn is the largest delay in the loop, and the
sidebar both shows and edits it. A draining bar counts the silence down while
a turn waits; the field above it takes a new window at any point, applied to
the next turn rather than the one already running. Values outside 0.25-30
seconds are refused in place rather than clamped, and Escape restores the
window in force.

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
