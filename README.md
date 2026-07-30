# TagAlong

TagAlong is a Linux desktop assistant that continuously transcribes a
microphone and an optional PipeWire/PulseAudio output, then presents the
conversation to Codex through its Textual interface.

## Reproducible setup

The project requires CPython 3.12 and [uv](https://docs.astral.sh/uv/). The
exact Python dependencies live in `uv.lock`; do not use an unlocked install for
development or CI.

```bash
uv sync --locked --all-groups
make hooks
```

`tagalong.yaml` is optional and holds local audio-device choices when present; it
is deliberately ignored by Git. With no file, TagAlong uses Piper speech,
listens for both voice sources, starts without a meeting application, and
chooses the first available microphone. `tagalong.example.yaml` documents every
setting if you want to create the file by hand.

The file is also where a session records itself: every setting the sidebar can
change — response policy, speech on/off, speech engine, turn silence, Codex
model and reasoning effort — is written back as you change it, so the next run
starts where the last one left off. Command-line options still override it.
Muting is deliberately not saved; it is how a session is being used at a
moment, not how it is configured.

Some runtime features also require operating-system tools: PipeWire
(`pw-record`, `pw-link`, and `pw-dump`) for capturing the far end, `pactl` for
naming output sinks, and `ffmpeg`/`ffplay` for spoken responses. They are
runtime integrations, not Python packages.

## Speech

Codex answers out loud through one of two engines, chosen with `--tts-provider`
or from the sidebar while the session runs:

- `piper` (default) synthesizes on this machine. It reaches the first word in
  roughly a quarter the time Edge needs, because no part of the answer crosses
  the network, and it keeps working offline. Its voice model downloads once, to
  `~/.cache/tagalong/piper`.
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
uv run tagalong.py
```

The command loads `tagalong.yaml` when it exists, initializes User Voice
transcription, and also transcribes Them when `them_stream` names an
application. If the file or an input device is absent, the typed/Codex session
still starts. The microphone picker under the mic meter refreshes as devices
are connected, so an input can be selected later without restarting.

## Hearing the far end

Them is captured from one application's own PipeWire playback stream rather
than from a speaker's monitor. `--them-stream` names the application — the name
the desktop shows for it, such as `Chromium` or `ZOOM VoiceEngine` — and the
session links that application's audio into a capture node of its own while it
goes on playing to the real speakers, unrouted and unchanged. Nothing has to be
pointed at a virtual device.

Codex's own speech is a different stream and is never linked, so it cannot be
transcribed back as the far end. That is a property of the wiring rather than a
filter that can miss.

An application only appears in the audio graph once it starts playing, so the
startup menu lists what is making sound right now. A name given with
`--them-stream` or saved in `tagalong.yaml` is taken as written and picked up
whenever that application appears — including after it restarts, and including
every stream it opens, which is what makes a browser with several tabs work.

The sidebar picks it too, at any point in the session: the far-end picker sits
under the speaker meter, refreshes itself as applications start and stop
playing, and offers `None` alongside them. It lists only applications it
has heard streaming — a system speech daemon holds a stream open from boot
without ever using it, and that is not a far end anyone can choose. Having
streamed is the test rather than streaming now, so a meeting stays selectable
during a pause in it. Starting a meeting after the
session is already running needs nothing but that picker.

The channel behind it is built the first time an application is chosen and
closed again when none is, so a session that only ever listens to the
microphone never loads the second speech model. Moving between two
applications is not a rebuild — the tap follows a name, so it is re-pointed
where it stands.

The Textual TUI is the only presentation surface. It is still under active
development; model selection remains a startup option because the Codex SDK
binds it when the conversation thread is created.

The executable scripts are deliberately small compatibility entry points. The
importable `tagalong` package separates pure configuration and transcript
domain logic from presentation and runtime integrations, so hardware and SDK
boundaries can be tested with deterministic fakes.

## Codex speed

Sessions ask for Codex Fast mode by default. Everything downstream of the
first token is already tuned for latency — sentences stream to speech as they
complete, and Piper starts speaking one in about 158ms — but none of it can
begin before Codex has produced that token, so the service tier is the part of
the wait the rest of the pipeline cannot recover.

It consumes more credits. `--no-codex-fast` asks for the standard tier
instead, and `codex_fast: false` in `tagalong.yaml` makes that the default for a
machine. The startup summary and the sidebar both name the tier in force.

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

## Real-environment smoke test

The default suite fakes hardware, processes, and network services so CI stays
deterministic. On a configured Linux desktop, explicitly exercise those real
boundaries with:

```bash
make smoke-real
```

This records a fraction of a second from the default microphone, queries
PipeWire/PulseAudio outputs and the installed Codex model catalog, and
synthesizes a short sentence through both Piper and Edge without playing it.
It may download the default Piper voice and contacts Microsoft's Edge speech
service. It is intentionally separate from `make ci`.
