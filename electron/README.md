# TagAlong Electron client

Control surface for a running TagAlong session over the local JSON-RPC Unix
socket (`$XDG_RUNTIME_DIR/tagalong/tagalong.sock`). It does **not** start a
second Python runtime. The TUI owns process lifecycle; when the TUI exits, the
socket dies with it (#96 attach-only).

```bash
# CI / gates: skip the ~176 MB Electron binary; types still install.
ELECTRON_SKIP_BINARY_DOWNLOAD=1 bun install --frozen-lockfile

# Local run: omit the skip flag once so the binary is present, then:
bun run start
```

`bun run start` puts `node_modules/.bin` on `PATH` (a bare `electron` in the shell
will fail with `command not found`). On Linux the start script passes
`--no-sandbox` so Chromium does not require a root-owned `chrome-sandbox`, and
`--disable-gpu --disable-gpu-sandbox --in-process-gpu` (mirrored in main via
`disableHardwareAcceleration()` / `commandLine.appendSwitch`) so a flaky GPU
process cannot abort the window.

Make targets from the repo root: `electron-typecheck`, `electron-lint`,
`electron-format-check`, `electron-test`, `electron-coverage` (all use the
skip-download install). `VERIFY_ELECTRON` runs `electron-coverage`, which is
`bun test` with lcov plus per-file floors in `coverage_floors.json`.
`make electron-actions` (in `VERIFY_QUICK`) checks that
`src/protocol/actions.ts` matches the Python `CATALOG`; regenerate with
`make electron-actions-write` after a catalog change.

## Architecture (issue #96)

| Piece           | Role                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| Command socket  | `snapshot`, `dispatch`, `devices.list`, …                                     |
| Event socket    | parked `poll` with `timeout_ms`; applies `state.changed`                      |
| On `lost: true` | resubscribe (terminal per subscription)                                       |
| Preload         | bundled allowlisted bridge (`bun build`) — sandbox forbids relative `require` |
| Dispatch        | main-process `DISPATCH_ALLOWLIST` + payload checks; no `session.quit`         |

Compose is **Agent** provenance under current socket policy
(`actor_for_client` → `ActorKind.AGENT`). Human `Text` is #94. Live transcript
UI / ownership / XSS gates / headless runtime are #102.

Security posture: `contextIsolation: true`, `nodeIntegration: false`,
allowlisted preload, main-process action allowlist, no `/tmp` socket. The
renderer sandbox (Electron default) only allows `require("electron")` in
preload, so `bun run build` bundles `src/preload.ts` into a single
`dist/preload.js` (tsc alone leaves a relative `require("./protocol/channels")`
that fails at runtime). Semgrep under `electron/src/` forbids non-literal
`nodeIntegration` / `contextIsolation`, bare `.handle(...)` outside `ipc.ts`,
and bare `call("dispatch", ...)` outside `ipc.ts`.

## Manual test plan

Prerequisites: PortAudio + a working mic (for device pickers), PipeWire (for
far-end applications), and Codex credentials as for a normal TUI session.

1. **Start the session** (TUI, or headless so this client is the only UI)

   ```bash
   uv run tagalong
   # or: uv run tagalong --headless
   ```

   Leave the TUI running.

2. **Start Electron** (separate terminal, repo root):

   ```bash
   cd electron && bun run start
   ```

   Expect the window status line to show `live` (not a connection error).

3. **Live settings sync**
   - In Electron, toggle **Voice reply**. Confirm the TUI sidebar TTS control
     updates without refreshing Electron.
   - In the TUI, change response policy. Confirm Electron’s **Taga responds
     to** select — and the chip under the prompt — update via long-poll
     (`state.changed`) without clicking Refresh.

4. **Overflow recovery** (optional / automated elsewhere)
   - Covered by
     `tests/test_transport.py::test_poll_reports_lost_after_overflow_and_resubscribe_clears_it`
     and `electron/tests/client.test.ts` SessionEvents cases. Manually: leave
     Electron backgrounded through a burst of settings changes, then confirm
     the sidebar catches up after resubscribe (status returns to `live`).

5. **Device pickers**
   - Open **Microphone** / **Audio stream** selects — options come from
     `devices.list` (inputs via PortAudio, applications via PipeWire graph).
   - Choose a mic; confirm TUI effective/desired update. Toggle mute both ways.

6. **Session actions**
   - Interrupt (`^X`), end voice turn (`^D`), new session (`^N`), save
     transcript (`^S`), plus `^P`, `^K`, `^T`, `^B` — there are no buttons for
     these; the empty screen is their reference and `renderer/shortcuts.js` is
     the one table both it and the key handler read. `/new` runs from the
     palette too. Confirm TUI behaviour matches, and that there is **no Quit**.

7. **Model pickers**
   - **Model** and **Reasoning effort** are selects fed by `codex.catalog`
     (the CLI's own catalog, read once at boot — probing shells out to
     `codex debug models`). Choosing a model whose efforts exclude the running
     one also dispatches the model's default effort, as `adopt_efforts_for`
     does in the TUI.
   - A model the catalog no longer lists still appears while the session runs
     it (`options_including`). Every sidebar change applies immediately — the
     silence field dispatches on `input`, not on blur.

7b. **Speech voice picker** (#124)

- With **Engine** set to Piper, **Voice** is a `<select>` fed by
  `speech.catalog` (curated ids + `downloaded` flags). Undownloaded voices
  are labeled `(download)`. Choosing one dispatches `tts.set_voice`; the
  selection settles when the ready-gated switch succeeds. Hide/disable the
  control when Engine is Edge.

8. **Compose**
   - Type text, optionally attach an image (**+** button or paste), press
     **Enter** — or the send button.
   - Shift+Enter adds a line; Esc dismisses the palette, then clears the draft.
   - Confirm the sent line appears in **both** clients' transcripts as an
     **Agent** line (not Human Text). Nothing local draws a socket peer's
     message; `application._show_remote_message` does, so a session started
     before that fix shows nothing here.
   - Confirm a spoken/model reply still works when Voice reply is on.

9. **Slash commands**
   - Type `/` — the palette lists `commands.list` below the prompt. ↑↓ browse,
     Tab completes, Enter runs, Esc dismisses. Ranking is ported from
     `commands.match_commands`; `tests/commands.test.ts` pins the tiers.
   - `/new` dispatches `session.new`. `/help` names no action — the catalog it
     would print is the menu already on screen.

10. **Markdown**

- A finished Taga answer renders fenced code, headings, lists, quotes,
  tables, and inline emphasis (`renderer/markdown.js`, mirroring
  `tui.uses_markdown_body`). Streaming turns and anything transcribed from
  the room stay literal.
- Links are shown but **never** given an `href`: nothing in this window may
  navigate away from the app, and there is no open-externally channel yet.

11. **Security smoke**
    - DevTools → console: `window.require` / `process` should be unavailable
      (`contextIsolation` + no `nodeIntegration`).
    - Attempting `session.quit` via a forged preload path must fail the
      allowlist (see `electron/tests/dispatch.test.ts`).
