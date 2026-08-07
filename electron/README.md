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
   - In the TUI, change response policy. Confirm Electron’s **Respond after**
     select updates via long-poll (`state.changed`) without clicking Refresh.

4. **Overflow recovery** (optional / automated elsewhere)
   - Covered by
     `tests/test_transport.py::test_poll_reports_lost_after_overflow_and_resubscribe_clears_it`
     and `electron/tests/client.test.ts` SessionEvents cases. Manually: leave
     Electron backgrounded through a burst of settings changes, then confirm
     the sidebar catches up after resubscribe (status returns to `live`).

5. **Device pickers**
   - Open **Microphone** / **Far end** selects — options come from
     `devices.list` (inputs via PortAudio, applications via PipeWire graph).
   - Choose a mic; confirm TUI effective/desired update. Toggle mute both ways.

6. **Session actions**
   - **Interrupt**, **End voice turn**, **New session**, **Save transcript**
     each invoke the matching allowlisted action. Confirm TUI behaviour
     matches. Confirm there is **no Quit** control in Electron.

7. **Compose**
   - Type text, optionally attach an image, **Send**.
   - Confirm the TUI shows the message as an **Agent** line (not Human Text).
   - Confirm a spoken/model reply still works when Voice reply is on.

8. **Security smoke**
   - DevTools → console: `window.require` / `process` should be unavailable
     (`contextIsolation` + no `nodeIntegration`).
   - Attempting `session.quit` via a forged preload path must fail the
     allowlist (see `electron/tests/dispatch.test.ts`).
