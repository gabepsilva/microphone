# TagAlong Electron client

Talks to a running TUI session over the local JSON-RPC Unix socket
(`$XDG_RUNTIME_DIR/tagalong/tagalong.sock`). It does not start a second
Python runtime.

```bash
# CI / gates: skip the ~176 MB Electron binary; types still install.
ELECTRON_SKIP_BINARY_DOWNLOAD=1 bun install --frozen-lockfile

# Local run: omit the skip flag once so the binary is present, then:
bun run start
```

Make targets from the repo root: `electron-typecheck`, `electron-lint`,
`electron-format-check`, `electron-test` (all use the skip-download install).
`make electron-actions` (in `VERIFY_QUICK`) checks that
`src/protocol/actions.ts` matches the Python `CATALOG`; regenerate with
`make electron-actions-write` after a catalog change.

`typescript` is pinned at 5.9.3 on purpose: `typescript-eslint@8.39.1` peers
`<6.0.0`, and an unpinned `bun install typescript` can resolve 7.x, which
hard-refuses and turns `bun run lint` red. `make ratchet` does not watch
`electron/package.json`.

Security posture: `contextIsolation: true`, `nodeIntegration: false`, and an
allowlisted preload API (`snapshot`, `setTts`). Semgrep under `electron/src/`
refuses flipping those prefs. The socket itself checks `SO_PEERCRED` and
refuses a `/tmp` fallback.

IPC channel names live in `src/protocol/channels.ts` so main and preload share
one spelling; `src/ipc.ts` registers every map entry so an orphan channel fails
`bun test`. Action ids come from generated `src/protocol/actions.ts`. The
socket client is `src/client.ts` — unit tests fake the socket; they do not
launch an Electron window.
