# TagAlong Electron client

Talks to a running TUI session over the local JSON-RPC Unix socket
(`$XDG_RUNTIME_DIR/tagalong/tagalong.sock`). It does not start a second
Python runtime.

```bash
npm install electron --no-save
npx electron .
```

Live state uses the same path as the in-process TUI pump: `subscribe` once,
then `poll` every 50ms for `state.changed` fragments. The main process owns
that loop and pushes updates into the renderer; the renderer does not scrape
`snapshot` on a timer.

Security posture: `contextIsolation: true`, `nodeIntegration: false`, and an
allowlisted preload API (`setTts`, `setMicrophoneMuted`, plus snapshot /
`state.changed` / error subscriptions). The socket itself checks
`SO_PEERCRED` and refuses a `/tmp` fallback.
