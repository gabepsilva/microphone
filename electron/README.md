# TagAlong Electron client

Talks to a running TUI session over the local JSON-RPC Unix socket
(`$XDG_RUNTIME_DIR/tagalong/tagalong.sock`). It does not start a second
Python runtime.

```bash
npm install electron --no-save
npx electron .
```

Security posture: `contextIsolation: true`, `nodeIntegration: false`, and an
allowlisted preload API (`snapshot`, `setTts`). The socket itself checks
`SO_PEERCRED` and refuses a `/tmp` fallback.
