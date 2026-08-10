#!/usr/bin/env bash
# Launch a local TagAlong session and/or the Electron client.
#
# Usage: make start-tui | make start-ui | make start-ui-tui  [DEV=0]
#
# Electron is attach-only (#96): the Python process owns the socket, and when
# it exits the socket dies with it. So every mode here is really a question of
# who starts the session and who has the terminal:
#
#   tui      the TUI, foreground. No Electron.
#   ui       a --headless session, backgrounded, torn down on exit -- unless a
#            session is already live, in which case this only attaches.
#   ui-tui   the TUI, foreground; Electron backgrounded and killed with it.
#
# DEV=1 (the default) runs `bun run dev`, which soft-reloads the window on
# renderer/ edits and rebuilds+restarts Electron on src/ edits. DEV=0 runs the
# plain `bun run start` launch path, which is what you want when smoke-testing
# the real thing. Neither reloads Python: that needs a session restart.

set -euo pipefail

MODE="${1:?usage: start.sh tui|ui|ui-tui}"
DEV="${DEV:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELECTRON_DIR="$REPO_ROOT/electron"

# Linux provides XDG_RUNTIME_DIR. macOS instead provides a randomized,
# per-user TMPDIR (normally mode 0700), which has the security property needed
# by the local socket. Export it so Python and Electron resolve the same path.
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ "$(uname -s)" = "Darwin" ]; then
    XDG_RUNTIME_DIR="${TMPDIR:?macOS did not provide TMPDIR}"
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR%/}"
    export XDG_RUNTIME_DIR
fi
# Mirrors tagalong/transport.py: $XDG_RUNTIME_DIR/tagalong/tagalong.sock, and
# no /tmp fallback.
SOCKET_PATH="${XDG_RUNTIME_DIR:-}/tagalong/tagalong.sock"

# A session can take a few seconds to come up -- it probes codex models and
# sweeps orphaned helpers before it ever binds.
SOCKET_TIMEOUT_SECONDS=60
POLL_INTERVAL_SECONDS=0.2

require_socket_dir() {
    if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
        echo "error: XDG_RUNTIME_DIR is unset, so the session refuses to open a socket"
        echo "and Electron would have nothing to attach to (tagalong/transport.py)."
        exit 2
    fi
}

# `test -S` is not enough: a socket file outlives a session that died hard.
# The only proof a session is listening is a connect that succeeds.
socket_live() {
    uv run python - "$SOCKET_PATH" <<'PY' 2>/dev/null
import socket
import sys

probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
probe.settimeout(1.0)
try:
    probe.connect(sys.argv[1])
except OSError:
    sys.exit(1)
finally:
    probe.close()
PY
}

# $1: pid to watch. A session that dies during startup (a held single-instance
# lock, a missing device) must report that, not spin until the timeout.
wait_for_socket() {
    local session_pid="${1:-}"
    local waited=0
    echo "==> waiting for the session socket at $SOCKET_PATH"
    while ! socket_live; do
        if [ -n "$session_pid" ] && ! kill -0 "$session_pid" 2>/dev/null; then
            echo "error: the session exited before it opened a socket (see its output above)."
            exit 1
        fi
        # Bash arithmetic is integer-only; count polls, not seconds.
        waited=$((waited + 1))
        if [ "$waited" -gt $((SOCKET_TIMEOUT_SECONDS * 5)) ]; then
            echo "error: no socket after ${SOCKET_TIMEOUT_SECONDS}s. Is the session still starting?"
            exit 1
        fi
        sleep "$POLL_INTERVAL_SECONDS"
    done
    echo "==> session socket is live"
}

require_electron_binary() {
    # `make electron-install` sets ELECTRON_SKIP_BINARY_DOWNLOAD=1 -- CI never
    # opens a window -- so a gate-installed tree has the types but no runnable
    # Chromium. Say so rather than failing inside bun.
    local executable="$ELECTRON_DIR/node_modules/electron/dist/electron"
    if [ "$(uname -s)" = "Darwin" ]; then
        executable="$ELECTRON_DIR/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
    fi
    if [ ! -x "$executable" ]; then
        echo "error: no Electron binary at $executable."
        echo "The CI install skips the ~176 MB download. Fetch it once with:"
        echo "    cd electron && bun install && bun run install-electron"
        exit 2
    fi
}

electron_script() {
    if [ "$DEV" = "0" ]; then
        echo "start"
    else
        echo "dev"
    fi
}

case "$MODE" in
tui)
    exec uv run tagalong
    ;;

ui)
    require_socket_dir
    require_electron_binary
    if socket_live; then
        echo "==> attaching to the session already on $SOCKET_PATH"
    else
        echo "==> starting a headless session"
        uv run tagalong --headless &
        SESSION_PID=$!
        # Kill the session we started -- and only that one -- when Electron
        # exits or this script is interrupted.
        trap 'kill "$SESSION_PID" 2>/dev/null || true; wait "$SESSION_PID" 2>/dev/null || true' EXIT
        wait_for_socket "$SESSION_PID"
    fi
    echo "==> launching Electron (bun run $(electron_script))"
    cd "$ELECTRON_DIR" && bun run "$(electron_script)"
    ;;

ui-tui)
    require_socket_dir
    require_electron_binary
    if socket_live; then
        echo "error: a TagAlong session is already running."
        echo "Use 'make start-ui' to attach a window to it."
        exit 2
    fi
    # The TUI needs the terminal, so Electron goes to the background, in its
    # own session: Ctrl-C reaches the TUI's process group only, and the trap
    # below stays the single path that stops Electron.
    #
    # Killing the launcher is not enough -- `bun run dev` spawns Electron as a
    # child and would orphan it -- so the whole group has to go. The new
    # session's leader records its own pid, which is that group's id; reading
    # it back beats guessing whether setsid forked.
    PGID_FILE="$(mktemp)"
    stop_electron() {
        local pgid
        pgid="$(cat "$PGID_FILE" 2>/dev/null || true)"
        if [ -n "$pgid" ]; then
            kill -- "-$pgid" 2>/dev/null || true
        fi
        rm -f "$PGID_FILE"
    }
    trap stop_electron EXIT
    (
        wait_for_socket >&2
        cd "$ELECTRON_DIR"
        exec setsid bash -c 'echo "$$" >"$1"; shift; exec bun run "$@"' \
            _ "$PGID_FILE" "$(electron_script)"
    ) &
    echo "==> starting the TUI; Electron attaches once the socket is up"
    uv run tagalong
    ;;

*)
    echo "error: unknown mode '$MODE' (expected tui, ui, or ui-tui)"
    exit 2
    ;;
esac
