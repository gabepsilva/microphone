import { watch } from "node:fs";
import path from "node:path";

import { app, BrowserWindow, Menu, ipcMain } from "electron";

import { SessionEvents, TagAlongClient, type TranscriptWireEvent } from "./client";
import { registerIpcHandlers } from "./ipc";
import { CHANNELS } from "./protocol/channels";
import type { AppState, TranscriptRow } from "./state";

// Control surface needs no WebGL. On some Linux GPU/driver stacks the GPU
// process dies (error_code=1002) and Chromium aborts the whole app. Set these
// before ready; the start script passes the same flags for earliest effect.
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-gpu-sandbox");
app.commandLine.appendSwitch("in-process-gpu");
// Under Wayland with fractional scaling, an XWayland window is rasterised at
// 1x and upscaled by the compositor — that, not the CSS, is what reads as
// blurry text. The hint selects Wayland natively when the session offers it
// and falls back to X11 otherwise, so Chromium rasterises glyphs at the
// monitor's real scale.
app.commandLine.appendSwitch("ozone-platform-hint", "auto");

/** Command connection — never park a long-poll here (#96 G1). */
const commands = new TagAlongClient();
/** Event connection — parked on poll; same actor id as commands. */
const events = new TagAlongClient();

let mainWindow: BrowserWindow | null = null;

function broadcastState(state: AppState): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.stateChanged, state);
  }
}

function broadcastTranscriptSnapshot(rows: TranscriptRow[]): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.transcriptSnapshot, rows);
  }
}

function broadcastTranscriptEvent(event: TranscriptWireEvent): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.transcriptEvent, event);
  }
}

const sessionEvents = new SessionEvents(events, {
  onState: broadcastState,
  onTranscriptSnapshot: broadcastTranscriptSnapshot,
  onTranscriptEvent: broadcastTranscriptEvent,
  onError: (error) => {
    console.error("tagalong event loop:", error.message);
  },
});

/** Renderer soft-reload for `bun run dev` only — never in a normal start. */
function enableDevRendererReload(): void {
  if (process.env.TAGALONG_ELECTRON_DEV !== "1") {
    return;
  }
  let timer: ReturnType<typeof setTimeout> | null = null;
  watch(path.join(__dirname, "..", "renderer"), { recursive: true }, () => {
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      if (mainWindow !== null && !mainWindow.isDestroyed()) {
        mainWindow.webContents.reloadIgnoringCache();
      }
    }, 50);
  });
}

async function createWindow(): Promise<void> {
  // No File/Edit/View/Window/Help: every entry it would offer is either a
  // lifecycle the TUI owns or a browser affordance this surface does not use.
  Menu.setApplicationMenu(null);
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  await mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
  // Push a real subscribe snapshot once the page can listen — never empty defaults.
  if (sessionEvents.hasSnapshot) {
    broadcastState(sessionEvents.state);
    broadcastTranscriptSnapshot([...sessionEvents.transcript]);
  }
}

registerIpcHandlers(ipcMain, commands);

void app.whenReady().then(async () => {
  // Show the window even when the TUI/socket is unhealthy — attach-only means
  // that case is expected, and an unbounded handshake must not hide the UI.
  await createWindow();
  enableDevRendererReload();
  void sessionEvents.start().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error("tagalong subscribe failed:", message);
  });
});

app.on("window-all-closed", () => {
  sessionEvents.stop();
  commands.close();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
