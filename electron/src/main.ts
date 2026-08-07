import path from "node:path";

import { app, BrowserWindow, ipcMain } from "electron";

import { SessionEvents, TagAlongClient } from "./client";
import { registerIpcHandlers } from "./ipc";
import { CHANNELS } from "./protocol/channels";
import type { AppState } from "./state";

// Control surface needs no WebGL. On some Linux GPU/driver stacks the GPU
// process dies (error_code=1002) and Chromium aborts the whole app. Set these
// before ready; the start script passes the same flags for earliest effect.
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-gpu-sandbox");
app.commandLine.appendSwitch("in-process-gpu");

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

const sessionEvents = new SessionEvents(events, {
  onState: broadcastState,
  onError: (error) => {
    console.error("tagalong event loop:", error.message);
  },
});

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 720,
    height: 820,
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
  }
}

registerIpcHandlers(ipcMain, commands);

void app.whenReady().then(async () => {
  // Show the window even when the TUI/socket is unhealthy — attach-only means
  // that case is expected, and an unbounded handshake must not hide the UI.
  await createWindow();
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
