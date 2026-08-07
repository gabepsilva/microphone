import path from "node:path";

import { app, BrowserWindow, ipcMain } from "electron";

import { SessionEvents, TagAlongClient } from "./client";
import { registerIpcHandlers } from "./ipc";
import type { AppState } from "./state";

/** Command connection — never park a long-poll here (#96 G1). */
const commands = new TagAlongClient();
/** Event connection — parked on poll; same actor id as commands. */
const events = new TagAlongClient();

let mainWindow: BrowserWindow | null = null;

function broadcastState(state: AppState): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("tagalong:stateChanged", state);
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
  // Push the already-subscribed snapshot once the page can listen.
  if (sessionEvents.state) {
    broadcastState(sessionEvents.state);
  }
}

registerIpcHandlers(ipcMain, commands);

void app.whenReady().then(async () => {
  try {
    await sessionEvents.start();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("tagalong subscribe failed:", message);
  }
  await createWindow();
});

app.on("window-all-closed", () => {
  sessionEvents.stop();
  commands.close();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
