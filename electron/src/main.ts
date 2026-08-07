import path from "node:path";

import { app, BrowserWindow, ipcMain } from "electron";

import { SessionEvents, TagAlongClient } from "./client";
import { registerIpcHandlers } from "./ipc";

/** Command connection — never park a long-poll here (#96 G1). */
const commands = new TagAlongClient();
/** Event connection — parked on poll; same actor id as commands. */
const events = new TagAlongClient();

const sessionEvents = new SessionEvents(events, {
  onState: () => {
    // Renderer binding lands in later #96 phases; this loop keeps live state
    // and recovers from lost / disconnect in the meantime.
  },
  onError: (error) => {
    console.error("tagalong event loop:", error.message);
  },
});

async function createWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 480,
    height: 320,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  await window.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
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
