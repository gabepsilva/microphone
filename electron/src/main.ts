import path from "node:path";

import { app, BrowserWindow, ipcMain } from "electron";

import { TagAlongClient } from "./client";
import { CHANNELS } from "./protocol/channels";

const client = new TagAlongClient();

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

ipcMain.handle(CHANNELS.snapshot, () => client.call("snapshot"));

ipcMain.handle(CHANNELS.setTts, (_event, enabled: unknown) => {
  if (typeof enabled !== "boolean") {
    return Promise.reject(new Error("enabled must be a boolean"));
  }
  return client.call("dispatch", {
    action: "tts.set_enabled",
    payload: { enabled },
  });
});

void app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
