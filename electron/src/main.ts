import path from "node:path";

import { app, BrowserWindow, ipcMain } from "electron";

import { TagAlongClient } from "./client";
import { registerIpcHandlers } from "./ipc";

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

registerIpcHandlers(ipcMain, client);

void app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
