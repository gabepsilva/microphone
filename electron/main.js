"use strict";

const { app, BrowserWindow, ipcMain } = require("electron");
const net = require("net");
const path = require("path");

function socketPath() {
  const runtime = process.env.XDG_RUNTIME_DIR;
  if (!runtime) {
    throw new Error("XDG_RUNTIME_DIR is unset; refusing a /tmp socket");
  }
  return path.join(runtime, "tagalong", "tagalong.sock");
}

function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const client = net.createConnection(socketPath());
    let buffer = "";
    client.on("error", reject);
    client.on("data", (chunk) => {
      buffer += chunk.toString("utf8");
      const newline = buffer.indexOf("\n");
      if (newline === -1) {
        return;
      }
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      let payload;
      try {
        payload = JSON.parse(line);
      } catch (error) {
        reject(error);
        client.end();
        return;
      }
      if (payload.method === "event") {
        return;
      }
      client.end();
      if (payload.error) {
        reject(new Error(payload.error.message));
        return;
      }
      resolve(payload.result);
    });
    client.on("connect", () => {
      client.write(
        `${JSON.stringify({ jsonrpc: "2.0", id: method, method, params })}\n`,
      );
    });
  });
}

async function createWindow() {
  const window = new BrowserWindow({
    width: 480,
    height: 320,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  await window.loadFile(path.join(__dirname, "renderer", "index.html"));
}

ipcMain.handle("tagalong:snapshot", async () => {
  await call("initialize", { client: "electron" });
  return call("snapshot");
});

ipcMain.handle("tagalong:setTts", async (_event, enabled) => {
  await call("initialize", { client: "electron" });
  return call("dispatch", {
    action: "tts.set_enabled",
    payload: { enabled },
  });
});

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
