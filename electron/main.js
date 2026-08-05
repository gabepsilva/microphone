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

class TagAlongClient {
  constructor() {
    this._socket = null;
    this._connecting = null;
    this._buffer = "";
    this._pending = new Map();
    this._nextId = 1;
  }

  async call(method, params = {}) {
    const socket = await this._ensure();
    const id = this._nextId;
    this._nextId += 1;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      socket.write(
        `${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`,
      );
    });
  }

  _ensure() {
    if (this._socket && !this._socket.destroyed) {
      return Promise.resolve(this._socket);
    }
    if (this._connecting) {
      return this._connecting;
    }
    this._connecting = new Promise((resolve, reject) => {
      const socket = net.createConnection(socketPath());
      const fail = (error) => {
        this._failAll(error);
        this._reset();
        reject(error);
      };
      socket.on("error", fail);
      socket.on("close", () => {
        this._failAll(new Error("connection closed"));
        this._reset();
      });
      socket.on("data", (chunk) => this._onData(chunk));
      socket.on("connect", () => {
        this._socket = socket;
        const id = this._nextId;
        this._nextId += 1;
        this._pending.set(id, {
          resolve: () => {
            this._connecting = null;
            resolve(socket);
          },
          reject: fail,
        });
        socket.write(
          `${JSON.stringify({
            jsonrpc: "2.0",
            id,
            method: "initialize",
            params: { client: "electron" },
          })}\n`,
        );
      });
    });
    return this._connecting;
  }

  _onData(chunk) {
    this._buffer += chunk.toString("utf8");
    let newline = this._buffer.indexOf("\n");
    while (newline !== -1) {
      const line = this._buffer.slice(0, newline);
      this._buffer = this._buffer.slice(newline + 1);
      let payload;
      try {
        payload = JSON.parse(line);
      } catch (error) {
        this._failAll(error);
        return;
      }
      if (payload.method !== "event") {
        const pending = this._pending.get(payload.id);
        if (pending) {
          this._pending.delete(payload.id);
          if (payload.error) {
            pending.reject(new Error(payload.error.message));
          } else {
            pending.resolve(payload.result);
          }
        }
      }
      newline = this._buffer.indexOf("\n");
    }
  }

  _failAll(error) {
    for (const pending of this._pending.values()) {
      pending.reject(error);
    }
    this._pending.clear();
  }

  _reset() {
    this._socket = null;
    this._connecting = null;
    this._buffer = "";
  }
}

const client = new TagAlongClient();

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

ipcMain.handle("tagalong:snapshot", () => client.call("snapshot"));

ipcMain.handle("tagalong:setTts", (_event, enabled) => {
  if (typeof enabled !== "boolean") {
    return Promise.reject(new Error("enabled must be a boolean"));
  }
  return client.call("dispatch", {
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
