"use strict";

const { app, BrowserWindow, ipcMain } = require("electron");
const net = require("net");
const path = require("path");

/** Same cadence as ``tagalong.transport.EventPump`` (50ms). */
const POLL_MS = 50;
const RECONNECT_MS = 250;

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

  get connected() {
    return this._socket !== null && !this._socket.destroyed;
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

  close() {
    if (this._socket && !this._socket.destroyed) {
      this._socket.destroy();
    }
    this._failAll(new Error("connection closed"));
    this._reset();
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
      // Reserved for a future push path; today the server is request/response
      // and live updates arrive through subscribe + poll, like LocalClient.
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

/**
 * Socket-side twin of ``EventPump``: subscribe once, poll on the same
 * cadence, and forward ``state.changed`` fragments to the display.
 */
class SessionSync {
  constructor(client, emit) {
    this._client = client;
    this._emit = emit;
    this._running = false;
    this._timer = null;
  }

  start() {
    if (this._running) {
      return;
    }
    this._running = true;
    this._connect();
  }

  stop() {
    this._running = false;
    if (this._timer !== null) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    this._client.close();
  }

  _schedule(fn, delayMs) {
    if (!this._running) {
      return;
    }
    if (this._timer !== null) {
      clearTimeout(this._timer);
    }
    this._timer = setTimeout(() => {
      this._timer = null;
      fn.call(this);
    }, delayMs);
  }

  async _connect() {
    if (!this._running) {
      return;
    }
    try {
      const snapshot = await this._client.call("subscribe");
      this._emit("snapshot", snapshot);
      this._schedule(this._tick, 0);
    } catch (error) {
      this._emit("error", { message: error.message });
      this._client.close();
      this._schedule(this._connect, RECONNECT_MS);
    }
  }

  async _tick() {
    if (!this._running) {
      return;
    }
    try {
      const polled = await this._client.call("poll");
      for (const event of polled.events || []) {
        if (event.name === "state.changed") {
          this._emit("stateChanged", event.payload || {});
        }
      }
      this._schedule(this._tick, POLL_MS);
    } catch (error) {
      this._emit("error", { message: error.message });
      this._client.close();
      this._schedule(this._connect, RECONNECT_MS);
    }
  }
}

const client = new TagAlongClient();
let mainWindow = null;
let sync = null;

function emitToRenderer(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(`tagalong:${channel}`, payload);
  }
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 720,
    height: 480,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  await mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  sync = new SessionSync(client, emitToRenderer);
  sync.start();
}

ipcMain.handle("tagalong:setTts", (_event, enabled) => {
  if (typeof enabled !== "boolean") {
    return Promise.reject(new Error("enabled must be a boolean"));
  }
  return client.call("dispatch", {
    action: "tts.set_enabled",
    payload: { enabled },
  });
});

ipcMain.handle("tagalong:setMicrophoneMuted", (_event, muted) => {
  if (typeof muted !== "boolean") {
    return Promise.reject(new Error("muted must be a boolean"));
  }
  return client.call("dispatch", {
    action: "microphone.set_muted",
    payload: { muted },
  });
});

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (sync) {
    sync.stop();
    sync = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});
