"use strict";
// Planted: sandboxed preload with a relative require (the tsc-only shape).
const electron_1 = require("electron");
const channels_1 = require("./protocol/channels");
electron_1.contextBridge.exposeInMainWorld("tagalong", {
  dispatch: (action, payload = {}) =>
    electron_1.ipcRenderer.invoke(channels_1.CHANNELS.dispatch, action, payload),
});
