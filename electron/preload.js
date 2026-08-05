"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tagalong", {
  snapshot: () => ipcRenderer.invoke("tagalong:snapshot"),
  setTts: (enabled) => ipcRenderer.invoke("tagalong:setTts", enabled),
});
