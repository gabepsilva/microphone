"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tagalong", {
  setTts: (enabled) => ipcRenderer.invoke("tagalong:setTts", enabled),
  setMicrophoneMuted: (muted) =>
    ipcRenderer.invoke("tagalong:setMicrophoneMuted", muted),
  onSnapshot: (handler) => {
    const listener = (_event, snapshot) => handler(snapshot);
    ipcRenderer.on("tagalong:snapshot", listener);
    return () => ipcRenderer.removeListener("tagalong:snapshot", listener);
  },
  onStateChanged: (handler) => {
    const listener = (_event, changed) => handler(changed);
    ipcRenderer.on("tagalong:stateChanged", listener);
    return () => ipcRenderer.removeListener("tagalong:stateChanged", listener);
  },
  onError: (handler) => {
    const listener = (_event, error) => handler(error);
    ipcRenderer.on("tagalong:error", listener);
    return () => ipcRenderer.removeListener("tagalong:error", listener);
  },
});
