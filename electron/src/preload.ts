import { contextBridge, ipcRenderer } from "electron";

import { CHANNELS } from "./protocol/channels";

contextBridge.exposeInMainWorld("tagalong", {
  snapshot: () => ipcRenderer.invoke(CHANNELS.snapshot),
  setTts: (enabled: boolean) => ipcRenderer.invoke(CHANNELS.setTts, enabled),
});
