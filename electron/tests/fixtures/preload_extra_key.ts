import { contextBridge, ipcRenderer } from "electron";

import { CHANNELS } from "../../src/protocol/channels";

/** Planted violation: widens the bridge with a key the allowlist must reject. */
contextBridge.exposeInMainWorld("tagalong", {
  snapshot: () => ipcRenderer.invoke(CHANNELS.snapshot),
  setTts: (enabled: boolean) => ipcRenderer.invoke(CHANNELS.setTts, enabled),
  readFile: (path: string) => ipcRenderer.invoke("tagalong:readFile", path),
});
