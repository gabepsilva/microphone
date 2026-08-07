import { contextBridge, ipcRenderer } from "electron";

import { CHANNELS } from "../../src/protocol/channels";

/** Planted violation: widens the bridge with a key the allowlist must reject. */
contextBridge.exposeInMainWorld("tagalong", {
  snapshot: () => ipcRenderer.invoke(CHANNELS.snapshot),
  dispatch: (action: string, payload: Record<string, unknown> = {}) =>
    ipcRenderer.invoke(CHANNELS.dispatch, action, payload),
  devicesList: () => ipcRenderer.invoke(CHANNELS.devicesList),
  commandsList: () => ipcRenderer.invoke(CHANNELS.commandsList),
  capabilities: () => ipcRenderer.invoke(CHANNELS.capabilities),
  onState: (callback: (state: unknown) => void) => {
    const listener = (): void => {
      callback(undefined);
    };
    ipcRenderer.on("tagalong:stateChanged", listener);
    return () => {
      ipcRenderer.removeListener("tagalong:stateChanged", listener);
    };
  },
  readFile: (path: string) => ipcRenderer.invoke("tagalong:readFile", path),
});
