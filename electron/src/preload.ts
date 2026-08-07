import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";

import { CHANNELS } from "./protocol/channels";

contextBridge.exposeInMainWorld("tagalong", {
  snapshot: () => ipcRenderer.invoke(CHANNELS.snapshot),
  dispatch: (action: string, payload: Record<string, unknown> = {}) =>
    ipcRenderer.invoke(CHANNELS.dispatch, action, payload),
  devicesList: () => ipcRenderer.invoke(CHANNELS.devicesList),
  commandsList: () => ipcRenderer.invoke(CHANNELS.commandsList),
  capabilities: () => ipcRenderer.invoke(CHANNELS.capabilities),
  onState: (callback: (state: unknown) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, state: unknown): void => {
      callback(state);
    };
    ipcRenderer.on(CHANNELS.stateChanged, listener);
    return () => {
      ipcRenderer.removeListener(CHANNELS.stateChanged, listener);
    };
  },
});
