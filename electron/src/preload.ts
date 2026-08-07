import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";

import { CHANNELS } from "./protocol/channels";

function invokeSnapshot(): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.snapshot);
}

function invokeDispatch(
  action: string,
  payload: Record<string, unknown> = {},
): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.dispatch, action, payload);
}

function invokeDevicesList(): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.devicesList);
}

function invokeCommandsList(): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.commandsList);
}

function invokeCodexCatalog(): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.codexCatalog);
}

function invokeCapabilities(): Promise<unknown> {
  return ipcRenderer.invoke(CHANNELS.capabilities);
}

function onState(callback: (state: unknown) => void): () => void {
  const listener = (_event: IpcRendererEvent, state: unknown): void => {
    callback(state);
  };
  ipcRenderer.on(CHANNELS.stateChanged, listener);
  return () => {
    ipcRenderer.removeListener(CHANNELS.stateChanged, listener);
  };
}

function onTranscriptSnapshot(callback: (rows: unknown) => void): () => void {
  const listener = (_event: IpcRendererEvent, rows: unknown): void => {
    callback(rows);
  };
  ipcRenderer.on(CHANNELS.transcriptSnapshot, listener);
  return () => {
    ipcRenderer.removeListener(CHANNELS.transcriptSnapshot, listener);
  };
}

function onTranscriptEvent(callback: (event: unknown) => void): () => void {
  const listener = (_event: IpcRendererEvent, event: unknown): void => {
    callback(event);
  };
  ipcRenderer.on(CHANNELS.transcriptEvent, listener);
  return () => {
    ipcRenderer.removeListener(CHANNELS.transcriptEvent, listener);
  };
}

contextBridge.exposeInMainWorld("tagalong", {
  snapshot: invokeSnapshot,
  dispatch: invokeDispatch,
  devicesList: invokeDevicesList,
  commandsList: invokeCommandsList,
  codexCatalog: invokeCodexCatalog,
  capabilities: invokeCapabilities,
  onState,
  onTranscriptSnapshot,
  onTranscriptEvent,
});
