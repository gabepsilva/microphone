/** IPC channel names shared by main and preload. One spelling, type-checked. */
export const CHANNELS = {
  snapshot: "tagalong:snapshot",
  dispatch: "tagalong:dispatch",
  devicesList: "tagalong:devicesList",
  commandsList: "tagalong:commandsList",
  capabilities: "tagalong:capabilities",
  /** Main → renderer push; not registered via ipcMain.handle. */
  stateChanged: "tagalong:stateChanged",
} as const;

export type ChannelName = (typeof CHANNELS)[keyof typeof CHANNELS];

/** Channels registered with ipcMain.handle (invoke direction). */
export const INVOKE_CHANNELS = [
  CHANNELS.snapshot,
  CHANNELS.dispatch,
  CHANNELS.devicesList,
  CHANNELS.commandsList,
  CHANNELS.capabilities,
] as const;

export type InvokeChannelName = (typeof INVOKE_CHANNELS)[number];
