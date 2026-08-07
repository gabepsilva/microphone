/** IPC channel names shared by main and preload. One spelling, type-checked. */
export const CHANNELS = {
  snapshot: "tagalong:snapshot",
  dispatch: "tagalong:dispatch",
  devicesList: "tagalong:devicesList",
  commandsList: "tagalong:commandsList",
  capabilities: "tagalong:capabilities",
} as const;

export type ChannelName = (typeof CHANNELS)[keyof typeof CHANNELS];
