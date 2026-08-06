/** IPC channel names shared by main and preload. One spelling, type-checked. */
export const CHANNELS = {
  snapshot: "tagalong:snapshot",
  setTts: "tagalong:setTts",
} as const;

export type ChannelName = (typeof CHANNELS)[keyof typeof CHANNELS];
