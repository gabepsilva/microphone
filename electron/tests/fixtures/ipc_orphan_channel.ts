/**
 * Planted violation: CHANNELS gains a map entry that never gets an
 * ipcMain.handle. Types cannot see a dead map entry; the orphan gate must.
 */
import type { IpcMainInvokeEvent } from "electron";

import type { TagAlongClient } from "../../src/client";
import { ACTIONS } from "../../src/protocol/actions";
import { CHANNELS, type ChannelName } from "../../src/protocol/channels";

const ORPHAN_CHANNELS = {
  ...CHANNELS,
  neverWired: "tagalong:neverWired",
} as const;

type IpcHandle = {
  handle: (
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ) => void;
};

/** Registers only the real CHANNELS keys — leaves neverWired without a handler. */
export function registerOrphanIpcHandlers(
  ipcMain: IpcHandle,
  client: Pick<TagAlongClient, "call">,
): ChannelName[] {
  const registered: ChannelName[] = [];
  const handle = (
    channel: ChannelName,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ): void => {
    ipcMain.handle(channel, listener);
    registered.push(channel);
  };

  handle(CHANNELS.snapshot, () => client.call("snapshot"));
  handle(CHANNELS.setTts, (_event, enabled) => {
    if (typeof enabled !== "boolean") {
      return Promise.reject(new Error("enabled must be a boolean"));
    }
    return client.call("dispatch", {
      action: ACTIONS.tts_set_enabled,
      payload: { enabled },
    });
  });

  return registered;
}

export { ORPHAN_CHANNELS };
