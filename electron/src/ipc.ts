import type { IpcMain, IpcMainInvokeEvent } from "electron";

import type { TagAlongClient } from "./client";
import { ACTIONS } from "./protocol/actions";
import { CHANNELS, type ChannelName } from "./protocol/channels";

type IpcHandle = {
  handle: (
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ) => void;
};

/** Register every CHANNELS entry on ipcMain. Returns the channels that got handlers. */
export function registerIpcHandlers(
  ipcMain: IpcHandle | Pick<IpcMain, "handle">,
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

/** True when registered channels are exactly the CHANNELS map values. */
export function ipcChannelsMatch(
  registered: readonly string[],
  channels: Readonly<Record<string, string>> = CHANNELS,
): boolean {
  const expected = Object.values(channels).slice().sort();
  const actual = registered.slice().sort();
  return (
    actual.length === expected.length &&
    actual.every((channel, index) => channel === expected[index])
  );
}
