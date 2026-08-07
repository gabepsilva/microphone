import type { IpcMain, IpcMainInvokeEvent } from "electron";

import type { TagAlongClient } from "./client";
import { CHANNELS, type ChannelName } from "./protocol/channels";
import { validateDispatch } from "./protocol/dispatch_allowlist";

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

  handle(CHANNELS.devicesList, () => client.call("devices.list"));

  handle(CHANNELS.commandsList, () => client.call("commands.list"));

  handle(CHANNELS.capabilities, () => client.call("capabilities"));

  // Single dispatch door: allowlist + per-action payload checks (#96 D3c).
  handle(CHANNELS.dispatch, (_event, action, payload) => {
    try {
      const validated = validateDispatch(action, payload ?? {});
      return client.call("dispatch", {
        action: validated.action,
        payload: validated.payload,
      });
    } catch (error) {
      return Promise.reject(error instanceof Error ? error : new Error(String(error)));
    }
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
