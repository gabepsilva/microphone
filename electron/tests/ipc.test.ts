import { describe, expect, it } from "bun:test";
import type { IpcMainInvokeEvent } from "electron";

import { ipcChannelsMatch, registerIpcHandlers } from "../src/ipc";
import { CHANNELS } from "../src/protocol/channels";
import { ORPHAN_CHANNELS } from "./fixtures/ipc_orphan_channel";

function fakeIpcMain(): {
  handle: (
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ) => void;
  channels: string[];
} {
  const channels: string[] = [];
  return {
    channels,
    handle: (channel) => {
      channels.push(channel);
    },
  };
}

const fakeClient = {
  call: async () => undefined,
};

describe("IPC channel registration", () => {
  it("registers exactly the CHANNELS map on the real registrar", () => {
    const ipc = fakeIpcMain();
    const registered = registerIpcHandlers(ipc, fakeClient);
    expect([...registered].sort()).toEqual(Object.values(CHANNELS).sort());
    expect(ipcChannelsMatch(registered)).toBe(true);
    expect(ipc.channels.sort()).toEqual(Object.values(CHANNELS).sort());
  });

  it("rejects a planted CHANNELS entry with no ipcMain.handle", () => {
    const ipc = fakeIpcMain();
    const registered = registerIpcHandlers(ipc, fakeClient);
    expect(Object.values(ORPHAN_CHANNELS)).toContain("tagalong:neverWired");
    expect(registered).not.toContain("tagalong:neverWired");
    expect(ipcChannelsMatch(registered, ORPHAN_CHANNELS)).toBe(false);
  });
});
