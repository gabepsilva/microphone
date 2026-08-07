import { describe, expect, it } from "bun:test";
import type { IpcMainInvokeEvent } from "electron";

import { ipcChannelsMatch, registerIpcHandlers } from "../src/ipc";
import { CHANNELS, INVOKE_CHANNELS, PUSH_CHANNELS } from "../src/protocol/channels";
import { ACTIONS } from "../src/protocol/actions";
import { ORPHAN_CHANNELS } from "./fixtures/ipc_orphan_channel";

type Handler = (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown;

function fakeIpcMain(): {
  handle: (channel: string, listener: Handler) => void;
  channels: string[];
  handlers: Map<string, Handler>;
} {
  const channels: string[] = [];
  const handlers = new Map<string, Handler>();
  return {
    channels,
    handlers,
    handle: (channel, listener) => {
      channels.push(channel);
      handlers.set(channel, listener);
    },
  };
}

describe("IPC channel registration", () => {
  it("registers exactly the invoke CHANNELS on the real registrar", () => {
    const ipc = fakeIpcMain();
    const client = { call: async () => undefined };
    const registered = registerIpcHandlers(ipc, client);
    expect([...registered].sort()).toEqual([...INVOKE_CHANNELS].sort());
    expect(ipcChannelsMatch(registered)).toBe(true);
    expect(ipc.channels.sort()).toEqual([...INVOKE_CHANNELS].sort());
    // Every CHANNELS entry is either invoke or push — no silent orphans.
    expect([...INVOKE_CHANNELS, ...PUSH_CHANNELS].sort()).toEqual(
      Object.values(CHANNELS).sort(),
    );
    expect(registered).not.toContain(CHANNELS.stateChanged);
  });

  it("rejects a planted CHANNELS entry with no ipcMain.handle", () => {
    const ipc = fakeIpcMain();
    const client = { call: async () => undefined };
    const registered = registerIpcHandlers(ipc, client);
    expect(Object.values(ORPHAN_CHANNELS)).toContain("tagalong:neverWired");
    expect(registered).not.toContain("tagalong:neverWired");
    expect(ipcChannelsMatch(registered, Object.values(ORPHAN_CHANNELS))).toBe(false);
  });

  it("each invoke handler forwards the matching transport method", async () => {
    const ipc = fakeIpcMain();
    const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
    const client = {
      call: async (method: string, params: Record<string, unknown> = {}) => {
        calls.push({ method, params });
        return { ok: true };
      },
    };
    registerIpcHandlers(ipc, client);
    const event = {} as IpcMainInvokeEvent;

    const expected: Array<{ channel: string; method: string; args?: unknown[] }> = [
      { channel: CHANNELS.snapshot, method: "snapshot" },
      { channel: CHANNELS.devicesList, method: "devices.list" },
      { channel: CHANNELS.commandsList, method: "commands.list" },
      { channel: CHANNELS.codexCatalog, method: "codex.catalog" },
      { channel: CHANNELS.capabilities, method: "capabilities" },
    ];
    for (const { channel, method } of expected) {
      const handler = ipc.handlers.get(channel);
      if (handler === undefined) {
        throw new Error(`missing handler for ${channel}`);
      }
      calls.length = 0;
      await expect(handler(event)).resolves.toEqual({ ok: true });
      expect(calls).toEqual([{ method, params: {} }]);
    }

    const dispatch = ipc.handlers.get(CHANNELS.dispatch);
    if (dispatch === undefined) {
      throw new Error("missing dispatch handler");
    }
    calls.length = 0;
    await expect(
      dispatch(event, ACTIONS.tts_set_enabled, { enabled: true }),
    ).resolves.toEqual({ ok: true });
    expect(calls).toEqual([
      {
        method: "dispatch",
        params: {
          action: ACTIONS.tts_set_enabled,
          payload: { enabled: true },
        },
      },
    ]);
  });
});
