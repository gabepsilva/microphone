import { describe, expect, it } from "bun:test";

import { ACTIONS } from "../src/protocol/actions";
import {
  DISPATCH_ALLOWLIST,
  isAllowedAction,
  validateDispatch,
} from "../src/protocol/dispatch_allowlist";
import { registerIpcHandlers } from "../src/ipc";
import { CHANNELS } from "../src/protocol/channels";
import type { IpcMainInvokeEvent } from "electron";

describe("DISPATCH_ALLOWLIST", () => {
  it("includes settings and session actions but never session.quit", () => {
    expect(DISPATCH_ALLOWLIST).toContain(ACTIONS.tts_set_enabled);
    expect(DISPATCH_ALLOWLIST).toContain(ACTIONS.session_interrupt);
    expect(DISPATCH_ALLOWLIST).toContain(ACTIONS.message_send);
    expect(isAllowedAction(ACTIONS.session_quit)).toBe(false);
    expect(DISPATCH_ALLOWLIST).not.toContain(ACTIONS.session_quit);
    // transcript.append is #102 ownership territory, not the #96 compose surface.
    expect(DISPATCH_ALLOWLIST).not.toContain(ACTIONS.transcript_append);
  });

  it("rejects unknown actions and session.quit", () => {
    expect(() => validateDispatch(ACTIONS.session_quit, {})).toThrow(
      "action not allowed",
    );
    expect(() => validateDispatch("not.an.action", {})).toThrow("action not allowed");
  });

  it("checks per-action payload fields", () => {
    expect(validateDispatch(ACTIONS.tts_set_enabled, { enabled: false })).toEqual({
      action: ACTIONS.tts_set_enabled,
      payload: { enabled: false },
    });
    expect(() => validateDispatch(ACTIONS.tts_set_enabled, { enabled: "yes" })).toThrow(
      "enabled must be a boolean",
    );
    expect(() => validateDispatch(ACTIONS.microphone_select, { name: 1 })).toThrow(
      "name must be a string or null",
    );
    expect(validateDispatch(ACTIONS.microphone_select, { name: null })).toEqual({
      action: ACTIONS.microphone_select,
      payload: { name: null },
    });
  });
});

describe("dispatch IPC handler", () => {
  it("forwards allowlisted actions and refuses the rest", async () => {
    const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
    const client = {
      call: async (method: string, params: Record<string, unknown> = {}) => {
        calls.push({ method, params });
        return { type: "applied" };
      },
    };
    const handlers = new Map<
      string,
      (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown
    >();
    registerIpcHandlers(
      {
        handle: (
          channel: string,
          listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
        ) => {
          handlers.set(channel, listener);
        },
      },
      client,
    );

    const dispatch = handlers.get(CHANNELS.dispatch);
    if (dispatch === undefined) {
      throw new Error("dispatch handler missing");
    }
    const event = {} as IpcMainInvokeEvent;

    await expect(
      dispatch(event, ACTIONS.tts_set_enabled, { enabled: true }),
    ).resolves.toEqual({ type: "applied" });
    expect(calls.at(-1)).toEqual({
      method: "dispatch",
      params: {
        action: ACTIONS.tts_set_enabled,
        payload: { enabled: true },
      },
    });

    await expect(dispatch(event, ACTIONS.session_quit, {})).rejects.toThrow(
      "action not allowed",
    );
    expect(calls.filter((call) => call.params.action === ACTIONS.session_quit)).toEqual(
      [],
    );
  });
});
