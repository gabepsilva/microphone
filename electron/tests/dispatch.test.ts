import { describe, expect, it } from "bun:test";

import { ACTIONS } from "../src/protocol/actions";
import {
  DISPATCH_ALLOWLIST,
  isAllowedAction,
  validateDispatch,
} from "../src/protocol/dispatch_allowlist";
import { dispatchAction, registerIpcHandlers } from "../src/ipc";
import { CHANNELS } from "../src/protocol/channels";
import type { IpcMainInvokeEvent } from "electron";

describe("DISPATCH_ALLOWLIST", () => {
  it("includes settings and session actions but never session.quit", () => {
    expect(DISPATCH_ALLOWLIST).toContain(ACTIONS.tts_set_enabled);
    expect(DISPATCH_ALLOWLIST).toContain(ACTIONS.tts_set_voice);
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
    expect(() => validateDispatch(ACTIONS.tts_set_enabled, null)).toThrow(
      "payload must be an object",
    );
    expect(() => validateDispatch(ACTIONS.tts_set_enabled, [])).toThrow(
      "payload must be an object",
    );

    expect(validateDispatch(ACTIONS.microphone_set_muted, { muted: true })).toEqual({
      action: ACTIONS.microphone_set_muted,
      payload: { muted: true },
    });
    expect(validateDispatch(ACTIONS.audio_stream_set_muted, { muted: false })).toEqual({
      action: ACTIONS.audio_stream_set_muted,
      payload: { muted: false },
    });
    expect(() =>
      validateDispatch(ACTIONS.audio_stream_set_muted, { muted: "no" }),
    ).toThrow("muted must be a boolean");

    expect(validateDispatch(ACTIONS.audio_stream_select, { name: "Zoom" })).toEqual({
      action: ACTIONS.audio_stream_select,
      payload: { name: "Zoom" },
    });

    expect(validateDispatch(ACTIONS.response_policy_set, { policy: "always" })).toEqual(
      {
        action: ACTIONS.response_policy_set,
        payload: { policy: "always" },
      },
    );
    expect(() => validateDispatch(ACTIONS.response_policy_set, { policy: 1 })).toThrow(
      "policy must be a string",
    );

    expect(validateDispatch(ACTIONS.tts_set_provider, { provider: "edge" })).toEqual({
      action: ACTIONS.tts_set_provider,
      payload: { provider: "edge" },
    });
    expect(() =>
      validateDispatch(ACTIONS.tts_set_provider, { provider: null }),
    ).toThrow("provider must be a string");

    expect(
      validateDispatch(ACTIONS.tts_set_voice, { voice: "en_US-amy-medium" }),
    ).toEqual({
      action: ACTIONS.tts_set_voice,
      payload: { voice: "en_US-amy-medium" },
    });
    expect(() => validateDispatch(ACTIONS.tts_set_voice, { voice: 1 })).toThrow(
      "voice must be a string",
    );

    expect(validateDispatch(ACTIONS.codex_set_model, { model: "o3" })).toEqual({
      action: ACTIONS.codex_set_model,
      payload: { model: "o3" },
    });
    expect(() => validateDispatch(ACTIONS.codex_set_model, { model: 3 })).toThrow(
      "model must be a string",
    );

    expect(validateDispatch(ACTIONS.codex_set_reasoning, { effort: "high" })).toEqual({
      action: ACTIONS.codex_set_reasoning,
      payload: { effort: "high" },
    });
    expect(() =>
      validateDispatch(ACTIONS.codex_set_reasoning, { effort: false }),
    ).toThrow("effort must be a string");

    expect(validateDispatch(ACTIONS.turn_silence_set, { seconds: 1.5 })).toEqual({
      action: ACTIONS.turn_silence_set,
      payload: { seconds: 1.5 },
    });
    expect(() =>
      validateDispatch(ACTIONS.turn_silence_set, { seconds: Number.NaN }),
    ).toThrow("seconds must be a number");
    expect(() => validateDispatch(ACTIONS.turn_silence_set, { seconds: "1" })).toThrow(
      "seconds must be a number",
    );

    expect(validateDispatch(ACTIONS.message_send, { text: "hello" })).toEqual({
      action: ACTIONS.message_send,
      payload: { text: "hello" },
    });
    expect(() => validateDispatch(ACTIONS.message_send, { text: 1 })).toThrow(
      "text must be a string",
    );

    expect(validateDispatch(ACTIONS.attachment_upload, { data: "abc" })).toEqual({
      action: ACTIONS.attachment_upload,
      payload: { data: "abc" },
    });
    expect(() => validateDispatch(ACTIONS.attachment_upload, { data: null })).toThrow(
      "data must be a string",
    );

    // No required payload fields — exercise the default arm.
    expect(validateDispatch(ACTIONS.session_new, {})).toEqual({
      action: ACTIONS.session_new,
      payload: {},
    });
    expect(validateDispatch(ACTIONS.session_interrupt, {})).toEqual({
      action: ACTIONS.session_interrupt,
      payload: {},
    });
    expect(validateDispatch(ACTIONS.voice_end_turn, {})).toEqual({
      action: ACTIONS.voice_end_turn,
      payload: {},
    });
    expect(validateDispatch(ACTIONS.transcript_save, {})).toEqual({
      action: ACTIONS.transcript_save,
      payload: {},
    });
  });
});

describe("dispatchAction", () => {
  it("validates then calls socket dispatch (tray and IPC share this door)", async () => {
    const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
    const client = {
      call: async (method: string, params: Record<string, unknown> = {}) => {
        calls.push({ method, params });
        return { type: "applied" };
      },
    };
    await expect(
      dispatchAction(client, ACTIONS.microphone_set_muted, { muted: true }),
    ).resolves.toEqual({ type: "applied" });
    expect(calls).toEqual([
      {
        method: "dispatch",
        params: {
          action: ACTIONS.microphone_set_muted,
          payload: { muted: true },
        },
      },
    ]);
    expect(() => dispatchAction(client, ACTIONS.session_quit, {})).toThrow(
      "action not allowed",
    );
    expect(calls).toHaveLength(1);
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
