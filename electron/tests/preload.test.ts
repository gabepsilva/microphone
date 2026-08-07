import { afterEach, describe, expect, it, mock } from "bun:test";

import { CHANNELS } from "../src/protocol/channels";
import {
  PRELOAD_ALLOWLIST,
  allowlistKeys,
  matchesAllowlist,
} from "./preload_allowlist";

type ExposedApi = Record<string, unknown>;

type StateListener = (event: unknown, state: unknown) => void;

async function loadPreload(modulePath: string): Promise<{
  exposed: ExposedApi;
  invokes: Array<{ channel: string; args: unknown[] }>;
  emit: (channel: string, payload: unknown) => void;
}> {
  let exposed: ExposedApi | undefined;
  const invokes: Array<{ channel: string; args: unknown[] }> = [];
  const listeners = new Map<string, Set<StateListener>>();
  mock.module("electron", () => ({
    contextBridge: {
      exposeInMainWorld: (_name: string, api: ExposedApi) => {
        exposed = api;
      },
    },
    ipcRenderer: {
      invoke: async (channel: string, ...args: unknown[]) => {
        invokes.push({ channel, args });
        return { ok: true };
      },
      on: (channel: string, listener: StateListener) => {
        const set = listeners.get(channel) ?? new Set<StateListener>();
        set.add(listener);
        listeners.set(channel, set);
      },
      removeListener: (channel: string, listener: StateListener) => {
        listeners.get(channel)?.delete(listener);
      },
    },
  }));
  // Cache-bust so each test loads a fresh module against the current mock.
  // Exercise every bridge method on that single import — Bun's lcov keeps one
  // SF record per path, so a later ?t= import would erase earlier hits (#102).
  await import(`${modulePath}?t=${Date.now()}-${Math.random()}`);
  if (exposed === undefined) {
    throw new Error(`preload did not call exposeInMainWorld: ${modulePath}`);
  }
  return {
    exposed,
    invokes,
    emit: (channel: string, payload: unknown) => {
      for (const listener of listeners.get(channel) ?? []) {
        listener({}, payload);
      }
    },
  };
}

afterEach(() => {
  mock.restore();
});

describe("preload allowlist", () => {
  it("exposes exactly the allowlisted keys from the real preload", async () => {
    const { exposed } = await loadPreload("../src/preload.ts");
    expect(allowlistKeys(exposed)).toEqual([...PRELOAD_ALLOWLIST].sort());
    expect(matchesAllowlist(exposed)).toBe(true);
  });

  it("rejects a planted preload that exposes an extra key", async () => {
    const { exposed } = await loadPreload("./fixtures/preload_extra_key.ts");
    expect(allowlistKeys(exposed)).toContain("readFile");
    expect(matchesAllowlist(exposed)).toBe(false);
  });

  it("wires invoke and push bindings on one module load", async () => {
    const { exposed, invokes, emit } = await loadPreload("../src/preload.ts");

    const snapshot = exposed.snapshot as () => Promise<unknown>;
    const devicesList = exposed.devicesList as () => Promise<unknown>;
    const commandsList = exposed.commandsList as () => Promise<unknown>;
    const capabilities = exposed.capabilities as () => Promise<unknown>;
    const dispatch = exposed.dispatch as (
      action: string,
      payload?: Record<string, unknown>,
    ) => Promise<unknown>;
    const onState = exposed.onState as (
      callback: (state: unknown) => void,
    ) => () => void;
    const onTranscriptSnapshot = exposed.onTranscriptSnapshot as (
      callback: (rows: unknown) => void,
    ) => () => void;
    const onTranscriptEvent = exposed.onTranscriptEvent as (
      callback: (event: unknown) => void,
    ) => () => void;

    await snapshot();
    await devicesList();
    await commandsList();
    await capabilities();
    await dispatch("tts.set_enabled", { enabled: false });

    expect(invokes).toEqual([
      { channel: CHANNELS.snapshot, args: [] },
      { channel: CHANNELS.devicesList, args: [] },
      { channel: CHANNELS.commandsList, args: [] },
      { channel: CHANNELS.capabilities, args: [] },
      {
        channel: CHANNELS.dispatch,
        args: ["tts.set_enabled", { enabled: false }],
      },
    ]);

    const states: unknown[] = [];
    const snapshots: unknown[] = [];
    const events: unknown[] = [];
    const unsubState = onState((state) => {
      states.push(state);
    });
    const unsubSnap = onTranscriptSnapshot((rows) => {
      snapshots.push(rows);
    });
    const unsubEvent = onTranscriptEvent((event) => {
      events.push(event);
    });

    emit(CHANNELS.stateChanged, { tts_enabled: true });
    emit(CHANNELS.transcriptSnapshot, [{ id: 1 }]);
    emit(CHANNELS.transcriptEvent, { name: "transcript.cleared" });
    expect(states).toEqual([{ tts_enabled: true }]);
    expect(snapshots).toEqual([[{ id: 1 }]]);
    expect(events).toEqual([{ name: "transcript.cleared" }]);

    unsubState();
    unsubSnap();
    unsubEvent();
    emit(CHANNELS.stateChanged, { tts_enabled: false });
    emit(CHANNELS.transcriptSnapshot, [{ id: 2 }]);
    emit(CHANNELS.transcriptEvent, { name: "transcript.cleared" });
    expect(states).toEqual([{ tts_enabled: true }]);
    expect(snapshots).toEqual([[{ id: 1 }]]);
    expect(events).toEqual([{ name: "transcript.cleared" }]);
  });
});
