import { afterEach, describe, expect, it, mock } from "bun:test";

import {
  PRELOAD_ALLOWLIST,
  allowlistKeys,
  matchesAllowlist,
} from "./preload_allowlist";

type ExposedApi = Record<string, unknown>;

type StateListener = (event: unknown, state: unknown) => void;

async function loadPreload(modulePath: string): Promise<{
  exposed: ExposedApi;
  emitState: (state: unknown) => void;
}> {
  let exposed: ExposedApi | undefined;
  const listeners = new Map<string, Set<StateListener>>();
  mock.module("electron", () => ({
    contextBridge: {
      exposeInMainWorld: (_name: string, api: ExposedApi) => {
        exposed = api;
      },
    },
    ipcRenderer: {
      invoke: async () => undefined,
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
  await import(`${modulePath}?t=${Date.now()}-${Math.random()}`);
  if (exposed === undefined) {
    throw new Error(`preload did not call exposeInMainWorld: ${modulePath}`);
  }
  return {
    exposed,
    emitState: (state: unknown) => {
      for (const set of listeners.values()) {
        for (const listener of set) {
          listener({}, state);
        }
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

  it("onState forwards ipc events and unsubscribe removes the listener", async () => {
    const { exposed, emitState } = await loadPreload("../src/preload.ts");
    const onState = exposed.onState as (
      callback: (state: unknown) => void,
    ) => () => void;

    const seen: unknown[] = [];
    const unsubscribe = onState((state) => {
      seen.push(state);
    });
    emitState({ tts_enabled: true });
    expect(seen).toEqual([{ tts_enabled: true }]);
    unsubscribe();
    emitState({ tts_enabled: false });
    expect(seen).toEqual([{ tts_enabled: true }]);
  });
});
