import type { IpcMain, IpcMainInvokeEvent } from "electron";

import type { TagAlongClient } from "./client";
import { CHANNELS, INVOKE_CHANNELS, type InvokeChannelName } from "./protocol/channels";
import { validateDispatch } from "./protocol/dispatch_allowlist";

type IpcHandle = {
  handle: (
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ) => void;
};

/**
 * Sole socket ``dispatch`` door: validate against DISPATCH_ALLOWLIST, then call.
 *
 * Semgrep forbids bare ``call("dispatch", ...)`` outside this file. Tray mute
 * (#128a) and the IPC handler both enter here so the allowlist cannot be
 * bypassed by a second call site. Marked ``async`` so validateDispatch throws
 * become rejected promises rather than sync main-process crashes (R4).
 */
export async function dispatchAction(
  client: Pick<TagAlongClient, "call">,
  action: unknown,
  payload: unknown = {},
): Promise<unknown> {
  const validated = validateDispatch(action, payload ?? {});
  return client.call("dispatch", {
    action: validated.action,
    payload: validated.payload,
  });
}

/**
 * Detail from a settled refusal/failure outcome, or null when dispatch succeeded.
 *
 * JSON-RPC resolves FORBIDDEN / INVALID / INAPPLICABLE as a *result* with
 * ``type: "rejected"`` (transport outcome_payload) — those never throw, so tray
 * hosts must inspect the outcome (#128 R3).
 */
export function outcomeFailureDetail(outcome: unknown): string | null {
  if (outcome === null || typeof outcome !== "object") {
    return null;
  }
  const record = outcome as { type?: unknown; detail?: unknown };
  if (record.type !== "rejected" && record.type !== "failed") {
    return null;
  }
  return typeof record.detail === "string" ? record.detail : "Request failed";
}

/** Register every invoke CHANNELS entry on ipcMain. Returns those channels. */
export function registerIpcHandlers(
  ipcMain: IpcHandle | Pick<IpcMain, "handle">,
  client: Pick<TagAlongClient, "call">,
): InvokeChannelName[] {
  const registered: InvokeChannelName[] = [];

  const handle = (
    channel: InvokeChannelName,
    listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
  ): void => {
    ipcMain.handle(channel, listener);
    registered.push(channel);
  };

  handle(CHANNELS.snapshot, () => client.call("snapshot"));

  handle(CHANNELS.devicesList, () => client.call("devices.list"));

  handle(CHANNELS.commandsList, () => client.call("commands.list"));

  handle(CHANNELS.codexCatalog, () => client.call("codex.catalog"));

  handle(CHANNELS.speechCatalog, () => client.call("speech.catalog"));

  handle(CHANNELS.capabilities, () => client.call("capabilities"));

  // Single dispatch door: allowlist + per-action payload checks (#96 D3c).
  handle(CHANNELS.dispatch, (_event, action, payload) =>
    dispatchAction(client, action, payload),
  );

  return registered;
}

/** True when registered channels are exactly the invoke CHANNELS set. */
export function ipcChannelsMatch(
  registered: readonly string[],
  channels: readonly string[] = INVOKE_CHANNELS,
): boolean {
  const expected = [...channels].sort();
  const actual = registered.slice().sort();
  return (
    actual.length === expected.length &&
    actual.every((channel, index) => channel === expected[index])
  );
}
