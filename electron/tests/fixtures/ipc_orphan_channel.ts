/**
 * Planted violation: CHANNELS gains a map entry that never gets an
 * ipcMain.handle. Types cannot see a dead map entry; the orphan gate must.
 * Assert against the real registerIpcHandlers — do not copy it here.
 */
import { CHANNELS } from "../../src/protocol/channels";

export const ORPHAN_CHANNELS = {
  ...CHANNELS,
  neverWired: "tagalong:neverWired",
} as const;
