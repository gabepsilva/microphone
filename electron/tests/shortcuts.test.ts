import { describe, expect, it } from "bun:test";

import {
  POLICY_ORDER,
  SHORTCUTS_PROMPT,
  SHORTCUTS_SESSION,
  commandForEvent,
  nextPolicy,
} from "../renderer/shortcuts.js";

import { isAllowedAction } from "../src/protocol/dispatch_allowlist.js";

describe("shortcuts", () => {
  it("cycles response policies in catalog order and wraps", () => {
    expect(nextPolicy("audio")).toBe("both");
    expect(nextPolicy("quiet")).toBe("audio");
    // An unrecognised policy must still land somewhere selectable.
    expect(POLICY_ORDER).toContain(nextPolicy("nonsense"));
  });

  it("resolves the session bindings it documents", () => {
    expect(commandForEvent({ key: "p", ctrlKey: true })).toBe("session.cycle_policy");
    expect(commandForEvent({ key: "K", ctrlKey: true })).toBe(
      "session.toggle_mic_mute",
    );
    expect(commandForEvent({ key: "b", ctrlKey: true })).toBe("view.toggle_sidebar");
  });

  it("leaves plain and modified keys to the browser", () => {
    expect(commandForEvent({ key: "p" })).toBeNull();
    expect(commandForEvent({ key: "p", ctrlKey: true, shiftKey: true })).toBeNull();
    expect(commandForEvent({ key: "c", ctrlKey: true })).toBeNull();
    expect(commandForEvent({ key: "q", ctrlKey: true })).toBeNull();
  });

  it("sends on Enter and newlines on Shift+Enter", () => {
    expect(commandForEvent({ key: "Enter" })).toBe("prompt.send");
    expect(commandForEvent({ key: "Enter", shiftKey: true })).toBeNull();
    expect(commandForEvent({ key: "Escape" })).toBe("prompt.clear");
  });

  it("documents every binding it can fire, and fires nothing undocumented", () => {
    const documented = [...SHORTCUTS_PROMPT, ...SHORTCUTS_SESSION]
      .map((shortcut) => shortcut.id)
      .filter((id): id is string => id !== undefined);
    const fired = [
      commandForEvent({ key: "Enter" }),
      commandForEvent({ key: "Escape" }),
      ...SHORTCUTS_SESSION.map((shortcut) =>
        commandForEvent({ key: shortcut.keys.slice(1), ctrlKey: true }),
      ),
    ];
    expect(new Set(fired)).toEqual(new Set(documented));
  });

  it("keeps Quit out of the client — the TUI owns process lifecycle (#96)", () => {
    const labels = [...SHORTCUTS_PROMPT, ...SHORTCUTS_SESSION].map(
      (shortcut) => shortcut.label,
    );
    expect(labels).not.toContain("Quit");
    // Nothing to bind to: the dispatch allowlist refuses the action outright.
    expect(isAllowedAction("session.quit")).toBe(false);
  });
});
