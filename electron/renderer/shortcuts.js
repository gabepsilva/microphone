/**
 * Keyboard shortcuts for the Electron client.
 *
 * The TUI's empty screen doubles as its key reference (tui.SHORTCUTS_PROMPT /
 * SHORTCUTS_SESSION). This module is the same idea for the renderer: one
 * table that both the splash and the key handler read, so a binding cannot
 * exist without being documented or be documented without existing.
 *
 * Quit is deliberately absent — the TUI owns process lifecycle (#96).
 */

/** Response policy order, mirroring domain.RESPONSE_POLICIES insertion order. */
export const POLICY_ORDER = ["audio", "both", "voice", "quiet"];

/** The policy after *current*, wrapping; unknown values start the cycle. */
export function nextPolicy(current) {
  const index = POLICY_ORDER.indexOf(current);
  return POLICY_ORDER[(index + 1) % POLICY_ORDER.length];
}

/**
 * @typedef {object} Shortcut
 * @property {string} keys   Glyphs shown on the splash (e.g. "^P").
 * @property {string} label  What it does.
 * @property {string} [id]   Command id, when a key handler can fire it.
 */

/** Bindings that act on the prompt. */
export const SHORTCUTS_PROMPT = [
  { keys: "↵", label: "Send message", id: "prompt.send" },
  { keys: "⇧↵", label: "New line" },
  { keys: "^V", label: "Paste text or image" },
  { keys: "/", label: "Open command palette" },
  { keys: "↑↓", label: "Browse commands" },
  { keys: "⇥", label: "Complete command" },
  { keys: "⎋", label: "Dismiss overlay", id: "prompt.clear" },
];

/** Bindings that act on the running session. */
export const SHORTCUTS_SESSION = [
  { keys: "^P", label: "Cycle response policy", id: "session.cycle_policy" },
  { keys: "^K", label: "Mute microphone", id: "session.toggle_mic_mute" },
  { keys: "^T", label: "Toggle voice reply", id: "session.toggle_tts" },
  { keys: "^X", label: "Interrupt Taga", id: "session.interrupt" },
  { keys: "^D", label: "End voice turn", id: "session.end_turn" },
  { keys: "^B", label: "Toggle sidebar", id: "view.toggle_sidebar" },
  { keys: "^S", label: "Save transcript", id: "session.save_transcript" },
  { keys: "^N", label: "New session", id: "session.new" },
];

/** Command id for a key event, or null when the event is not a binding. */
export function commandForEvent(event) {
  if (event.key === "Escape") {
    return "prompt.clear";
  }
  if (event.key === "Enter") {
    return event.shiftKey ? null : "prompt.send";
  }
  if (!event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) {
    return null;
  }
  const found = SHORTCUTS_SESSION.find(
    (shortcut) => shortcut.keys === `^${String(event.key).toUpperCase()}`,
  );
  return found?.id ?? null;
}
