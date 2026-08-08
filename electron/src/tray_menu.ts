import type { MenuItemConstructorOptions } from "electron";

import type { AppState } from "./state";

/** Mute (and optional read-aloud) callbacks injected by the tray host. */
export type TrayMenuHandlers = {
  onToggleMicrophoneMute: () => void;
  onToggleAudioStreamMute: () => void;
  /**
   * When set, the menu includes a separator and **Read aloud** (#128b).
   * Absent in 128a so the item is not present-and-no-op.
   */
  onReadAloud?: () => void;
};

/**
 * Pure tray menu template from live mute state.
 *
 * Labels flip with ``microphone_muted`` / ``audio_stream_muted`` (same fields
 * the sidebar already consumes). ``main.ts`` only hosts ``Tray`` + rebuild.
 */
export function buildTrayMenu(
  state: Pick<AppState, "microphone_muted" | "audio_stream_muted">,
  handlers: TrayMenuHandlers,
): MenuItemConstructorOptions[] {
  const items: MenuItemConstructorOptions[] = [
    {
      label: state.microphone_muted ? "Unmute microphone" : "Mute microphone",
      click: () => {
        handlers.onToggleMicrophoneMute();
      },
    },
    {
      label: state.audio_stream_muted ? "Unmute Audio Stream" : "Mute Audio Stream",
      click: () => {
        handlers.onToggleAudioStreamMute();
      },
    },
  ];
  if (handlers.onReadAloud !== undefined) {
    const readAloud = handlers.onReadAloud;
    items.push(
      { type: "separator" },
      {
        label: "Read aloud",
        click: () => {
          readAloud();
        },
      },
    );
  }
  return items;
}
