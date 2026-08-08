import { describe, expect, it, mock } from "bun:test";

import { buildTrayMenu, trayMenuKey } from "../src/tray_menu";

describe("buildTrayMenu", () => {
  it("labels mute items unmuted when both streams are live", () => {
    const items = buildTrayMenu(
      { microphone_muted: false, audio_stream_muted: false },
      {
        onToggleMicrophoneMute: () => undefined,
        onToggleAudioStreamMute: () => undefined,
      },
    );
    expect(items.map((item) => item.label ?? item.type)).toEqual([
      "Mute microphone",
      "Mute Audio Stream",
    ]);
  });

  it("flips labels when muted and omits Read aloud without a handler", () => {
    const items = buildTrayMenu(
      { microphone_muted: true, audio_stream_muted: true },
      {
        onToggleMicrophoneMute: () => undefined,
        onToggleAudioStreamMute: () => undefined,
      },
    );
    expect(items.map((item) => item.label ?? item.type)).toEqual([
      "Unmute microphone",
      "Unmute Audio Stream",
    ]);
    expect(items.some((item) => item.label === "Read aloud")).toBe(false);
    expect(items.some((item) => item.type === "separator")).toBe(false);
  });

  it("includes separator and Read aloud only when onReadAloud is set", () => {
    const items = buildTrayMenu(
      { microphone_muted: false, audio_stream_muted: true },
      {
        onToggleMicrophoneMute: () => undefined,
        onToggleAudioStreamMute: () => undefined,
        onReadAloud: () => undefined,
      },
    );
    expect(items.map((item) => item.label ?? item.type)).toEqual([
      "Mute microphone",
      "Unmute Audio Stream",
      "separator",
      "Read aloud",
    ]);
  });

  it("invokes the matching callback when a menu item is clicked", () => {
    const onToggleMicrophoneMute = mock(() => undefined);
    const onToggleAudioStreamMute = mock(() => undefined);
    const onReadAloud = mock(() => undefined);
    const items = buildTrayMenu(
      { microphone_muted: false, audio_stream_muted: false },
      { onToggleMicrophoneMute, onToggleAudioStreamMute, onReadAloud },
    );

    items[0]?.click?.(undefined as never, undefined as never, undefined as never);
    items[1]?.click?.(undefined as never, undefined as never, undefined as never);
    items[3]?.click?.(undefined as never, undefined as never, undefined as never);

    expect(onToggleMicrophoneMute).toHaveBeenCalledTimes(1);
    expect(onToggleAudioStreamMute).toHaveBeenCalledTimes(1);
    expect(onReadAloud).toHaveBeenCalledTimes(1);
  });
});

describe("trayMenuKey", () => {
  it("changes only when mute flags or read-aloud presence change", () => {
    const live = { microphone_muted: false, audio_stream_muted: false };
    const micMuted = { microphone_muted: true, audio_stream_muted: false };
    const muteOnly = {
      onToggleMicrophoneMute: () => undefined,
      onToggleAudioStreamMute: () => undefined,
    };
    const withReadAloud = {
      ...muteOnly,
      onReadAloud: () => undefined,
    };
    expect(trayMenuKey(live, muteOnly)).toBe(trayMenuKey({ ...live }, muteOnly));
    expect(trayMenuKey(live, muteOnly)).not.toBe(trayMenuKey(micMuted, muteOnly));
    expect(trayMenuKey(live, muteOnly)).not.toBe(trayMenuKey(live, withReadAloud));
  });
});
