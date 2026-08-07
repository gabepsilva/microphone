import { describe, expect, it } from "bun:test";

import { voiceOptionsIncluding } from "../renderer/speech_catalog.js";
import {
  selectedVoiceId,
  syncVoicePicker,
  voiceChangePayload,
  voiceEffectiveText,
  voicePickerActive,
} from "../renderer/voice_picker.js";

import { makeDocument, type FakeNode } from "./fake_dom.js";

function makeSelect(document: ReturnType<typeof makeDocument>["document"]): FakeNode & {
  disabled: boolean;
  value: string;
  options: FakeNode[];
} {
  const select = document.createElement("select") as FakeNode & {
    disabled: boolean;
    value: string;
    options: FakeNode[];
  };
  select.disabled = false;
  select.value = "";
  Object.defineProperty(select, "options", {
    get() {
      return this.children;
    },
  });
  const originalAppend = select.appendChild.bind(select);
  select.appendChild = (child) => {
    const node = child as FakeNode & { value?: string };
    if (typeof node.value !== "string") {
      node.value = "";
    }
    return originalAppend(child);
  };
  return select;
}

describe("selectedVoiceId", () => {
  it("prefers desired, then effective, then piper_voice", () => {
    expect(
      selectedVoiceId({
        tts_voice: { desired: "en_US-amy-medium", effective: "en_US-lessac-medium" },
        piper_voice: "en_US-ryan-high",
      }),
    ).toBe("en_US-amy-medium");
    expect(
      selectedVoiceId({
        tts_voice: { desired: null, effective: "en_US-lessac-medium" },
        piper_voice: "en_US-ryan-high",
      }),
    ).toBe("en_US-lessac-medium");
    expect(selectedVoiceId({ piper_voice: "en_US-ryan-high" })).toBe("en_US-ryan-high");
    expect(selectedVoiceId({})).toBe("");
  });
});

describe("voiceEffectiveText", () => {
  it("only names effective when it diverges from desired", () => {
    expect(
      voiceEffectiveText({
        tts_voice: {
          desired: "en_US-amy-medium",
          effective: "en_US-lessac-medium",
        },
      }),
    ).toBe("effective: en_US-lessac-medium");
    expect(
      voiceEffectiveText({
        tts_voice: {
          desired: "en_US-amy-medium",
          effective: "en_US-amy-medium",
        },
      }),
    ).toBe("");
    expect(voiceEffectiveText({ tts_voice: { desired: "x", effective: null } })).toBe(
      "",
    );
  });
});

describe("voicePickerActive", () => {
  it("is on for Piper and off for Edge, including Selection shape", () => {
    expect(voicePickerActive({ tts_provider: "piper" })).toBe(true);
    expect(voicePickerActive({})).toBe(true);
    expect(voicePickerActive({ tts_provider: "edge" })).toBe(false);
    expect(
      voicePickerActive({
        tts_provider: { desired: "edge", effective: "piper" },
      }),
    ).toBe(false);
    expect(
      voicePickerActive({
        tts_provider: { desired: "piper", effective: "piper" },
      }),
    ).toBe(true);
  });
});

describe("syncVoicePicker", () => {
  it("hides and disables the control on Edge", () => {
    const { document } = makeDocument();
    const field = document.createElement("div") as FakeNode;
    const select = makeSelect(document);
    syncVoicePicker(
      field,
      select as never,
      { tts_provider: "edge", piper_voice: "en_US-lessac-medium" },
      [{ id: "en_US-lessac-medium", label: "Lessac medium", downloaded: true }],
      (tag) => document.createElement(tag) as never,
      voiceOptionsIncluding,
    );
    expect(field.hidden).toBe(true);
    expect(select.disabled).toBe(true);
    expect(select.children).toEqual([]);
  });

  it("fills Piper options and selects desired", () => {
    const { document } = makeDocument();
    const field = document.createElement("div") as FakeNode;
    const select = makeSelect(document);
    syncVoicePicker(
      field,
      select as never,
      {
        tts_provider: "piper",
        tts_voice: {
          desired: "en_US-amy-medium",
          effective: "en_US-lessac-medium",
        },
      },
      [
        { id: "en_US-lessac-medium", label: "Lessac medium", downloaded: true },
        { id: "en_US-amy-medium", label: "Amy medium", downloaded: false },
      ],
      (tag) => document.createElement(tag) as never,
      voiceOptionsIncluding,
    );
    expect(field.hidden).toBe(false);
    expect(select.disabled).toBe(false);
    expect(select.children.map((child) => (child as { value?: string }).value)).toEqual(
      ["en_US-lessac-medium", "en_US-amy-medium"],
    );
    expect(select.children.map((child) => child.textContent)).toEqual([
      "Lessac medium",
      "Amy medium (download)",
    ]);
    expect(select.value).toBe("en_US-amy-medium");
  });
});

describe("voiceChangePayload", () => {
  it("skips while applyState is redrawing the select", () => {
    expect(voiceChangePayload(true, "en_US-amy-medium")).toBeNull();
    expect(voiceChangePayload(false, "en_US-amy-medium")).toEqual({
      voice: "en_US-amy-medium",
    });
  });
});
