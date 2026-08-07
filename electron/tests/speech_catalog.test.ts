import { describe, expect, it } from "bun:test";

import {
  parseSpeechCatalog,
  voiceOptionsIncluding,
} from "../renderer/speech_catalog.js";

describe("speech.catalog parsing", () => {
  it("keeps well-formed rows and drops the rest", () => {
    expect(
      parseSpeechCatalog({
        voices: [
          { id: "en_US-lessac-medium", label: "Lessac medium", downloaded: true },
          { id: "", label: "bad" },
          { label: "no-id", downloaded: false },
          null,
          { id: "en_US-amy-medium", label: "", downloaded: false },
        ],
      }),
    ).toEqual([
      {
        id: "en_US-lessac-medium",
        label: "Lessac medium",
        downloaded: true,
      },
      {
        id: "en_US-amy-medium",
        label: "en_US-amy-medium",
        downloaded: false,
      },
    ]);
  });

  it("accepts a bare array as well as a {voices} object", () => {
    expect(parseSpeechCatalog([{ id: "en_US-joe-medium", downloaded: true }])).toEqual([
      { id: "en_US-joe-medium", label: "en_US-joe-medium", downloaded: true },
    ]);
    expect(parseSpeechCatalog(null)).toEqual([]);
  });
});

describe("voiceOptionsIncluding", () => {
  it("marks undownloaded voices and keeps the running id selectable", () => {
    expect(
      voiceOptionsIncluding(
        [
          { id: "en_US-lessac-medium", label: "Lessac medium", downloaded: true },
          { id: "en_US-amy-medium", label: "Amy medium", downloaded: false },
        ],
        "en_US-ryan-high",
      ),
    ).toEqual([
      { value: "en_US-ryan-high", label: "en_US-ryan-high" },
      { value: "en_US-lessac-medium", label: "Lessac medium" },
      { value: "en_US-amy-medium", label: "Amy medium (download)" },
    ]);
  });
});
