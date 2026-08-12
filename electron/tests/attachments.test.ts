import { describe, expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  DraftAttachments,
  IMAGE_TOKEN_RE,
  MAX_IMAGE_BYTES,
  base64FromBytes,
  imageToken,
  looksLikeImage,
  magicRefusal,
  oversizeRefusal,
  parseImageNumbers,
  parseImageTokens,
  renderTokenTextInto,
} from "../renderer/attachments.js";

import { flatText, makeDocument, type FakeNode } from "./fake_dom.js";

const MIB = 1024 * 1024;

function bytes(...values: number[]): Uint8Array {
  return new Uint8Array(values);
}

function pngBytes() {
  return bytes(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00);
}

function webpBytes() {
  return bytes(0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50);
}

describe("token parity corpus (JS vs Python)", () => {
  it("parses every corpus entry like tagalong/attachments.py does", () => {
    const fixture = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "token_parity.json"), "utf8"),
    );
    for (const { text, numbers } of fixture.corpus) {
      expect(parseImageNumbers(text)).toEqual(numbers);
    }
  });

  it("mirrors the Python token renderer", () => {
    expect(imageToken(1)).toBe("[Image #1]");
    expect(imageToken(12)).toBe("[Image #12]");
    expect(() => imageToken(0)).toThrow("1-based");
  });

  it("scans with the same regex as attachments.py:32", () => {
    expect(IMAGE_TOKEN_RE.source).toBe("\\[Image #(\\d+)\\]");
  });

  it("reports token spans for the chip view", () => {
    expect(parseImageTokens("see [Image #1]?")).toEqual([
      { number: 1, token: "[Image #1]", start: 4, end: 14 },
    ]);
    expect(parseImageTokens("plain")).toEqual([]);
  });
});

describe("looksLikeImage magic sniff", () => {
  it("accepts PNG, JPEG, GIF, and RIFF/WEBP prefixes", () => {
    expect(looksLikeImage(pngBytes())).toBe(true);
    expect(looksLikeImage(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe(true);
    expect(looksLikeImage(bytes(0x47, 0x49, 0x46, 0x38, 0x39, 0x61))).toBe(true);
    expect(looksLikeImage(webpBytes())).toBe(true);
  });

  it("rejects everything else, including truncated magic input", () => {
    expect(looksLikeImage(new Uint8Array(0))).toBe(false);
    expect(looksLikeImage(bytes(0x89, 0x50))).toBe(false);
    expect(looksLikeImage(bytes(0xff, 0xd8))).toBe(false);
    expect(looksLikeImage(bytes(0x47, 0x49, 0x46))).toBe(false);
    // RIFF without the WEBP4 mark at bytes 8..11.
    expect(looksLikeImage(bytes(0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0, 0, 0, 0))).toBe(
      false,
    );
    // RIFF too short to carry the mark.
    expect(looksLikeImage(bytes(0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0))).toBe(false);
    expect(looksLikeImage(bytes(1, 2, 3))).toBe(false);
  });
});

describe("preflight: size cap and the magic gate (D6)", () => {
  it("lets a file at exactly the 20 MiB cap through", () => {
    expect(oversizeRefusal({ name: "cap.png", size: MAX_IMAGE_BYTES })).toBeNull();
    expect(oversizeRefusal({ name: "cap.png", size: 0 })).toBeNull();
  });

  it("refuses anything past the cap with copy naming the limit", () => {
    const over = oversizeRefusal({ name: "big.png", size: MAX_IMAGE_BYTES + 1 });
    expect(over).toContain("big.png");
    expect(over).toContain("20 MiB");
  });

  it("refuses the whole 20–22.5 MiB band before anything touches the wire", () => {
    // The transport's MAX_FRAME is 30 MiB of wire text; base64 is 4/3, so a
    // file past ~22.5 MiB would make a frame over the cap and the socket
    // would hang up instead of refusing (test_transport.py:660-676). The
    // client-side cap turns every size in the band into a refusal.
    for (const size of [21 * MIB, 22 * MIB, 22.4 * MIB, 22.5 * MIB, 25 * MIB]) {
      expect(oversizeRefusal({ name: "band.png", size })).not.toBeNull();
    }
  });

  it("names an unnamed file for the refusal", () => {
    expect(oversizeRefusal({ size: MAX_IMAGE_BYTES + 1 })).toBe(
      "this file is 20.0 MiB — images are capped at 20 MiB",
    );
  });

  it("refuses a non-image with the two concept copy (Q6b)", () => {
    expect(magicRefusal("notes.txt", bytes(0x25, 0x50, 0x44, 0x46))).toBe(
      "notes.txt is not an image — only images can be attached right now",
    );
    expect(magicRefusal("note.png", pngBytes())).toBeNull();
  });
});

describe("base64FromBytes", () => {
  it("encodes arbitrary bytes via btoa, not the per-byte loop (F5)", () => {
    const input = new Uint8Array([0, 1, 2, 255, 254, 253, 128]);
    expect(base64FromBytes(input)).toBe(btoa(String.fromCharCode(...input)));
  });

  it("matches the native single-shot encode on a 5 MiB input", () => {
    // The size class #139 F5 measured at ~346 ms through the old loop.
    // Build the expected value with Buffer rather than a second fromCharCode
    // pass: under coverage the duplicated 5 MiB string walk exceeds bun's
    // default 5 s timeout and blocks an otherwise-green pre-push gate.
    const input = new Uint8Array(5 * MIB);
    for (let index = 0; index < input.length; index += 1) {
      input[index] = (index * 7 + 3) % 256;
    }
    expect(base64FromBytes(input)).toBe(Buffer.from(input).toString("base64"));
  }, 30_000);
});

describe("DraftAttachments", () => {
  it("numbers tokens 1-based and resolves by text order (D1)", () => {
    const draft = new DraftAttachments();
    expect(draft.add("img1")).toBe("[Image #1]");
    expect(draft.add("img2")).toBe("[Image #2]");
    expect(draft.resolve("see [Image #1] and [Image #2]")).toEqual(["img1", "img2"]);
  });

  it("resolves only tokens still in the text", () => {
    const draft = new DraftAttachments();
    draft.add("img1");
    draft.add("img2");
    expect(draft.resolve("[Image #2] then [Image #2]")).toEqual(["img2"]);
    expect(draft.resolve("nothing")).toEqual([]);
    // A hand-edited draft cannot invent ids beyond what was staged.
    expect(draft.resolve("[Image #99]")).toEqual([]);
  });

  it("clears with the draft", () => {
    const draft = new DraftAttachments();
    draft.add("img1");
    draft.clear();
    expect(draft.resolve("[Image #1]")).toEqual([]);
  });
});

describe("renderTokenTextInto", () => {
  it("draws chips and literal text runs, never markup", () => {
    const { document } = makeDocument();
    const host = document.createElement("div");
    renderTokenTextInto(document, host, "see [Image #1] twice [Image #1]");
    const kids = Array.from(host.children) as unknown as FakeNode[];
    expect(kids.map((child) => child.tag)).toEqual(["#text", "span", "#text", "span"]);
    expect(kids[0]?.textContent).toBe("see ");
    const chip = kids[1]!;
    expect(chip.className).toBe("image-chip");
    expect(chip.textContent).toBe("[Image #1]");
    expect(kids[2]?.textContent).toBe(" twice ");
  });

  it("renders plain text as a single literal run", () => {
    const { document } = makeDocument();
    const host = document.createElement("div");
    renderTokenTextInto(document, host, "plain");
    expect(flatText(host as unknown as FakeNode)).toBe("plain");
    expect((Array.from(host.children)[0] as unknown as FakeNode).tag).toBe("#text");
  });
});
