import { describe, expect, it } from "bun:test";

import { stageFiles, refusalForFile } from "../renderer/staging.js";

const MIB = 1024 * 1024;

/**
 * A Real File with a PNG magic prefix — valid enough for the sniff, never
 * parsed by the sandbox.
 */
function pngFile(name: string, size = 64): File {
  const buffer = new Uint8Array(size);
  const prefix = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  prefix.forEach((byte, index) => {
    buffer[index] = byte;
  });
  return new File([buffer], name, { type: "image/png" });
}

function textFile(name: string, size = 64) {
  return new File([new ArrayBuffer(size)], name, { type: "text/plain" });
}

describe("preflight for a single file", () => {
  it("passes a PNG under the cap", async () => {
    expect(await refusalForFile(pngFile("shot.png", 1024))).toBeNull();
  });

  it("refuses a text file with the honest two-concept copy", async () => {
    const refusal = await refusalForFile(textFile("notes.txt"));
    expect(refusal).toContain("notes.txt");
    expect(refusal).toContain("not an image");
  });

  it("refuses an oversize file by size alone, with the limit in the copy", async () => {
    const refusal = await refusalForFile(pngFile("big.png", 21 * MIB));
    expect(refusal).toContain("big.png");
    expect(refusal).toContain("20 MiB");
  });
});

describe("stageFiles: one loop for paste, drop, and picker (D2)", () => {
  it("uploads each acceptable file exactly once, in order", async () => {
    const uploads: Uint8Array[] = [];
    const upload = async (bytes: Uint8Array) => {
      const id = `id-${uploads.length + 1}`;
      uploads.push(new Uint8Array(bytes));
      return id;
    };
    const results = await stageFiles([pngFile("a.png"), pngFile("b.png")], upload);
    expect(results).toEqual([
      { file: results[0]!.file, id: "id-1", refused: null },
      { file: results[1]!.file, id: "id-2", refused: null },
    ]);
    expect(uploads.map((bytes) => bytes.length)).toEqual([64, 64]);
  });

  it("never calls upload for a file the staging loop refuses", async () => {
    const uploads: Uint8Array[] = [];
    const results = await stageFiles(
      [
        pngFile("ok.png", 128),
        textFile("notes.txt"),
        pngFile("too-big.png", 22.5 * MIB),
      ],
      async (bytes: Uint8Array): Promise<string> => {
        uploads.push(bytes);
        return "id";
      },
    );
    expect(results[0]).toEqual({ file: results[0]!.file, id: "id", refused: null });
    expect(results[1]!.refused).toContain("notes.txt");
    expect(results[2]!.refused).not.toBeNull();
    // One upload, for the one preflight that passed — the band never reaches
    // the wire (transport.py:284 would hang the socket instead).
    expect(uploads).toHaveLength(1);
  });

  it("surfaces a declined upload as a null id", async () => {
    const results = await stageFiles([pngFile("ok.png")], async () => null);
    expect(results).toHaveLength(1);
    expect(results[0]!.id).toBeNull();
    expect(results[0]!.refused).toBeNull();
  });
});
