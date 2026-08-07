import { describe, expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { hasRelativeRequire } from "./preload_bundle";

const root = join(__dirname, "..");
const builtPreload = join(root, "dist", "preload.js");
const planted = join(root, "tests", "fixtures", "preload_relative_require.js");

describe("sandboxed preload bundle", () => {
  it("build emits a preload with no relative requires", () => {
    const result = Bun.spawnSync(["bun", "run", "build"], {
      cwd: root,
      stdout: "pipe",
      stderr: "pipe",
    });
    expect(result.exitCode).toBe(0);
    const source = readFileSync(builtPreload, "utf8");
    expect(hasRelativeRequire(source)).toBe(false);
    expect(source).toContain("tagalong:dispatch");
    expect(source).toContain('require("electron")');
  });

  it("rejects a planted preload that relative-requires a local module", () => {
    const source = readFileSync(planted, "utf8");
    expect(hasRelativeRequire(source)).toBe(true);
  });
});
