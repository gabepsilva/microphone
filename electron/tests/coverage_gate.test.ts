import { describe, expect, it } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  checkCoverage,
  loadFloors,
  main,
  parseLcov,
  type FileCoverage,
  type FloorsConfig,
} from "../tools/coverage_gate";

function cov(percent: number, found = 100): FileCoverage {
  const hit = Math.round((percent / 100) * found);
  return { hit, found, percent: (100 * hit) / found };
}

const baseConfig: FloorsConfig = {
  new_file_floor: 60,
  floors: { "client.ts": 90 },
  unmeasured_entrypoints: {
    "main.ts": "Electron process entry; unit tests must not launch a window (#97).",
  },
};

describe("parseLcov", () => {
  it("computes per-file line percents from DA records", () => {
    const text = [
      "TN:",
      "SF:src/client.ts",
      "DA:1,1",
      "DA:2,0",
      "DA:3,4",
      "end_of_record",
      "SF:src/ipc.ts",
      "DA:1,1",
      "end_of_record",
      "",
    ].join("\n");
    const measured = parseLcov(text);
    expect(measured["src/client.ts"]?.percent).toBeCloseTo(66.666, 2);
    expect(measured["src/ipc.ts"]?.percent).toBe(100);
  });
});

describe("checkCoverage", () => {
  it("rejects a file below its recorded floor", () => {
    const { failures } = checkCoverage({ "src/client.ts": cov(70) }, baseConfig, [
      "client.ts",
      "main.ts",
    ]);
    expect(
      failures.some((line) => line.includes("client.ts") && line.includes("70.0%")),
    ).toBe(true);
  });

  it("rejects a new src file below NEW_FILE_FLOOR", () => {
    const { failures } = checkCoverage(
      { "src/brand_new.ts": cov(5) },
      { ...baseConfig, floors: {} },
      ["brand_new.ts", "main.ts"],
    );
    expect(failures.some((line) => line.includes("brand_new.ts"))).toBe(true);
  });

  it("rejects a floor for a file that vanished from src/", () => {
    const { failures } = checkCoverage(
      { "src/client.ts": cov(95) },
      { ...baseConfig, floors: { "deleted.ts": 50, "client.ts": 90 } },
      ["client.ts", "main.ts"],
    );
    expect(failures.some((line) => line.includes("deleted.ts"))).toBe(true);
  });

  it("treats an unimported non-entrypoint as 0% (cannot escape measurement)", () => {
    const { failures } = checkCoverage(
      {},
      { ...baseConfig, floors: { "client.ts": 90 } },
      ["client.ts", "main.ts"],
    );
    expect(
      failures.some((line) => line.includes("client.ts") && line.includes("0.0%")),
    ).toBe(true);
  });

  it("skips unmeasured_entrypoints even at 0%", () => {
    const { failures } = checkCoverage({}, baseConfig, ["client.ts", "main.ts"]);
    // client still fails (0%); main must not appear.
    expect(failures.some((line) => line.startsWith("main.ts:"))).toBe(false);
    expect(failures.some((line) => line.startsWith("client.ts:"))).toBe(true);
  });

  it("rejects an unmeasured_entrypoint without a reason", () => {
    const { failures } = checkCoverage(
      { "src/client.ts": cov(95) },
      {
        ...baseConfig,
        unmeasured_entrypoints: { "main.ts": "   " },
      },
      ["client.ts", "main.ts"],
    );
    expect(failures.some((line) => line.includes("non-empty reason"))).toBe(true);
  });

  it("notes a full point of headroom so the ratchet gets turned", () => {
    const { improvements, failures } = checkCoverage(
      { "src/client.ts": cov(95) },
      baseConfig,
      ["client.ts", "main.ts"],
    );
    expect(failures).toEqual([]);
    expect(improvements.some((line) => line.includes("raise its floor to 95"))).toBe(
      true,
    );
  });
});

describe("coverage_gate main", () => {
  it("fails when lcov is missing", () => {
    const dir = mkdtempSync(join(tmpdir(), "electron-cov-"));
    const floors = join(dir, "floors.json");
    writeFileSync(
      floors,
      JSON.stringify({
        new_file_floor: 60,
        floors: {},
        unmeasured_entrypoints: {},
      }),
    );
    expect(main(floors, join(dir, "missing.lcov"), dir)).toBe(1);
  });

  it("loadFloors rejects a malformed config", () => {
    const dir = mkdtempSync(join(tmpdir(), "electron-cov-"));
    const floors = join(dir, "floors.json");
    writeFileSync(floors, JSON.stringify({ floors: {} }));
    expect(() => loadFloors(floors)).toThrow("invalid floors config");
  });
});
