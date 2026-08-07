import { describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
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

function cov(
  percent: number,
  funcPercent = percent,
  found = 100,
  funcFound = 10,
): FileCoverage {
  const hit = Math.round((percent / 100) * found);
  const funcHit = Math.round((funcPercent / 100) * funcFound);
  return {
    hit,
    found,
    percent: (100 * hit) / found,
    funcHit,
    funcFound,
    funcPercent: funcFound === 0 ? 0 : (100 * funcHit) / funcFound,
  };
}

const baseConfig: FloorsConfig = {
  new_file_floor: 60,
  new_file_func_floor: 60,
  floors: { "src/client.ts": 90 },
  func_floors: { "src/client.ts": 80 },
  unmeasured_entrypoints: {
    "src/main.ts": "Electron process entry; unit tests must not launch a window (#97).",
  },
};

describe("parseLcov", () => {
  it("computes per-file line and function percents", () => {
    const text = [
      "TN:",
      "SF:src/client.ts",
      "FNF:4",
      "FNH:2",
      "DA:1,1",
      "DA:2,0",
      "DA:3,4",
      "end_of_record",
      "SF:src/ipc.ts",
      "FNF:1",
      "FNH:1",
      "DA:1,1",
      "end_of_record",
      "",
    ].join("\n");
    const measured = parseLcov(text);
    expect(measured["src/client.ts"]?.percent).toBeCloseTo(66.666, 2);
    expect(measured["src/client.ts"]?.funcPercent).toBe(50);
    expect(measured["src/ipc.ts"]?.percent).toBe(100);
    expect(measured["src/ipc.ts"]?.funcPercent).toBe(100);
  });

  it("treats a zero-line SF record as 0% not 100%", () => {
    const text = ["SF:src/empty.ts", "FNF:0", "FNH:0", "end_of_record", ""].join("\n");
    const measured = parseLcov(text);
    expect(measured["src/empty.ts"]?.percent).toBe(0);
    expect(measured["src/empty.ts"]?.funcPercent).toBe(0);
  });
});

describe("checkCoverage", () => {
  it("rejects a file below its recorded line floor", () => {
    const { failures } = checkCoverage({ "src/client.ts": cov(70, 90) }, baseConfig, [
      "src/client.ts",
      "src/main.ts",
    ]);
    expect(
      failures.some((line) => line.includes("src/client.ts") && line.includes("lines")),
    ).toBe(true);
  });

  it("rejects a file below its recorded func floor", () => {
    const { failures } = checkCoverage({ "src/client.ts": cov(95, 50) }, baseConfig, [
      "src/client.ts",
      "src/main.ts",
    ]);
    expect(
      failures.some((line) => line.includes("src/client.ts") && line.includes("funcs")),
    ).toBe(true);
  });

  it("rejects a new src file below NEW_FILE_FLOOR", () => {
    const { failures } = checkCoverage(
      { "src/brand_new.ts": cov(5) },
      { ...baseConfig, floors: {}, func_floors: {} },
      ["src/brand_new.ts", "src/main.ts"],
    );
    expect(failures.some((line) => line.includes("src/brand_new.ts"))).toBe(true);
  });

  it("rejects a floor for a file that vanished from governed dirs", () => {
    const { failures } = checkCoverage(
      { "src/client.ts": cov(95) },
      {
        ...baseConfig,
        floors: { "src/deleted.ts": 50, "src/client.ts": 90 },
        func_floors: { "src/client.ts": 80 },
      },
      ["src/client.ts", "src/main.ts"],
    );
    expect(failures.some((line) => line.includes("src/deleted.ts"))).toBe(true);
  });

  it("treats an unimported non-entrypoint as 0% (cannot escape measurement)", () => {
    const { failures } = checkCoverage({}, baseConfig, [
      "src/client.ts",
      "src/main.ts",
    ]);
    expect(
      failures.some((line) => line.includes("src/client.ts") && line.includes("0.0%")),
    ).toBe(true);
  });

  it("skips unmeasured_entrypoints even at 0%", () => {
    const { failures } = checkCoverage({}, baseConfig, [
      "src/client.ts",
      "src/main.ts",
    ]);
    expect(failures.some((line) => line.startsWith("src/main.ts:"))).toBe(false);
    expect(failures.some((line) => line.startsWith("src/client.ts:"))).toBe(true);
  });

  it("rejects an unmeasured_entrypoint without a reason", () => {
    const { failures } = checkCoverage(
      { "src/client.ts": cov(95) },
      {
        ...baseConfig,
        unmeasured_entrypoints: { "src/main.ts": "   " },
      },
      ["src/client.ts", "src/main.ts"],
    );
    expect(failures.some((line) => line.includes("non-empty reason"))).toBe(true);
  });

  it("notes a full point of headroom so the ratchet gets turned", () => {
    const { improvements, failures } = checkCoverage(
      { "src/client.ts": cov(95, 90) },
      baseConfig,
      ["src/client.ts", "src/main.ts"],
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
    mkdirSync(join(dir, "src"));
    mkdirSync(join(dir, "tools"));
    const floors = join(dir, "floors.json");
    writeFileSync(
      floors,
      JSON.stringify({
        new_file_floor: 60,
        new_file_func_floor: 60,
        floors: {},
        func_floors: {},
        unmeasured_entrypoints: {},
      }),
    );
    expect(main(floors, join(dir, "missing.lcov"), dir)).toBe(1);
  });

  it("passes a clean measured tree against recorded floors", () => {
    const dir = mkdtempSync(join(tmpdir(), "electron-cov-"));
    mkdirSync(join(dir, "src"));
    mkdirSync(join(dir, "tools"));
    writeFileSync(join(dir, "src", "client.ts"), "export {}\n");
    writeFileSync(join(dir, "tools", "coverage_gate.ts"), "export {}\n");
    const floors = join(dir, "floors.json");
    writeFileSync(
      floors,
      JSON.stringify({
        new_file_floor: 60,
        new_file_func_floor: 60,
        floors: { "src/client.ts": 90, "tools/coverage_gate.ts": 70 },
        func_floors: { "src/client.ts": 80, "tools/coverage_gate.ts": 50 },
        unmeasured_entrypoints: {},
      }),
    );
    const lcov = join(dir, "lcov.info");
    writeFileSync(
      lcov,
      [
        "SF:src/client.ts",
        "FNF:10",
        "FNH:9",
        "DA:1,1",
        "DA:2,1",
        "end_of_record",
        "SF:tools/coverage_gate.ts",
        "FNF:4",
        "FNH:3",
        "DA:1,1",
        "DA:2,1",
        "end_of_record",
        "",
      ].join("\n"),
    );
    expect(main(floors, lcov, dir)).toBe(0);
  });

  it("loadFloors rejects a malformed config", () => {
    const dir = mkdtempSync(join(tmpdir(), "electron-cov-"));
    const floors = join(dir, "floors.json");
    writeFileSync(floors, JSON.stringify({ floors: {} }));
    expect(() => loadFloors(floors)).toThrow("invalid floors config");
  });
});
