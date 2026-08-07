/**
 * Enforce per-file line and function coverage floors under electron/ that
 * only ratchet up.
 *
 * Bun's lcov only lists files the suite loaded. This gate also walks every
 * governed TypeScript path so an unimported module cannot escape with
 * "not measured". Process entry wiring that must not launch Electron in CI
 * is listed in coverage_floors.json under unmeasured_entrypoints with a reason.
 *
 * Function floors matter here more than in Python: Bun 1.3.9 lcov has no
 * BRDA/branch records, so FNF/FNH is the only sub-line signal that catches
 * registration-only hits (handle(...) call runs; arrow body never does).
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export type FloorsConfig = {
  new_file_floor: number;
  new_file_func_floor: number;
  /** Paths relative to electron/ (e.g. src/client.ts, tools/coverage_gate.ts). */
  floors: Record<string, number>;
  func_floors: Record<string, number>;
  /** Paths relative to electron/ excluded from measurement (process entry). */
  unmeasured_entrypoints: Record<string, string>;
};

export type FileCoverage = {
  hit: number;
  found: number;
  percent: number;
  funcHit: number;
  funcFound: number;
  funcPercent: number;
};

export type CheckResult = {
  failures: string[];
  improvements: string[];
};

/** Make targets and package scripts run with cwd = electron/. */
export function defaultRoot(): string {
  return process.cwd();
}

export function floorsPath(root: string = defaultRoot()): string {
  return join(root, "coverage_floors.json");
}

export function lcovPath(root: string = defaultRoot()): string {
  return join(root, "coverage", "lcov.info");
}

/** Directories whose .ts files must meet floors (relative to electron/). */
export const GOVERNED_DIRS = ["src", "tools"] as const;

export function loadFloors(path: string = floorsPath()): FloorsConfig {
  const raw = JSON.parse(readFileSync(path, "utf8")) as FloorsConfig;
  if (
    typeof raw.new_file_floor !== "number" ||
    typeof raw.new_file_func_floor !== "number" ||
    typeof raw.floors !== "object" ||
    raw.floors === null ||
    typeof raw.func_floors !== "object" ||
    raw.func_floors === null ||
    typeof raw.unmeasured_entrypoints !== "object" ||
    raw.unmeasured_entrypoints === null
  ) {
    throw new Error(`invalid floors config: ${path}`);
  }
  return raw;
}

/** Parse lcov.info into path → line + function coverage (SF: record paths). */
export function parseLcov(text: string): Record<string, FileCoverage> {
  const measured: Record<string, FileCoverage> = {};
  let current: string | null = null;
  let hit = 0;
  let found = 0;
  let funcHit = 0;
  let funcFound = 0;

  const flush = () => {
    if (current === null) {
      return;
    }
    measured[current] = {
      hit,
      found,
      // Empty SF record: treat as 0 so it cannot silently certify coverage.
      percent: found === 0 ? 0 : (100 * hit) / found,
      funcHit,
      funcFound,
      funcPercent: funcFound === 0 ? 0 : (100 * funcHit) / funcFound,
    };
    current = null;
    hit = 0;
    found = 0;
    funcHit = 0;
    funcFound = 0;
  };

  for (const line of text.split("\n")) {
    if (line.startsWith("SF:")) {
      flush();
      current = line.slice(3);
    } else if (line.startsWith("DA:") && current !== null) {
      const rest = line.slice(3);
      const comma = rest.indexOf(",");
      const hits = Number(rest.slice(comma + 1));
      found += 1;
      if (hits > 0) {
        hit += 1;
      }
    } else if (line.startsWith("FNF:") && current !== null) {
      funcFound = Number(line.slice(4));
    } else if (line.startsWith("FNH:") && current !== null) {
      funcHit = Number(line.slice(4));
    } else if (line === "end_of_record") {
      flush();
    }
  }
  flush();
  return measured;
}

export function listGovernedFiles(root: string = defaultRoot()): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) {
        walk(full);
      } else if (name.endsWith(".ts")) {
        out.push(relative(root, full).split("\\").join("/"));
      }
    }
  };
  for (const dir of GOVERNED_DIRS) {
    const abs = join(root, dir);
    if (statSync(abs).isDirectory()) {
      walk(abs);
    }
  }
  return out.sort();
}

function checkMetric(
  rel: string,
  label: string,
  actual: number,
  floor: number | undefined,
  newFloor: number,
  failures: string[],
  improvements: string[],
): void {
  if (floor === undefined) {
    if (actual < newFloor) {
      failures.push(
        `${rel}: new file at ${actual.toFixed(1)}% ${label} must reach ` +
          `${newFloor.toFixed(0)}% or record an explicit floor with a reason.`,
      );
    }
    return;
  }
  if (actual < floor) {
    failures.push(
      `${rel}: ${actual.toFixed(1)}% ${label} is below its floor of ${floor.toFixed(1)}%`,
    );
  } else if (Math.floor(actual) >= floor + 1) {
    improvements.push(
      `${rel}: ${actual.toFixed(1)}% ${label} — raise its floor to ${Math.floor(actual)}`,
    );
  }
}

/**
 * Compare measured coverage to floors. `governed` paths are relative to
 * electron/ (`src/client.ts`). `measured` keys are as in lcov (same shape).
 */
export function checkCoverage(
  measured: Record<string, FileCoverage>,
  config: FloorsConfig,
  governed: string[],
): CheckResult {
  const failures: string[] = [];
  const improvements: string[] = [];
  const excluded = new Set(Object.keys(config.unmeasured_entrypoints));
  const governedSet = new Set(governed);

  for (const [path, reason] of Object.entries(config.unmeasured_entrypoints)) {
    if (!reason || !String(reason).trim()) {
      failures.push(`${path}: unmeasured_entrypoints entry needs a non-empty reason`);
    }
    if (!governedSet.has(path)) {
      failures.push(
        `${path}: listed as unmeasured_entrypoint but is not under a governed dir`,
      );
    }
  }

  for (const rel of governed) {
    if (excluded.has(rel)) {
      continue;
    }
    const file = measured[rel];
    const lineActual = file?.percent ?? 0;
    const funcActual = file?.funcPercent ?? 0;
    checkMetric(
      rel,
      "lines",
      lineActual,
      config.floors[rel],
      config.new_file_floor,
      failures,
      improvements,
    );
    checkMetric(
      rel,
      "funcs",
      funcActual,
      config.func_floors[rel],
      config.new_file_func_floor,
      failures,
      improvements,
    );
  }

  for (const path of Object.keys(config.floors).sort()) {
    if (!governedSet.has(path) && !excluded.has(path)) {
      failures.push(
        `${path}: has a recorded line floor but is not governed. Remove its floors entry if the file is gone.`,
      );
    }
  }
  for (const path of Object.keys(config.func_floors).sort()) {
    if (!governedSet.has(path) && !excluded.has(path)) {
      failures.push(
        `${path}: has a recorded func floor but is not governed. Remove its func_floors entry if the file is gone.`,
      );
    }
  }

  return { failures, improvements };
}

export function main(
  floorsFile: string = floorsPath(),
  lcovFile: string = lcovPath(),
  root: string = defaultRoot(),
): number {
  let config: FloorsConfig;
  try {
    config = loadFloors(floorsFile);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`error: ${message}`);
    return 1;
  }

  let lcovText: string;
  try {
    lcovText = readFileSync(lcovFile, "utf8");
  } catch {
    console.error(
      `error: ${lcovFile} is missing; run \`bun test --coverage --coverage-reporter=lcov\` first.`,
    );
    return 1;
  }

  const measured = parseLcov(lcovText);
  const governed = listGovernedFiles(root);
  const { failures, improvements } = checkCoverage(measured, config, governed);

  for (const failure of failures) {
    console.error(`error: ${failure}`);
  }
  for (const note of improvements) {
    console.log(`note: ${note}`);
  }

  if (failures.length > 0) {
    console.error(`\n${failures.length} per-file Electron coverage failure(s).`);
    return 1;
  }

  const floored = Object.keys(config.floors).length;
  console.log(
    `electron per-file coverage: ${floored} files at or above their line/func floors ` +
      `(${Object.keys(config.unmeasured_entrypoints).length} entrypoint(s) exempt).`,
  );
  return 0;
}

const invoked = process.argv[1] ?? "";
if (
  invoked.endsWith(`${join("tools", "coverage_gate.ts")}`) ||
  invoked.endsWith(`${join("tools", "coverage_gate.js")}`) ||
  invoked.endsWith("coverage_gate.ts") ||
  invoked.endsWith("coverage_gate.js")
) {
  process.exit(main());
}
