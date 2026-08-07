/**
 * Enforce per-file line-coverage floors under electron/src that only ratchet up.
 *
 * Bun's lcov only lists files the suite loaded. This gate also walks every
 * TypeScript file under src/ so an unimported module cannot escape with
 * "not measured". Process entry wiring that must not launch Electron in CI
 * is listed in coverage_floors.json under unmeasured_entrypoints with a reason.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export type FloorsConfig = {
  new_file_floor: number;
  floors: Record<string, number>;
  /** src-relative paths excluded from measurement (process entry / window). */
  unmeasured_entrypoints: Record<string, string>;
};

export type FileCoverage = {
  hit: number;
  found: number;
  percent: number;
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

export function srcRoot(root: string = defaultRoot()): string {
  return join(root, "src");
}

export function loadFloors(path: string = floorsPath()): FloorsConfig {
  const raw = JSON.parse(readFileSync(path, "utf8")) as FloorsConfig;
  if (
    typeof raw.new_file_floor !== "number" ||
    typeof raw.floors !== "object" ||
    raw.floors === null ||
    typeof raw.unmeasured_entrypoints !== "object" ||
    raw.unmeasured_entrypoints === null
  ) {
    throw new Error(`invalid floors config: ${path}`);
  }
  return raw;
}

/** Parse lcov.info into path → line coverage (paths as in the SF: records). */
export function parseLcov(text: string): Record<string, FileCoverage> {
  const measured: Record<string, FileCoverage> = {};
  let current: string | null = null;
  let hit = 0;
  let found = 0;

  const flush = () => {
    if (current === null) {
      return;
    }
    measured[current] = {
      hit,
      found,
      percent: found === 0 ? 100 : (100 * hit) / found,
    };
    current = null;
    hit = 0;
    found = 0;
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
    } else if (line === "end_of_record") {
      flush();
    }
  }
  flush();
  return measured;
}

export function listSrcFiles(root: string = srcRoot()): string[] {
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
  walk(root);
  return out.sort();
}

/**
 * Compare measured coverage to floors. `srcFiles` are paths relative to src/
 * (e.g. `client.ts`, `protocol/channels.ts`). `measured` keys are as in lcov
 * (`src/client.ts`).
 */
export function checkCoverage(
  measured: Record<string, FileCoverage>,
  config: FloorsConfig,
  srcFiles: string[],
): CheckResult {
  const failures: string[] = [];
  const improvements: string[] = [];
  const excluded = new Set(Object.keys(config.unmeasured_entrypoints));

  for (const [path, reason] of Object.entries(config.unmeasured_entrypoints)) {
    if (!reason || !String(reason).trim()) {
      failures.push(`${path}: unmeasured_entrypoints entry needs a non-empty reason`);
    }
    if (!srcFiles.includes(path)) {
      failures.push(`${path}: listed as unmeasured_entrypoint but is not under src/`);
    }
  }

  for (const rel of srcFiles) {
    if (excluded.has(rel)) {
      continue;
    }
    const lcovKey = `src/${rel}`;
    const file = measured[lcovKey];
    const actual = file?.percent ?? 0;
    const floor = config.floors[rel];

    if (floor === undefined) {
      if (actual < config.new_file_floor) {
        failures.push(
          `${rel}: new file at ${actual.toFixed(1)}% must reach ` +
            `${config.new_file_floor.toFixed(0)}% or record an explicit floor ` +
            `with a reason.`,
        );
      }
      continue;
    }

    if (actual < floor) {
      failures.push(
        `${rel}: ${actual.toFixed(1)}% is below its floor of ${floor.toFixed(1)}%`,
      );
    } else if (Math.floor(actual) >= floor + 1) {
      improvements.push(
        `${rel}: ${actual.toFixed(1)}% — raise its floor to ${Math.floor(actual)}`,
      );
    }
  }

  for (const path of Object.keys(config.floors).sort()) {
    if (!srcFiles.includes(path) && !excluded.has(path)) {
      failures.push(
        `${path}: has a recorded floor but is not under src/. Remove its floors entry if the file is gone.`,
      );
    }
  }

  return { failures, improvements };
}

export function main(
  floorsFile: string = floorsPath(),
  lcovFile: string = lcovPath(),
  sourceRoot: string = srcRoot(),
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
  const srcFiles = listSrcFiles(sourceRoot);
  const { failures, improvements } = checkCoverage(measured, config, srcFiles);

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
    `electron per-file coverage: ${floored} files at or above their floors ` +
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
