/**
 * Local Electron watch loop: rebuild + process restart for src/, soft reload
 * for renderer/ (main watches that dir when TAGALONG_ELECTRON_DEV=1).
 *
 * Usage: `bun run dev` from electron/ (TagAlong socket must already be up).
 */

import { spawn, type Subprocess } from "bun";
import { watch } from "node:fs";
import { join } from "node:path";

/** package.json scripts run with cwd = electron/. */
const root = process.cwd();
const electronBin = join(root, "node_modules", ".bin", "electron");
const electronArgs = [
  "--no-sandbox",
  "--disable-gpu",
  "--disable-gpu-sandbox",
  "--in-process-gpu",
  "--ozone-platform-hint=auto",
  ".",
];

const DEBOUNCE_MS = 120;

let child: Subprocess | null = null;
let restarting = false;
let pending: "rebuild" | null = null;
let srcTimer: ReturnType<typeof setTimeout> | null = null;

async function build(): Promise<boolean> {
  console.log("[dev] building…");
  const tsc = spawn({
    cmd: ["bun", "x", "tsc", "-p", "tsconfig.json"],
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
  });
  const tscCode = await tsc.exited;
  if (tscCode !== 0) {
    console.error(`[dev] tsc failed (${tscCode})`);
    return false;
  }
  const preload = spawn({
    cmd: [
      "bun",
      "build",
      "src/preload.ts",
      "--outfile=dist/preload.js",
      "--target=node",
      "--format=cjs",
      "--external",
      "electron",
    ],
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
  });
  const preloadCode = await preload.exited;
  if (preloadCode !== 0) {
    console.error(`[dev] preload bundle failed (${preloadCode})`);
    return false;
  }
  return true;
}

async function stopElectron(): Promise<void> {
  if (child === null) {
    return;
  }
  const current = child;
  child = null;
  current.kill();
  try {
    await current.exited;
  } catch {
    // already gone
  }
}

async function startElectron(): Promise<void> {
  child = spawn({
    cmd: [electronBin, ...electronArgs],
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
    env: {
      ...process.env,
      TAGALONG_ELECTRON_DEV: "1",
    },
  });
  void child.exited.then((code) => {
    if (child === null) {
      return;
    }
    child = null;
    console.log(`[dev] electron exited (${code ?? "?"})`);
  });
}

async function rebuildAndRestart(): Promise<void> {
  if (restarting) {
    pending = "rebuild";
    return;
  }
  restarting = true;
  try {
    do {
      pending = null;
      await stopElectron();
      const ok = await build();
      if (!ok) {
        console.error("[dev] fix the build, then save again");
        break;
      }
      await startElectron();
      console.log("[dev] electron running (renderer soft-reloads; src/ restarts)");
    } while (pending === "rebuild");
  } finally {
    restarting = false;
  }
}

function scheduleSrcRebuild(): void {
  if (srcTimer !== null) {
    clearTimeout(srcTimer);
  }
  srcTimer = setTimeout(() => {
    srcTimer = null;
    void rebuildAndRestart();
  }, DEBOUNCE_MS);
}

async function main(): Promise<number> {
  watch(join(root, "src"), { recursive: true }, (_event, filename) => {
    if (filename === null || filename.endsWith(".tsbuildinfo")) {
      return;
    }
    console.log(`[dev] src change: ${filename}`);
    scheduleSrcRebuild();
  });

  console.log("[dev] watching src/ and renderer/ (soft reload via main)");
  await rebuildAndRestart();
  // Keep the watch process alive until Ctrl+C.
  await new Promise<void>(() => undefined);
  return 0;
}

const invoked = process.argv[1] ?? "";
if (
  invoked.endsWith(`${join("electron", "dev.ts")}`) ||
  invoked.endsWith(`${join("electron", "dev.js")}`) ||
  invoked.endsWith("dev.ts") ||
  invoked.endsWith("dev.js")
) {
  void main().then((code) => {
    process.exit(code);
  });
}
