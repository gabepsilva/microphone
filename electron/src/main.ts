import { watch } from "node:fs";
import path from "node:path";

import {
  Notification,
  app,
  BrowserWindow,
  Menu,
  Tray,
  ipcMain,
  nativeImage,
} from "electron";

import { SessionEvents, TagAlongClient, type TranscriptWireEvent } from "./client";
import { dispatchAction, outcomeFailureDetail, registerIpcHandlers } from "./ipc";
import { ACTIONS } from "./protocol/actions";
import { CHANNELS } from "./protocol/channels";
import type { AppState, TranscriptRow } from "./state";
import { buildTrayMenu, trayMenuKey } from "./tray_menu";

// Control surface needs no WebGL. On some Linux GPU/driver stacks the GPU
// process dies (error_code=1002) and Chromium aborts the whole app. Set these
// before ready; the start script passes the same flags for earliest effect.
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-gpu-sandbox");
app.commandLine.appendSwitch("in-process-gpu");
// Under Wayland with fractional scaling, an XWayland window is rasterised at
// 1x and upscaled by the compositor — that, not the CSS, is what reads as
// blurry text. The hint selects Wayland natively when the session offers it
// and falls back to X11 otherwise, so Chromium rasterises glyphs at the
// monitor's real scale.
app.commandLine.appendSwitch("ozone-platform-hint", "auto");

/** Command connection — never park a long-poll here (#96 G1). */
const commands = new TagAlongClient();
/** Event connection — parked on poll; same actor id as commands. */
const events = new TagAlongClient();

let mainWindow: BrowserWindow | null = null;
/** Session tray — null when the icon asset is missing or Tray no-ops. */
let tray: Tray | null = null;
/** Last rendered menu identity — skip rebuild on partial-only state.changed. */
let lastTrayMenuKey: string | null = null;

function showTrayNotification(title: string, body: string): void {
  if (!Notification.isSupported()) {
    console.error("tray notification unsupported:", title, body);
    return;
  }
  new Notification({ title, body }).show();
}

function broadcastState(state: AppState): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.stateChanged, state);
  }
  rebuildTrayMenu(state);
}

function broadcastTranscriptSnapshot(rows: TranscriptRow[]): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.transcriptSnapshot, rows);
  }
}

function broadcastTranscriptEvent(event: TranscriptWireEvent): void {
  if (mainWindow !== null && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(CHANNELS.transcriptEvent, event);
  }
}

const sessionEvents = new SessionEvents(events, {
  onState: broadcastState,
  onTranscriptSnapshot: broadcastTranscriptSnapshot,
  onTranscriptEvent: broadcastTranscriptEvent,
  onError: (error) => {
    console.error("tagalong event loop:", error.message);
  },
});

async function dispatchMuted(
  action: typeof ACTIONS.microphone_set_muted | typeof ACTIONS.audio_stream_set_muted,
  muted: boolean,
): Promise<void> {
  try {
    // Goes through ipc.dispatchAction so Semgrep's allowlist chokepoint holds.
    const outcome = await dispatchAction(commands, action, { muted });
    const detail = outcomeFailureDetail(outcome);
    if (detail !== null) {
      // Window may be covered — console alone is silent for the tray use case (R3).
      showTrayNotification("Mute", detail);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // Window may be covered — console alone is silent for the tray use case (R3).
    showTrayNotification("Mute", message);
  }
}

function rebuildTrayMenu(state: AppState): void {
  if (tray === null) {
    return;
  }
  // Read aloud stays absent until #128b wires speech.read_selection.
  const handlers = {
    onToggleMicrophoneMute: () => {
      void dispatchMuted(
        ACTIONS.microphone_set_muted,
        !sessionEvents.state.microphone_muted,
      );
    },
    onToggleAudioStreamMute: () => {
      void dispatchMuted(
        ACTIONS.audio_stream_set_muted,
        !sessionEvents.state.audio_stream_muted,
      );
    },
  };
  const key = trayMenuKey(state, handlers);
  if (key === lastTrayMenuKey) {
    return;
  }
  lastTrayMenuKey = key;
  tray.setContextMenu(Menu.buildFromTemplate(buildTrayMenu(state, handlers)));
}

function createTray(): void {
  const iconPath = path.join(__dirname, "..", "assets", "tray-icon.png");
  const icon = nativeImage.createFromPath(iconPath);
  if (icon.isEmpty()) {
    console.error("tray icon missing or empty:", iconPath);
    return;
  }
  tray = new Tray(icon);
  tray.setToolTip("TagAlong");
  lastTrayMenuKey = null;
  rebuildTrayMenu(sessionEvents.state);
}

/** Renderer soft-reload for `bun run dev` only — never in a normal start. */
function enableDevRendererReload(): void {
  if (process.env.TAGALONG_ELECTRON_DEV !== "1") {
    return;
  }
  let timer: ReturnType<typeof setTimeout> | null = null;
  watch(path.join(__dirname, "..", "renderer"), { recursive: true }, () => {
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      if (mainWindow !== null && !mainWindow.isDestroyed()) {
        mainWindow.webContents.reloadIgnoringCache();
      }
    }, 50);
  });
}

async function createWindow(): Promise<void> {
  // No File/Edit/View/Window/Help: every entry it would offer is either a
  // lifecycle the TUI owns or a browser affordance this surface does not use.
  Menu.setApplicationMenu(null);
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  await mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
  // Push a real subscribe snapshot once the page can listen — never empty defaults.
  if (sessionEvents.hasSnapshot) {
    broadcastState(sessionEvents.state);
    broadcastTranscriptSnapshot([...sessionEvents.transcript]);
  }
}

registerIpcHandlers(ipcMain, commands);

void app.whenReady().then(async () => {
  // Show the window even when the TUI/socket is unhealthy — attach-only means
  // that case is expected, and an unbounded handshake must not hide the UI.
  await createWindow();
  createTray();
  enableDevRendererReload();
  void sessionEvents.start().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error("tagalong subscribe failed:", message);
  });
});

app.on("window-all-closed", () => {
  sessionEvents.stop();
  commands.close();
  if (tray !== null) {
    tray.destroy();
    tray = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});
