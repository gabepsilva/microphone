import {
  applyTranscriptDomEvent,
  isTailing,
  renderPartialLine,
  renderTranscriptSnapshot,
} from "./transcript_view.js";
import {
  SHORTCUTS_PROMPT,
  SHORTCUTS_SESSION,
  commandForEvent,
  nextPolicy,
} from "./shortcuts.js";
import {
  clampIndex,
  commandQuery,
  findCommand,
  matchCommands,
  parseCommandList,
  slashArguments,
} from "./commands.js";
import {
  effortForModel,
  effortOptions,
  modelOptions,
  parseCodexCatalog,
} from "./codex_catalog.js";
import { parseSpeechCatalog, voiceOptionsIncluding } from "./speech_catalog.js";
import {
  syncVoicePicker as drawVoicePicker,
  voiceChangePayload,
  voiceEffectiveText,
} from "./voice_picker.js";

const NO_MIC = "__none__";
const NO_AUDIO = "none";
const api = window.tagalong;
const banner = document.getElementById("banner");
const transcriptList = document.getElementById("transcript-list");
const partialLine = document.getElementById("partial");
const emptyScreen = document.getElementById("empty-transcript");
const transcriptArea = document.getElementById("transcript-area");
const composeText = document.getElementById("compose-text");
const palette = document.getElementById("command-palette");
let applying = false;
let attachmentIds = [];
// Last state seen, so a keyboard shortcut can toggle a value it must first read.
let current = {};
// The session's model catalog, for the two Codex pickers.
let speechVoices = [];
let codexModels = [];
// The session's slash catalog, and what the open menu is showing of it.
let catalog = [];
let paletteRows = [];
let paletteIndex = 0;

const LIVE_BANNER = "live · session attached";

function setBanner(text, isError) {
  banner.textContent = text;
  banner.classList.toggle("error", Boolean(isError));
}

/**
 * Say the session is live again, unless the banner is carrying an error.
 *
 * `state.changed` arrives for anything the session does, including things the
 * failed call had nothing to do with, so restating "live" on every fragment
 * would wipe an error before it had been read. An error stays until something
 * actually succeeds.
 */
function setBannerLive() {
  if (!banner.classList.contains("error")) {
    banner.textContent = LIVE_BANNER;
  }
}

/** Clear an error the banner is showing, because a call has just worked. */
function clearBannerError() {
  if (banner.classList.contains("error")) {
    setBanner(LIVE_BANNER, false);
  }
}

if (!api) {
  setBanner("preload bridge missing (window.tagalong undefined)", true);
}

/** How the sidebar picker labels a policy, so the prompt chip can restate it. */
const POLICY_LABELS = {
  audio: "Audio",
  both: "Voice + Audio",
  voice: "Voice",
  quiet: "Stay silent",
};

/** Draw the splash key reference from the one shortcut table. */
function renderShortcuts(el, shortcuts) {
  el.replaceChildren();
  for (const shortcut of shortcuts) {
    const row = document.createElement("div");
    row.className = "shortcut";
    const keys = document.createElement("kbd");
    keys.className = "shortcut-keys";
    keys.textContent = shortcut.keys;
    const label = document.createElement("span");
    label.className = "shortcut-label";
    label.textContent = shortcut.label;
    row.appendChild(keys);
    row.appendChild(label);
    el.appendChild(row);
  }
}

/** The welcome pane shows only until the first entry lands. */
function syncEmptyScreen() {
  emptyScreen.hidden = transcriptList.children.length > 0;
}

function scrollToLatest() {
  transcriptArea.scrollTop = transcriptArea.scrollHeight;
}

function fillSelect(select, options, selected) {
  const wanted = selected;
  select.replaceChildren();
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.label;
    select.appendChild(el);
  }
  if ([...select.options].some((o) => o.value === wanted)) {
    select.value = wanted;
  }
}

function syncEffective(state) {
  document.getElementById("microphone-effective").textContent = state.microphone
    ?.effective
    ? `effective: ${state.microphone.effective}`
    : "";
  document.getElementById("audio-effective").textContent = state.audio_stream?.effective
    ? `effective: ${state.audio_stream.effective}`
    : "";
  document.getElementById("tts-voice-effective").textContent =
    voiceEffectiveText(state);
  // A channel reads as live when something is selected and nothing is muting
  // it. The dot is drawn in CSS — giving it a glyph too would show two marks.
  const micLive = Boolean(state.microphone?.effective) && !state.microphone_muted;
  document.getElementById("mic-dot").classList.toggle("live", micLive);
  const audioEffective = state.audio_stream?.effective;
  const audioLive =
    Boolean(audioEffective) && audioEffective !== NO_AUDIO && !state.audio_stream_muted;
  document.getElementById("audio-dot").classList.toggle("live", audioLive);
}

/** Draw the Piper voice picker when the engine is local. */
function syncVoicePicker(state) {
  drawVoicePicker(
    document.getElementById("tts-voice-field"),
    document.getElementById("tts-voice"),
    state,
    speechVoices,
    (tag) => document.createElement(tag),
    voiceOptionsIncluding,
  );
}

/** Draw the model and effort pickers from the catalog and the running state. */
function syncCodexPickers(state) {
  const model = state.codex_model || "";
  const effort = state.codex_reasoning || "";
  fillSelect(
    document.getElementById("codex-model"),
    modelOptions(codexModels, model),
    model,
  );
  fillSelect(
    document.getElementById("codex-reasoning"),
    effortOptions(codexModels, model, effort).map((name) => ({
      value: name,
      label: name,
    })),
    effort,
  );
}

async function refreshDevices(state) {
  let listed;
  try {
    listed = await api.devicesList();
    clearBannerError();
  } catch (error) {
    setBanner(error.message, true);
    listed = { inputs: [], applications: [] };
  }
  const micOptions = [
    { value: NO_MIC, label: "None" },
    ...listed.inputs.map((d) => ({ value: d.name, label: d.name })),
  ];
  const audioOptions = [
    { value: NO_AUDIO, label: "None" },
    ...listed.applications.map((a) => ({
      value: a.name,
      label: a.label || a.name,
    })),
  ];
  fillSelect(
    document.getElementById("microphone"),
    micOptions,
    state.microphone?.desired ?? NO_MIC,
  );
  fillSelect(
    document.getElementById("audio-stream"),
    audioOptions,
    state.audio_stream?.desired ?? NO_AUDIO,
  );
  syncEffective(state);
}

async function refreshSpeechCatalog() {
  speechVoices = parseSpeechCatalog(await api.speechCatalog());
  syncVoicePicker(current);
}

function applyState(state) {
  applying = true;
  current = state;
  document.getElementById("mic-mute").checked = Boolean(state.microphone_muted);
  document.getElementById("audio-mute").checked = Boolean(state.audio_stream_muted);
  document.getElementById("tts-enabled").checked = Boolean(state.tts_enabled);
  document.getElementById("tts-state").textContent = state.tts_enabled ? "on" : "off";
  document.getElementById("response-policy").value = state.response_policy || "both";
  document.getElementById("tts-provider").value = state.tts_provider || "piper";
  syncVoicePicker(state);
  syncCodexPickers(state);
  document.getElementById("turn-silence").value = state.turn_silence ?? 3;
  // The prompt restates what the sidebar decides, the way a chat client shows
  // its model beside the box you type in.
  document.getElementById("prompt-model").textContent = state.codex_model || "";
  const policy = state.response_policy || "both";
  document.getElementById("policy-chip").textContent =
    `Replies to ${POLICY_LABELS[policy] ?? policy}`;
  const mic = document.getElementById("microphone");
  if (state.microphone?.desired != null) {
    if (![...mic.options].some((o) => o.value === state.microphone.desired)) {
      const opt = document.createElement("option");
      opt.value = state.microphone.desired;
      opt.textContent = state.microphone.desired;
      mic.appendChild(opt);
    }
    mic.value = state.microphone.desired;
  } else {
    mic.value = NO_MIC;
  }
  const audio = document.getElementById("audio-stream");
  const wanted = state.audio_stream?.desired ?? NO_AUDIO;
  if (![...audio.options].some((o) => o.value === wanted)) {
    const opt = document.createElement("option");
    opt.value = wanted;
    opt.textContent = wanted;
    audio.appendChild(opt);
  }
  audio.value = wanted;
  syncEffective(state);
  renderPartialLine(partialLine, state);
  applying = false;
  setBannerLive();
}

/**
 * Run an action. Answers whether it was accepted, so a caller that is about
 * to throw away what the user typed can decline to.
 * @returns {Promise<boolean>}
 */
async function dispatch(action, payload) {
  try {
    await api.dispatch(action, payload);
  } catch (error) {
    setBanner(error.message, true);
    return false;
  }
  clearBannerError();
  return true;
}

/** Upload an image and return its attachment id, or null when it failed. */
async function uploadImage(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  try {
    const outcome = await api.dispatch("attachment.upload", { data: btoa(binary) });
    return outcome && outcome.effective ? outcome.effective : null;
  } catch (error) {
    setBanner(error.message, true);
    return null;
  }
}

function markAttached(count) {
  const button = document.getElementById("attach-button");
  button.classList.toggle("has-file", count > 0);
  button.title = count > 0 ? `${count} image attached` : "Attach an image";
}

async function send() {
  const text = composeText.value;
  // A slash line is a command, never something Taga is asked to answer.
  if (text.startsWith("/")) {
    const trimmed = text.trim();
    const spec = findCommand(catalog, trimmed);
    if (spec === null) {
      setBanner(`unknown command: ${trimmed}`, true);
      return;
    }
    // Action-backed commands dispatch with an empty payload — there is no
    // CommandRouter here to validate args the way the TUI does. Refuse
    // leftovers so `/new keep` cannot silently run `session.new`.
    if (spec.action_id !== null && slashArguments(trimmed).length > 0) {
      setBanner(`usage: /${spec.name}`, true);
      return;
    }
    paletteRows = [spec];
    paletteIndex = 0;
    await runSelectedCommand();
    return;
  }
  const fileInput = document.getElementById("compose-image");
  const ids = [...attachmentIds];
  if (fileInput.files && fileInput.files[0]) {
    const id = await uploadImage(fileInput.files[0]);
    if (id === null) {
      return;
    }
    ids.push(id);
  }
  if (!text.trim() && ids.length === 0) {
    return;
  }
  // Only clear the draft once the session has taken it. A refused or
  // undeliverable send otherwise leaves the banner explaining what went wrong
  // above an empty box, with the message itself gone.
  if (!(await dispatch("message.send", { text, images: ids, respond: true }))) {
    return;
  }
  composeText.value = "";
  autoGrow();
  fileInput.value = "";
  attachmentIds = [];
  markAttached(0);
}

/* -- slash-command palette ------------------------------------------- */

/** Draw the menu for what is typed, and say whether it is open. */
function syncPalette() {
  const query = commandQuery(composeText.value);
  paletteRows = query === null ? [] : matchCommands(catalog, query);
  if (query === null) {
    palette.hidden = true;
    palette.replaceChildren();
    paletteIndex = 0;
    return false;
  }
  paletteIndex = clampIndex(paletteIndex, paletteRows.length);
  palette.hidden = false;
  palette.replaceChildren();
  if (paletteRows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "command-empty";
    empty.textContent = "no matching commands";
    palette.appendChild(empty);
    return true;
  }
  paletteRows.forEach((spec, index) => {
    const row = document.createElement("div");
    row.className = index === paletteIndex ? "command-row selected" : "command-row";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(index === paletteIndex));
    const name = document.createElement("span");
    name.className = "command-name";
    name.textContent = `/${spec.name}`;
    row.appendChild(name);
    if (spec.aliases.length > 0) {
      const aliases = document.createElement("span");
      aliases.className = "command-aliases";
      aliases.textContent = spec.aliases.map((alias) => `/${alias}`).join(", ");
      row.appendChild(aliases);
    }
    const summary = document.createElement("span");
    summary.className = "command-summary";
    summary.textContent = spec.summary;
    row.appendChild(summary);
    row.addEventListener("mousedown", (event) => {
      // mousedown, not click: the prompt must not lose focus first.
      event.preventDefault();
      paletteIndex = index;
      void runSelectedCommand();
    });
    palette.appendChild(row);
  });
  return true;
}

function paletteOpen() {
  return palette.hidden === false;
}

function closePalette() {
  palette.hidden = true;
  palette.replaceChildren();
  paletteRows = [];
  paletteIndex = 0;
}

function movePaletteSelection(delta) {
  paletteIndex = clampIndex(paletteIndex + delta, paletteRows.length);
  syncPalette();
}

/** Put the highlighted command in the prompt, the way Tab completes it. */
function completeSelectedCommand() {
  const spec = paletteRows[paletteIndex];
  if (spec === undefined) {
    return;
  }
  composeText.value = `/${spec.name}`;
  autoGrow();
  syncPalette();
}

/**
 * Run the highlighted command.
 *
 * A command that names an action dispatches it. `/help` names none — the
 * catalog it would print is the menu already on screen — so it just leaves
 * the menu open. It says nothing in the banner either: the banner reports
 * connection state, and echoing a summary there only restates a row the
 * user is already looking at.
 */
async function runSelectedCommand() {
  const spec = paletteRows[paletteIndex];
  if (spec === undefined) {
    return;
  }
  if (spec.action_id === null) {
    completeSelectedCommand();
    return;
  }
  composeText.value = "";
  autoGrow();
  closePalette();
  await dispatch(spec.action_id, {});
}

// Height of an empty draft, measured once: what "still one line" means.
let promptBaseHeight = 0;

/** The prompt grows with the draft, the way the TUI's does. */
function autoGrow() {
  composeText.style.height = "auto";
  const height = composeText.scrollHeight;
  composeText.style.height = `${height}px`;
  if (promptBaseHeight === 0) {
    promptBaseHeight = height;
  }
  // The pill shape belongs to a single line; once the box grows, so does the
  // corner it needs.
  document
    .getElementById("prompt-shell")
    .classList.toggle("multiline", height > promptBaseHeight + 2);
}

const commands = {
  "prompt.send": () => {
    // Prefer the highlighted row so Enter on `/ne` runs `/new`, not an
    // unknown partial — same rule as the TUI's prompt.
    if (paletteOpen() && paletteRows.length > 0) {
      void runSelectedCommand();
      return;
    }
    void send();
  },
  "prompt.clear": () => {
    if (paletteOpen()) {
      closePalette();
      return;
    }
    composeText.value = "";
    autoGrow();
  },
  "session.cycle_policy": () =>
    void dispatch("response_policy.set", {
      policy: nextPolicy(current.response_policy),
    }),
  "session.toggle_mic_mute": () =>
    void dispatch("microphone.set_muted", { muted: !current.microphone_muted }),
  "session.toggle_tts": () =>
    void dispatch("tts.set_enabled", { enabled: !current.tts_enabled }),
  "session.interrupt": () => void dispatch("session.interrupt", {}),
  "session.end_turn": () => void dispatch("voice.end_turn", {}),
  "session.save_transcript": () => void dispatch("transcript.save", {}),
  "session.new": () => void dispatch("session.new", {}),
  "view.toggle_sidebar": () =>
    document.getElementById("body").classList.toggle("sidebar-hidden"),
};

function bind() {
  document.getElementById("microphone").addEventListener("change", (e) => {
    if (applying) return;
    const name = e.target.value === NO_MIC ? null : e.target.value;
    void dispatch("microphone.select", { name });
  });
  document.getElementById("audio-stream").addEventListener("change", (e) => {
    if (applying) return;
    const name = e.target.value;
    void dispatch("audio_stream.select", { name });
  });
  document.getElementById("mic-mute").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("microphone.set_muted", { muted: e.target.checked });
  });
  document.getElementById("audio-mute").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("audio_stream.set_muted", { muted: e.target.checked });
  });
  document.getElementById("response-policy").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("response_policy.set", { policy: e.target.value });
  });
  document.getElementById("tts-enabled").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("tts.set_enabled", { enabled: e.target.checked });
  });
  document.getElementById("tts-provider").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("tts.set_provider", { provider: e.target.value });
  });
  document.getElementById("tts-voice").addEventListener("change", (e) => {
    const payload = voiceChangePayload(applying, e.target.value);
    if (payload === null) return;
    void dispatch("tts.set_voice", payload);
  });
  document.getElementById("codex-model").addEventListener("change", (e) => {
    if (applying) return;
    const model = e.target.value;
    void dispatch("codex.set_model", { model }).then(() => {
      // A model that does not accept the running effort has to be given one
      // it does, or the session keeps a setting the model refuses.
      const effort = effortForModel(codexModels, model, current.codex_reasoning || "");
      if (effort !== null) {
        void dispatch("codex.set_reasoning", { effort });
      }
    });
  });
  document.getElementById("codex-reasoning").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("codex.set_reasoning", { effort: e.target.value });
  });
  // `input`, not `change`: a number field only fires `change` on blur, and a
  // setting typed into the sidebar should be in force before the focus moves.
  document.getElementById("turn-silence").addEventListener("input", (e) => {
    if (applying) return;
    const seconds = Number(e.target.value);
    if (e.target.value === "" || !Number.isFinite(seconds)) {
      return;
    }
    void dispatch("turn_silence.set", { seconds });
  });

  composeText.addEventListener("input", () => {
    autoGrow();
    syncPalette();
  });
  // Browsing and completing the menu happen before the shortcut table sees
  // the key, so ↑↓ and Tab mean the menu while it is open.
  composeText.addEventListener("keydown", (event) => {
    if (!paletteOpen()) {
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      movePaletteSelection(event.key === "ArrowDown" ? 1 : -1);
      return;
    }
    if (event.key === "Tab") {
      event.preventDefault();
      completeSelectedCommand();
    }
  });
  document.getElementById("send").addEventListener("click", () => {
    commands["prompt.send"]();
  });
  for (const id of ["sidebar-toggle", "sidebar-restore"]) {
    document.getElementById(id).addEventListener("click", () => {
      commands["view.toggle_sidebar"]();
    });
  }
  const fileInput = document.getElementById("compose-image");
  document.getElementById("attach-button").addEventListener("click", () => {
    fileInput.click();
  });
  fileInput.addEventListener("change", (e) => {
    markAttached(e.target.files && e.target.files.length ? 1 : 0);
  });
  // ^V of an image lands here rather than in the file picker.
  composeText.addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files ?? [])];
    const image = files.find((file) => file.type.startsWith("image/"));
    if (image === undefined) {
      return;
    }
    event.preventDefault();
    void uploadImage(image).then((id) => {
      if (id !== null) {
        attachmentIds.push(id);
        markAttached(attachmentIds.length);
      }
    });
  });

  // One handler for every binding: the shortcut table decides what a key is.
  window.addEventListener("keydown", (event) => {
    const command = commandForEvent(event);
    if (command === null) {
      return;
    }
    // Enter belongs to the prompt; a select or a text field keeps its own.
    const target = event.target;
    if (
      command === "prompt.send" &&
      target !== composeText &&
      target?.tagName !== "BODY"
    ) {
      return;
    }
    const run = commands[command];
    if (run === undefined) {
      return;
    }
    event.preventDefault();
    run();
  });
}

async function boot() {
  renderShortcuts(document.getElementById("shortcuts-prompt"), SHORTCUTS_PROMPT);
  renderShortcuts(document.getElementById("shortcuts-session"), SHORTCUTS_SESSION);
  syncEmptyScreen();
  renderPartialLine(partialLine, {});
  // The prompt has the keyboard from the first frame, as in the TUI.
  composeText.focus();
  autoGrow();
  if (!api) {
    return;
  }
  bind();
  api.onState((state) => {
    applyState(state);
  });
  api.onTranscriptSnapshot((rows) => {
    if (!Array.isArray(rows)) {
      return;
    }
    // A snapshot is a fresh start — boot, or resubscribe after `lost`. There
    // is no reading position to preserve across a list that was replaced.
    renderTranscriptSnapshot(transcriptList, document, rows);
    syncEmptyScreen();
    scrollToLatest();
  });
  api.onTranscriptEvent((event) => {
    if (event === null || typeof event !== "object") {
      return;
    }
    const follow = isTailing(transcriptArea);
    applyTranscriptDomEvent(transcriptList, document, event);
    syncEmptyScreen();
    if (follow) {
      scrollToLatest();
    }
  });
  // The transcript and the settings come first: everything below is a catalog
  // the UI degrades gracefully without, and `codex.catalog` in particular
  // shells out to `codex debug models` on the session's side. Awaiting either
  // one here would hold the first paint behind a subprocess.
  try {
    const snapshot = await api.snapshot();
    await refreshDevices(snapshot.state);
    applyState(snapshot.state);
    if (Array.isArray(snapshot.transcript)) {
      renderTranscriptSnapshot(transcriptList, document, snapshot.transcript);
      syncEmptyScreen();
      scrollToLatest();
    }
  } catch (error) {
    setBanner(error.message, true);
  }
  try {
    catalog = parseCommandList(await api.commandsList());
  } catch (error) {
    // A session without a catalog still types; only the menu is missing.
    setBanner(error.message, true);
  }
  try {
    // Read once: the CLI's catalog does not change under a running session.
    codexModels = parseCodexCatalog(await api.codexCatalog());
    // Redraw the pickers: the state that would have drawn them has landed.
    syncCodexPickers(current);
  } catch (error) {
    // Without a catalog the pickers still offer what the session is running.
    setBanner(error.message, true);
  }
  try {
    await refreshSpeechCatalog();
  } catch (error) {
    setBanner(error.message, true);
  }
  // Refresh device catalogs periodically — hardware changes are not
  // on the state.changed stream. `devices.list`, not `snapshot`: the snapshot
  // payload carries the whole transcript (transport.snapshot_payload), and
  // re-serialising every accepted row across the socket and IPC every five
  // seconds to redraw two selects costs more the longer the session runs.
  // The desired/effective values it needs are already on the last state seen.
  // The same interval refreshes speech.catalog: `downloaded` flips when a
  // voice finishes fetching, and that fact is not on state.changed either.
  setInterval(() => {
    void refreshDevices(current).catch(() => {});
    void refreshSpeechCatalog().catch(() => {});
  }, 5000);
}

boot();
