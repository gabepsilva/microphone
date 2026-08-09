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
  decideSubmit,
  matchCommands,
  parseCommandList,
} from "./commands.js";
import {
  effortForModel,
  effortOptions,
  modelOptions,
  parseCodexCatalog,
} from "./codex_catalog.js";
import { bindSidebarResize } from "./sidebar_resize.js";
import { DraftAttachments, base64FromBytes, parseImageTokens } from "./attachments.js";
import { stageFiles } from "./staging.js";
import { parseSpeechCatalog, voiceOptionsIncluding } from "./speech_catalog.js";
import {
  selectedProviderId,
  selectionEffectiveText,
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
// The draft is the token model: `[Image #N]` in the composer text names the
// staged attachment, exactly like the TUI's DraftAttachments (#139 D1).
// Removes and reorders are text edits; the chips below only render the scan.
const draft = new DraftAttachments();
// Chip thumbnails keyed by token number — derived view data, never the model.
const stagedThumbs = new Map();
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

/**
 * Update the status banner.
 *
 * `.error` is red styling only. `.sticky` means "do not overwrite on
 * state.changed" — errors set both (default `sticky = isError`); topic-help
 * info sets sticky without error so a correct answer is not painted red.
 *
 * Lifetime: sticky info survives `setBannerLive` and background
 * `clearBannerError` polls; user-initiated `dispatch()` success restores
 * LIVE_BANNER (retires sticky and error without a timer).
 *
 * @param {string} text
 * @param {boolean} isError
 * @param {boolean} [sticky]
 */
function setBanner(text, isError, sticky = isError) {
  banner.textContent = text;
  banner.classList.toggle("error", Boolean(isError));
  banner.classList.toggle("sticky", Boolean(sticky));
}

/**
 * Say the session is live again, unless the banner is sticky.
 *
 * `state.changed` arrives for anything the session does, including partial
 * ASR commits, so restating "live" on every fragment would wipe command
 * feedback (errors and topic-help info) before it had been read.
 */
function setBannerLive() {
  if (!banner.classList.contains("sticky")) {
    banner.textContent = LIVE_BANNER;
  }
}

/**
 * Clear an error the banner is showing, because a background call worked.
 *
 * Sticky *info* is left alone: success polls (devices.list every 5s) must not
 * retire a help answer on a timer. Errors set both classes, so clearing an
 * error still drops stickiness via `setBanner(LIVE_BANNER, false)`.
 * User-initiated success goes through `dispatch()`, which restores LIVE_BANNER
 * unconditionally.
 */
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

function syncChannelDots(state) {
  document.getElementById("tts-voice-effective").textContent =
    voiceEffectiveText(state);
  document.getElementById("tts-provider-effective").textContent =
    selectionEffectiveText(
      state.tts_provider && typeof state.tts_provider === "object"
        ? state.tts_provider
        : undefined,
    );
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

function syncSession(state) {
  const thread = document.getElementById("session-codex-thread");
  thread.textContent = state.codex_thread || "—";
  const codexState = state.codex_state || "idle";
  const activity =
    codexState !== "idle" ? codexState : state.codex_speaking ? "speaking" : "idle";
  const activityElement = document.getElementById("session-codex-state");
  activityElement.textContent = activity;
  activityElement.classList.toggle("active", activity !== "idle");
  const confidence = Number(state.confidence);
  document.getElementById("session-confidence").textContent = Number.isFinite(
    confidence,
  )
    ? confidence.toFixed(2)
    : "—";
  document.getElementById("session-language").textContent = state.language || "—";
  document.getElementById("session-moonshine").textContent = state.moonshine || "—";
  const tokens = Number(state.tokens);
  document.getElementById("session-tokens").textContent = Number.isFinite(tokens)
    ? tokens.toLocaleString("en-US")
    : "—";
  const echoesCut = Number(state.echoes_cut);
  document.getElementById("session-echoes-cut").textContent = Number.isFinite(echoesCut)
    ? String(echoesCut)
    : "—";
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
  syncChannelDots(state);
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
  document.getElementById("tts-provider").value = selectedProviderId(state);
  syncVoicePicker(state);
  syncCodexPickers(state);
  document.getElementById("turn-silence").value = state.turn_silence ?? 3;
  // The prompt restates what the sidebar decides, the way a chat client shows
  // its model beside the box you type in.
  document.getElementById("prompt-model").textContent = state.codex_model || "";
  const policy = state.response_policy || "both";
  document.getElementById("policy-chip").textContent =
    `Replies to ${POLICY_LABELS[policy] ?? policy}`;
  syncSession(state);
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
  syncChannelDots(state);
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
  // User-initiated path only (the 5s interval never calls dispatch). Restore
  // LIVE_BANNER so sticky topic-help does not outlive the next successful
  // command or message — without coupling to background clearBannerError.
  setBanner(LIVE_BANNER, false);
  return true;
}

/**
 * Upload image bytes and return the attachment id, or null when it failed.
 *
 * The payload goes through base64FromBytes (chunked btoa), not the
 * per-byte String.fromCharCode loop that cost ~346 ms for a 5 MiB image
 * (#139 F5). Refusal stays the staging loop's job: no rejected file must
 * ever reach this call, or the transport would drop the socket (#139 F1).
 * @param {Uint8Array} bytes
 * @returns {Promise<string | null>}
 */
async function uploadImage(bytes) {
  try {
    const outcome = await api.dispatch("attachment.upload", {
      data: base64FromBytes(bytes),
    });
    return outcome && outcome.effective ? outcome.effective : null;
  } catch (error) {
    setBanner(error.message, true);
    return null;
  }
}

/**
 * Read a stage file once, decode it straight from the File, and return a
 * small canvas thumbnail (data: URI, CSP allows img-src data: — #139 D7).
 * createImageBitmap takes the File directly, so there is no second read, no
 * second base64 pass, and no blob URL for the CSP to block. Null when the
 * decode fails; the chip renders its token text either way.
 * @param {File} file
 * @returns {Promise<string | null>}
 */
async function chipThumbnail(file) {
  let bitmap = null;
  try {
    bitmap = await createImageBitmap(file);
    const size = 96;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      return null;
    }
    const scale = Math.min(size / bitmap.width, size / bitmap.height);
    const width = bitmap.width * scale;
    const height = bitmap.height * scale;
    ctx.drawImage(bitmap, (size - width) / 2, (size - height) / 2, width, height);
    return canvas.toDataURL("image/png");
  } catch {
    return null;
  } finally {
    // The full-resolution decode must die on every path — a 20 MiB JPEG can
    // be ~192 MB of RGBA, and a throw inside drawImage/toDataURL otherwise
    // leaves it to GC.
    bitmap?.close();
  }
}

/**
 * Insert one or more tokens at the cursor, like the TUI prompt inserts a
 * pasted image's token at the draft point.
 * @param {string[]} tokens
 */
function insertTokensAtCursor(tokens) {
  const start = composeText.selectionStart ?? composeText.value.length;
  const end = composeText.selectionEnd ?? composeText.value.length;
  const inserted = tokens.join(" ");
  composeText.value =
    composeText.value.slice(0, start) + inserted + composeText.value.slice(end);
  const next = start + inserted.length;
  composeText.selectionStart = next;
  composeText.selectionEnd = next;
  autoGrow();
  syncPalette();
  renderComposerDraft();
}

/**
 * One staging loop for paste, drop, and the picker (#139 D2): preflight
 * every file, upload the acceptable ones, refuse the rest with copy the
 * banner can show. Refusals never reach the socket; uploaded images insert
 * their token into the composer text.
 * @param {File[]} files
 */
async function stageAndInsert(files) {
  const results = await stageFiles(files, uploadImage);
  const refusals = [];
  for (const result of results) {
    if (result.refused !== null) {
      refusals.push(result.refused);
      continue;
    }
    const id = result.id;
    if (id == null || id === "") {
      continue;
    }
    const token = draft.add(id);
    insertTokensAtCursor([token]);
    const number = draft.ids.length;
    if (number !== undefined) {
      void chipThumbnail(result.file).then((thumb) => {
        if (thumb !== null) {
          stagedThumbs.set(number, thumb);
          renderComposerDraft();
        }
      });
    }
  }
  if (refusals.length > 0) {
    setBanner(refusals.join(" — "), true);
  }
}

/**
 * Draw the token chips and the attach button from the composer text scan.
 *
 * The chips are a rendering of the tokens in the draft, so deleting a token
 * from the text removes its chip (and its attachment from the submit set).
 * A hand-typed token gets a chip too — chips echo, they do not assert.
 */
function renderComposerDraft() {
  const tokens = parseImageTokens(composeText.value);
  document
    .getElementById("attach-button")
    .classList.toggle("has-file", tokens.length > 0);
  document.getElementById("attach-button").title =
    tokens.length > 0 ? `${tokens.length} image attached` : "Attach an image";
  const strip = document.getElementById("composer-chips");
  strip.replaceChildren();
  strip.hidden = tokens.length === 0;
  for (const parsed of tokens) {
    const chip = document.createElement("span");
    chip.className = "stage-chip";
    const thumb = stagedThumbs.get(parsed.number);
    if (thumb !== undefined) {
      const image = document.createElement("img");
      image.className = "stage-thumb";
      image.src = thumb;
      image.alt = "";
      chip.appendChild(image);
    }
    const label = document.createElement("span");
    label.className = "stage-token";
    label.textContent = parsed.token;
    chip.appendChild(label);
    strip.appendChild(chip);
  }
}

/** A line over the composer, lit while a file drag hovers the window. */
function dragAffordance(on) {
  document.getElementById("prompt-shell").classList.toggle("drag-target", on);
}

async function send() {
  // decideSubmit strips once (TUI parity) and classifies; message branch
  // ships decision.text so trailing spaces never ride the wire. The token
  // model stays in the text — the record is the token plus its ids.
  const decision = decideSubmit(composeText.value, catalog);
  if (decision.kind === "error") {
    setBanner(decision.text, true);
    return;
  }
  if (decision.kind === "info") {
    // Sticky, non-error: survives setBannerLive on partial ASR until the next
    // successful dispatch() restores LIVE_BANNER. Lifetime stays in
    // uncovered app.js — pure decideSubmit is tested.
    setBanner(decision.text, false, true);
    composeText.value = "";
    autoGrow();
    closePalette();
    return;
  }
  if (decision.kind === "help") {
    // Bare `/help`, `/?`, or `/`: list every command. Lifetime stays in
    // uncovered app.js — the classification is tested in decideSubmit.
    showBareHelp();
    return;
  }
  if (decision.kind === "command") {
    paletteRows = [decision.spec];
    paletteIndex = 0;
    await runSelectedCommand();
    return;
  }
  const text = decision.text;
  const ids = draft.resolve(text);
  if (!text && ids.length === 0) {
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
  draft.clear();
  stagedThumbs.clear();
  renderComposerDraft();
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
 * Park the prompt on `/` and draw the whole catalog: bare help.
 *
 * `/help` names no action, and the listing the TUI prints for it is exactly
 * what a palette row already shows — same three fields. So bare help is the
 * menu with nothing filtered out.
 *
 * It must not go through `completeSelectedCommand`. That writes `/help` into
 * the prompt, and `syncPalette` recomputes rows from `commandQuery` of the
 * prompt, so the catalog collapses to the single `/help` row — the command
 * describing itself. `/` is the query that matches everything.
 */
function showBareHelp() {
  composeText.value = "/";
  autoGrow();
  syncPalette();
  // Arm the help row, not row 0. Enter takes the highlighted row, so leaving
  // `/new` armed would make the keystroke after an informational command start
  // a fresh thread and clear the transcript, with no confirm. The TUI does not
  // leave you there either: it clears the draft before dispatching. Typing `/`
  // fresh still arms `/new`, which is the TUI's behaviour.
  const help = paletteRows.findIndex((spec) => spec.action_id === null);
  paletteIndex = help === -1 ? 0 : help;
  // clampIndex is modulo, so a second sync redraws the highlight rather than
  // resetting it.
  syncPalette();
}

/**
 * Run the highlighted command.
 *
 * A command that names an action dispatches it. `/help` names none, so it
 * shows the full menu. It says nothing in the banner either: the banner
 * reports connection state, and echoing a summary there only restates a row
 * the user is already looking at.
 */
async function runSelectedCommand() {
  const spec = paletteRows[paletteIndex];
  if (spec === undefined) {
    return;
  }
  if (spec.action_id === null) {
    showBareHelp();
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
    // Typing or deleting a token re-derives the chip strip (#139 D1).
    renderComposerDraft();
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
  bindSidebarResize({
    body: document.getElementById("body"),
    handle: document.getElementById("sidebar-resize"),
  });
  const fileInput = document.getElementById("compose-image");
  document.getElementById("attach-button").addEventListener("click", () => {
    fileInput.click();
  });
  // The picker feeds the same staging loop as paste and drop (#139 D2).
  fileInput.addEventListener("change", (e) => {
    const files = [...(e.target.files ?? [])];
    e.target.value = "";
    if (files.length === 0) {
      return;
    }
    void stageAndInsert(files);
  });
  // `image/*` paste lands here. Text (with or without rich source) keeps the
  // browser default, which inserts plain text only (#139 Q4).
  composeText.addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files ?? [])];
    if (files.length === 0) {
      return;
    }
    event.preventDefault();
    void stageAndInsert(files);
  });
  // One staging loop for drop as well: the browser default for a dragged
  // file would navigate this window away from index.html — unrecoverable
  // (menu and dev reload are gone, #139 F2). The default is refused for the
  // whole window: a file drag anywhere lights the composer, and only the
  // composer region stages files (#139 D2, Q5). A file dropped anywhere
  // else gets a real refusal in the banner, never a silent vanish.
  const promptbar = document.getElementById("promptbar");
  let dragDepth = 0;
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    const types = event.dataTransfer ? [...event.dataTransfer.types] : [];
    // Only file drags are ours: a text or link drag gets no affordance and
    // no staging path, just the refused default.
    if (!types.includes("Files")) {
      return;
    }
    dragDepth += 1;
    dragAffordance(true);
  });
  window.addEventListener("dragover", (event) => {
    event.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    dragDepth -= 1;
    if (dragDepth <= 0) {
      dragDepth = 0;
      dragAffordance(false);
    }
  });
  window.addEventListener("drop", (event) => {
    dragDepth = 0;
    dragAffordance(false);
    // Refuse the default first, always: whatever the drag was, the window
    // must not navigate or load the file.
    event.preventDefault();
    const files = [...(event.dataTransfer?.files ?? [])];
    if (files.length === 0) {
      return;
    }
    if (!promptbar.contains(event.target)) {
      setBanner("Files stage on the composer — drop images on the prompt box", true);
      return;
    }
    void stageAndInsert(files);
  });
  window.addEventListener("dragend", () => {
    dragDepth = 0;
    dragAffordance(false);
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
