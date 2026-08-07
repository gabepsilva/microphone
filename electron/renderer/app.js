import {
  applyTranscriptDomEvent,
  renderPartialLine,
  renderTranscriptSnapshot,
} from "./transcript_view.js";
import {
  SHORTCUTS_PROMPT,
  SHORTCUTS_SESSION,
  commandForEvent,
  nextPolicy,
} from "./shortcuts.js";

const NO_MIC = "__none__";
const NO_AUDIO = "none";
const api = window.tagalong;
const banner = document.getElementById("banner");
const transcriptList = document.getElementById("transcript-list");
const partialLine = document.getElementById("partial");
const emptyScreen = document.getElementById("empty-transcript");
const transcriptArea = document.getElementById("transcript-area");
const composeText = document.getElementById("compose-text");
let applying = false;
let attachmentIds = [];
// Last state seen, so a keyboard shortcut can toggle a value it must first read.
let current = {};

function setBanner(text, isError) {
  banner.textContent = text;
  banner.classList.toggle("error", Boolean(isError));
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
  // A channel reads as live when something is selected and nothing is muting it.
  const mic = document.getElementById("mic-dot");
  const micLive = Boolean(state.microphone?.effective) && !state.microphone_muted;
  mic.textContent = micLive ? "●" : "○";
  mic.classList.toggle("live", micLive);
  const audio = document.getElementById("audio-dot");
  const audioEffective = state.audio_stream?.effective;
  const audioLive =
    Boolean(audioEffective) && audioEffective !== NO_AUDIO && !state.audio_stream_muted;
  audio.textContent = audioLive ? "●" : "○";
  audio.classList.toggle("live", audioLive);
}

async function refreshDevices(state) {
  let listed;
  try {
    listed = await api.devicesList();
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

function applyState(state) {
  applying = true;
  current = state;
  document.getElementById("mic-mute").checked = Boolean(state.microphone_muted);
  document.getElementById("audio-mute").checked = Boolean(state.audio_stream_muted);
  document.getElementById("tts-enabled").checked = Boolean(state.tts_enabled);
  document.getElementById("tts-state").textContent = state.tts_enabled ? "on" : "off";
  document.getElementById("response-policy").value = state.response_policy || "both";
  document.getElementById("tts-provider").value = state.tts_provider || "piper";
  document.getElementById("codex-model").value = state.codex_model || "";
  document.getElementById("codex-reasoning").value = state.codex_reasoning || "";
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
  setBanner("live · session attached");
}

async function dispatch(action, payload) {
  try {
    await api.dispatch(action, payload);
  } catch (error) {
    setBanner(error.message, true);
  }
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
  await dispatch("message.send", { text, images: ids, respond: true });
  composeText.value = "";
  autoGrow();
  fileInput.value = "";
  attachmentIds = [];
  markAttached(0);
}

/** The prompt grows with the draft, the way the TUI's does. */
function autoGrow() {
  composeText.style.height = "auto";
  composeText.style.height = `${composeText.scrollHeight}px`;
}

const commands = {
  "prompt.send": () => void send(),
  "prompt.clear": () => {
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
  document.getElementById("codex-model").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("codex.set_model", { model: e.target.value });
  });
  document.getElementById("codex-reasoning").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("codex.set_reasoning", { effort: e.target.value });
  });
  document.getElementById("turn-silence").addEventListener("change", (e) => {
    if (applying) return;
    void dispatch("turn_silence.set", { seconds: Number(e.target.value) });
  });

  document.getElementById("interrupt").addEventListener("click", () => {
    commands["session.interrupt"]();
  });
  document.getElementById("end-turn").addEventListener("click", () => {
    commands["session.end_turn"]();
  });
  document.getElementById("new-session").addEventListener("click", () => {
    commands["session.new"]();
  });
  document.getElementById("save-transcript").addEventListener("click", () => {
    commands["session.save_transcript"]();
  });

  composeText.addEventListener("input", autoGrow);
  document.getElementById("send").addEventListener("click", () => {
    commands["prompt.send"]();
  });
  document.getElementById("sidebar-toggle").addEventListener("click", () => {
    commands["view.toggle_sidebar"]();
  });
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
    renderTranscriptSnapshot(transcriptList, document, rows);
    syncEmptyScreen();
    scrollToLatest();
  });
  api.onTranscriptEvent((event) => {
    if (event === null || typeof event !== "object") {
      return;
    }
    applyTranscriptDomEvent(transcriptList, document, event);
    syncEmptyScreen();
    scrollToLatest();
  });
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
  // Refresh device catalogs periodically — hardware changes are not
  // on the state.changed stream.
  setInterval(() => {
    api
      .snapshot()
      .then((snap) => refreshDevices(snap.state))
      .catch(() => {});
  }, 5000);
}

boot();
