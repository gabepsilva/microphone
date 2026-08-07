import {
  applyTranscriptDomEvent,
  renderPartialLine,
  renderTranscriptSnapshot,
} from "./transcript_view.js";

const NO_MIC = "__none__";
const NO_AUDIO = "none";
const api = window.tagalong;
const banner = document.getElementById("banner");
const transcriptList = document.getElementById("transcript-list");
const partialLine = document.getElementById("partial-line");
let applying = false;
let attachmentIds = [];

function setBanner(text, isError) {
  banner.textContent = text;
  banner.classList.toggle("error", Boolean(isError));
}

if (!api) {
  setBanner("preload bridge missing (window.tagalong undefined)", true);
}

function fillSelect(select, options, selected) {
  const current = selected;
  select.replaceChildren();
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.label;
    select.appendChild(el);
  }
  if ([...select.options].some((o) => o.value === current)) {
    select.value = current;
  }
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
  document.getElementById("microphone-effective").textContent = state.microphone
    ?.effective
    ? `effective: ${state.microphone.effective}`
    : "";
  document.getElementById("audio-effective").textContent = state.audio_stream?.effective
    ? `effective: ${state.audio_stream.effective}`
    : "";
}

function applyState(state) {
  applying = true;
  document.getElementById("mic-mute").checked = Boolean(state.microphone_muted);
  document.getElementById("audio-mute").checked = Boolean(state.audio_stream_muted);
  document.getElementById("tts-enabled").checked = Boolean(state.tts_enabled);
  document.getElementById("response-policy").value = state.response_policy || "both";
  document.getElementById("tts-provider").value = state.tts_provider || "piper";
  document.getElementById("codex-model").value = state.codex_model || "";
  document.getElementById("codex-reasoning").value = state.codex_reasoning || "";
  document.getElementById("turn-silence").value = state.turn_silence ?? 3;
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
  document.getElementById("microphone-effective").textContent = state.microphone
    ?.effective
    ? `effective: ${state.microphone.effective}`
    : "";
  document.getElementById("audio-effective").textContent = state.audio_stream?.effective
    ? `effective: ${state.audio_stream.effective}`
    : "";
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
    void dispatch("turn_silence.set", {
      seconds: Number(e.target.value),
    });
  });

  document.getElementById("interrupt").addEventListener("click", () => {
    void dispatch("session.interrupt", {});
  });
  document.getElementById("end-turn").addEventListener("click", () => {
    void dispatch("voice.end_turn", {});
  });
  document.getElementById("new-session").addEventListener("click", () => {
    void dispatch("session.new", {});
  });
  document.getElementById("save-transcript").addEventListener("click", () => {
    void dispatch("transcript.save", {});
  });

  document.getElementById("send").addEventListener("click", async () => {
    const text = document.getElementById("compose-text").value;
    const fileInput = document.getElementById("compose-image");
    attachmentIds = [];
    if (fileInput.files && fileInput.files[0]) {
      const buffer = await fileInput.files[0].arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let binary = "";
      for (let i = 0; i < bytes.length; i += 1) {
        binary += String.fromCharCode(bytes[i]);
      }
      const data = btoa(binary);
      try {
        const outcome = await api.dispatch("attachment.upload", { data });
        if (outcome && outcome.effective) {
          attachmentIds = [outcome.effective];
        }
      } catch (error) {
        setBanner(error.message, true);
        return;
      }
    }
    await dispatch("message.send", {
      text,
      images: attachmentIds,
      respond: true,
    });
    document.getElementById("compose-text").value = "";
    fileInput.value = "";
    attachmentIds = [];
  });
}

async function boot() {
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
  });
  api.onTranscriptEvent((event) => {
    if (event === null || typeof event !== "object") {
      return;
    }
    applyTranscriptDomEvent(transcriptList, document, event);
  });
  try {
    const snapshot = await api.snapshot();
    await refreshDevices(snapshot.state);
    applyState(snapshot.state);
    if (Array.isArray(snapshot.transcript)) {
      renderTranscriptSnapshot(transcriptList, document, snapshot.transcript);
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
