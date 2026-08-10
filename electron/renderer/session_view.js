export const SESSION_FIELD_IDS = {
  thread: "session-codex-thread",
  state: "session-codex-state",
  confidence: "session-confidence",
  language: "session-language",
  moonshine: "session-moonshine",
  tokens: "session-tokens",
  echoesCut: "session-echoes-cut",
};

/** Return the activity label used by the TUI's Codex status line. */
export function sessionActivity(state) {
  const codexState = state.codex_state || "idle";
  return codexState !== "idle"
    ? codexState
    : state.codex_speaking
      ? "speaking"
      : "idle";
}

/** Draw the live session details in the sidebar. */
export function syncSession(document, state) {
  const thread = document.getElementById(SESSION_FIELD_IDS.thread);
  thread.textContent = state.codex_thread || "—";
  const activity = sessionActivity(state);
  const activityElement = document.getElementById(SESSION_FIELD_IDS.state);
  activityElement.textContent = activity;
  activityElement.classList.toggle("active", activity !== "idle");
  const confidence = Number(state.confidence);
  document.getElementById(SESSION_FIELD_IDS.confidence).textContent = Number.isFinite(
    confidence,
  )
    ? confidence.toFixed(2)
    : "—";
  document.getElementById(SESSION_FIELD_IDS.language).textContent =
    state.language || "—";
  document.getElementById(SESSION_FIELD_IDS.moonshine).textContent =
    state.moonshine || "—";
  const tokens = Number(state.tokens);
  document.getElementById(SESSION_FIELD_IDS.tokens).textContent = Number.isFinite(
    tokens,
  )
    ? tokens.toLocaleString("en-US")
    : "—";
  const echoesCut = Number(state.echoes_cut);
  document.getElementById(SESSION_FIELD_IDS.echoesCut).textContent = Number.isFinite(
    echoesCut,
  )
    ? String(echoesCut)
    : "—";
}
