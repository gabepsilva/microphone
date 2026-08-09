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
  const thread = document.getElementById("session-codex-thread");
  thread.textContent = state.codex_thread || "—";
  const activity = sessionActivity(state);
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
