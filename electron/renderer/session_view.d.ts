import type { AppState } from "../src/state";

export const SESSION_FIELD_IDS: {
  readonly thread: "session-codex-thread";
  readonly state: "session-codex-state";
  readonly confidence: "session-confidence";
  readonly language: "session-language";
  readonly moonshine: "session-moonshine";
  readonly tokens: "session-tokens";
  readonly echoesCut: "session-echoes-cut";
};

type SessionState = Pick<
  AppState,
  | "codex_thread"
  | "codex_state"
  | "codex_speaking"
  | "confidence"
  | "language"
  | "moonshine"
  | "tokens"
  | "echoes_cut"
>;

type SessionElement = {
  textContent: string;
  classList: { toggle(name: string, force: boolean): void };
};

type SessionDocument = {
  getElementById(id: string): SessionElement;
};

export function sessionActivity(
  state: Pick<SessionState, "codex_state" | "codex_speaking">,
): string;
export function syncSession(document: SessionDocument, state: SessionState): void;
