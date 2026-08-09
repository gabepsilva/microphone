import type { AppState } from "../src/state";

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
