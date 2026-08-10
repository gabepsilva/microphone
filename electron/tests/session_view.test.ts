import { describe, expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  SESSION_FIELD_IDS,
  sessionActivity,
  syncSession,
} from "../renderer/session_view.js";

function documentWithSessionFields() {
  const fields = new Map<string, { textContent: string; active: boolean }>();
  for (const id of Object.values(SESSION_FIELD_IDS)) {
    fields.set(id, { textContent: "", active: false });
  }
  return {
    fields,
    document: {
      getElementById(id: string) {
        const field = fields.get(id);
        if (field === undefined) {
          throw new Error(`missing session field: ${id}`);
        }
        return {
          get textContent() {
            return field.textContent;
          },
          set textContent(value: string) {
            field.textContent = value;
          },
          classList: {
            toggle(_name: string, active: boolean) {
              field.active = active;
            },
          },
        };
      },
    },
  };
}

describe("session markup", () => {
  it("contains each renderer field exactly once", () => {
    const html = readFileSync(join(__dirname, "../renderer/index.html"), "utf8");
    for (const id of Object.values(SESSION_FIELD_IDS)) {
      expect(html.match(new RegExp(`id="${id}"`, "g")) ?? []).toHaveLength(1);
    }
  });
});

describe("sessionActivity", () => {
  it("keeps a specific Codex state ahead of speech", () => {
    expect(sessionActivity({ codex_state: "thinking", codex_speaking: true })).toBe(
      "thinking",
    );
  });

  it("shows speech after the stream becomes idle", () => {
    expect(sessionActivity({ codex_state: "idle", codex_speaking: true })).toBe(
      "speaking",
    );
    expect(sessionActivity({ codex_state: "idle", codex_speaking: false })).toBe(
      "idle",
    );
  });
});

describe("syncSession", () => {
  it("renders the live TUI session fields", () => {
    const view = documentWithSessionFields();
    syncSession(view.document, {
      codex_thread: "thread-9",
      codex_state: "thinking",
      codex_speaking: false,
      confidence: 0.83,
      language: "fr",
      moonshine: "small-streaming",
      tokens: 42000,
      echoes_cut: 3,
    });

    expect(view.fields.get("session-codex-thread")?.textContent).toBe("thread-9");
    expect(view.fields.get("session-codex-state")?.textContent).toBe("thinking");
    expect(view.fields.get("session-codex-state")?.active).toBe(true);
    expect(view.fields.get("session-confidence")?.textContent).toBe("0.83");
    expect(view.fields.get("session-language")?.textContent).toBe("fr");
    expect(view.fields.get("session-moonshine")?.textContent).toBe("small-streaming");
    expect(view.fields.get("session-tokens")?.textContent).toBe("42,000");
    expect(view.fields.get("session-echoes-cut")?.textContent).toBe("3");
  });
});
