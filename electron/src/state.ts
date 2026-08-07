/** Mirrors AppState fields shipped on the wire via snapshot / state.changed. */

export type Selection = {
  desired: string | null;
  effective: string | null;
};

export type AppState = {
  microphone: Selection;
  microphone_muted: boolean;
  audio_stream: Selection;
  audio_stream_muted: boolean;
  response_policy: string;
  tts_enabled: boolean;
  tts_provider: string;
  codex_model: string;
  codex_reasoning: string;
  turn_silence: number;
};

export function emptySelection(): Selection {
  return { desired: null, effective: null };
}

export function emptyAppState(): AppState {
  return {
    microphone: emptySelection(),
    microphone_muted: false,
    audio_stream: emptySelection(),
    audio_stream_muted: false,
    response_policy: "both",
    tts_enabled: true,
    tts_provider: "piper",
    codex_model: "",
    codex_reasoning: "",
    turn_silence: 3.0,
  };
}

function asSelection(value: unknown): Selection | null {
  if (value === null || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const desired = record.desired;
  const effective = record.effective;
  if (
    !(desired === null || typeof desired === "string") ||
    !(effective === null || typeof effective === "string")
  ) {
    return null;
  }
  return { desired, effective };
}

/** Merge a ``state.changed`` payload into a local AppState copy. */
export function applyStateFragment(
  state: AppState,
  changed: Record<string, unknown>,
): AppState {
  const next: AppState = {
    ...state,
    microphone: { ...state.microphone },
    audio_stream: { ...state.audio_stream },
  };
  for (const [name, value] of Object.entries(changed)) {
    switch (name) {
      case "microphone":
      case "audio_stream": {
        const selection = asSelection(value);
        if (selection !== null) {
          next[name] = selection;
        }
        break;
      }
      case "microphone_muted":
      case "audio_stream_muted":
      case "tts_enabled":
        if (typeof value === "boolean") {
          next[name] = value;
        }
        break;
      case "response_policy":
      case "tts_provider":
      case "codex_model":
      case "codex_reasoning":
        if (typeof value === "string") {
          next[name] = value;
        }
        break;
      case "turn_silence":
        if (typeof value === "number" && Number.isFinite(value)) {
          next.turn_silence = value;
        }
        break;
      default:
        break;
    }
  }
  return next;
}

/**
 * Validate a whole snapshot ``state`` object.
 *
 * Missing or invalid fields fall back to {@link emptyAppState} defaults so a
 * bad subscribe result cannot poison later ``applyStateFragment`` merges.
 */
export function parseAppState(value: unknown): AppState {
  const base = emptyAppState();
  if (value === null || typeof value !== "object") {
    return base;
  }
  return applyStateFragment(base, value as Record<string, unknown>);
}
