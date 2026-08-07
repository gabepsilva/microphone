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
  tts_provider: Selection;
  tts_voice: Selection;
  piper_voice: string;
  edge_voice: string;
  codex_model: string;
  codex_reasoning: string;
  turn_silence: number;
  /** Live recognition line (#102 Q3a). */
  partial_source: string;
  partial_text: string;
};

/** Every AppState key — used to pin applyStateFragment exhaustiveness. */
export const APP_STATE_KEYS = [
  "microphone",
  "microphone_muted",
  "audio_stream",
  "audio_stream_muted",
  "response_policy",
  "tts_enabled",
  "tts_provider",
  "tts_voice",
  "piper_voice",
  "edge_voice",
  "codex_model",
  "codex_reasoning",
  "turn_silence",
  "partial_source",
  "partial_text",
] as const satisfies ReadonlyArray<keyof AppState>;

// Compile-time exhaustiveness: every AppState key must appear in APP_STATE_KEYS
// so applyStateFragment's key-coverage test cannot miss a new field (#102).
type MissingAppStateKeys = Exclude<keyof AppState, (typeof APP_STATE_KEYS)[number]>;
const _appStateKeysExhaustive: MissingAppStateKeys extends never ? true : never = true;
void _appStateKeysExhaustive;

export type TranscriptEntry = {
  kind: string;
  source: string;
  text: string;
  stamp: string;
  reply_to: string;
  interrupted: boolean;
  output: string[];
  exit_code: number | null;
  streaming: boolean;
  seconds: number | null;
};

export type TranscriptRow = {
  id: number;
  entry: TranscriptEntry;
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
    tts_provider: { desired: "piper", effective: "piper" },
    tts_voice: emptySelection(),
    piper_voice: "en_US-lessac-medium",
    edge_voice: "en-US-AndrewNeural",
    codex_model: "",
    codex_reasoning: "",
    turn_silence: 3.0,
    partial_source: "",
    partial_text: "",
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
    tts_voice: { ...state.tts_voice },
  };
  for (const [name, value] of Object.entries(changed)) {
    switch (name) {
      case "microphone": {
        const selection = asSelection(value);
        if (selection !== null) {
          next.microphone = selection;
        }
        break;
      }
      case "audio_stream": {
        const selection = asSelection(value);
        if (selection !== null) {
          next.audio_stream = selection;
        }
        break;
      }
      case "microphone_muted":
        if (typeof value === "boolean") {
          next.microphone_muted = value;
        }
        break;
      case "audio_stream_muted":
        if (typeof value === "boolean") {
          next.audio_stream_muted = value;
        }
        break;
      case "tts_enabled":
        if (typeof value === "boolean") {
          next.tts_enabled = value;
        }
        break;
      case "response_policy":
        if (typeof value === "string") {
          next.response_policy = value;
        }
        break;
      case "tts_provider": {
        const selection = asSelection(value);
        if (selection !== null) {
          next.tts_provider = selection;
        }
        break;
      }
      case "tts_voice": {
        const selection = asSelection(value);
        if (selection !== null) {
          next.tts_voice = selection;
        }
        break;
      }
      case "piper_voice":
        if (typeof value === "string") {
          next.piper_voice = value;
        }
        break;
      case "edge_voice":
        if (typeof value === "string") {
          next.edge_voice = value;
        }
        break;
      case "codex_model":
        if (typeof value === "string") {
          next.codex_model = value;
        }
        break;
      case "codex_reasoning":
        if (typeof value === "string") {
          next.codex_reasoning = value;
        }
        break;
      case "partial_source":
        if (typeof value === "string") {
          next.partial_source = value;
        }
        break;
      case "partial_text":
        if (typeof value === "string") {
          next.partial_text = value;
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

export function parseTranscriptRows(value: unknown): TranscriptRow[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const rows: TranscriptRow[] = [];
  for (const item of value) {
    if (item === null || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    const id = record.id;
    const entry = record.entry;
    if (typeof id !== "number" || !Number.isFinite(id)) {
      continue;
    }
    if (entry === null || typeof entry !== "object") {
      continue;
    }
    const fields = entry as Record<string, unknown>;
    rows.push({
      id,
      entry: {
        kind: typeof fields.kind === "string" ? fields.kind : "",
        source: typeof fields.source === "string" ? fields.source : "",
        text: typeof fields.text === "string" ? fields.text : "",
        stamp: typeof fields.stamp === "string" ? fields.stamp : "",
        reply_to: typeof fields.reply_to === "string" ? fields.reply_to : "",
        interrupted: Boolean(fields.interrupted),
        output: Array.isArray(fields.output)
          ? fields.output.filter((line): line is string => typeof line === "string")
          : [],
        exit_code: typeof fields.exit_code === "number" ? fields.exit_code : null,
        streaming: Boolean(fields.streaming),
        seconds: typeof fields.seconds === "number" ? fields.seconds : null,
      },
    });
  }
  return rows;
}
