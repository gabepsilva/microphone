import { ACTIONS, type ActionId } from "./actions";

/**
 * Actions the renderer may invoke through the single dispatch channel.
 *
 * ``session.quit`` stays out: socket agents are refused it by policy, and the
 * Electron UI must not offer a shutdown the peer cannot perform (#96).
 */
export const DISPATCH_ALLOWLIST = [
  ACTIONS.message_send,
  ACTIONS.attachment_upload,
  ACTIONS.transcript_append,
  ACTIONS.session_new,
  ACTIONS.session_interrupt,
  ACTIONS.voice_end_turn,
  ACTIONS.microphone_select,
  ACTIONS.microphone_set_muted,
  ACTIONS.audio_stream_select,
  ACTIONS.audio_stream_set_muted,
  ACTIONS.response_policy_set,
  ACTIONS.tts_set_enabled,
  ACTIONS.tts_set_provider,
  ACTIONS.codex_set_model,
  ACTIONS.codex_set_reasoning,
  ACTIONS.turn_silence_set,
  ACTIONS.transcript_save,
] as const satisfies readonly ActionId[];

export type AllowedActionId = (typeof DISPATCH_ALLOWLIST)[number];

const ALLOWED = new Set<string>(DISPATCH_ALLOWLIST);

export function isAllowedAction(action: string): action is AllowedActionId {
  return ALLOWED.has(action);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Validate action id + payload shape before the socket sees them. */
export function validateDispatch(
  action: unknown,
  payload: unknown,
): { action: AllowedActionId; payload: Record<string, unknown> } {
  if (typeof action !== "string" || !isAllowedAction(action)) {
    throw new Error(`action not allowed: ${String(action)}`);
  }
  if (!isPlainObject(payload)) {
    throw new Error("payload must be an object");
  }
  switch (action) {
    case ACTIONS.tts_set_enabled:
      requireBoolean(payload, "enabled");
      break;
    case ACTIONS.microphone_set_muted:
    case ACTIONS.audio_stream_set_muted:
      requireBoolean(payload, "muted");
      break;
    case ACTIONS.microphone_select:
    case ACTIONS.audio_stream_select:
      requireNullableString(payload, "name");
      break;
    case ACTIONS.response_policy_set:
      requireString(payload, "policy");
      break;
    case ACTIONS.tts_set_provider:
      requireString(payload, "provider");
      break;
    case ACTIONS.codex_set_model:
      requireString(payload, "model");
      break;
    case ACTIONS.codex_set_reasoning:
      requireString(payload, "effort");
      break;
    case ACTIONS.turn_silence_set:
      requireNumber(payload, "seconds");
      break;
    case ACTIONS.message_send:
      requireString(payload, "text");
      break;
    case ACTIONS.attachment_upload:
      requireString(payload, "data");
      break;
    case ACTIONS.transcript_append:
      requireString(payload, "text");
      break;
    default:
      break;
  }
  return { action, payload };
}

function requireBoolean(payload: Record<string, unknown>, key: string): void {
  if (typeof payload[key] !== "boolean") {
    throw new Error(`${key} must be a boolean`);
  }
}

function requireString(payload: Record<string, unknown>, key: string): void {
  if (typeof payload[key] !== "string") {
    throw new Error(`${key} must be a string`);
  }
}

function requireNullableString(payload: Record<string, unknown>, key: string): void {
  const value = payload[key];
  if (!(value === null || typeof value === "string")) {
    throw new Error(`${key} must be a string or null`);
  }
}

function requireNumber(payload: Record<string, unknown>, key: string): void {
  if (typeof payload[key] !== "number" || !Number.isFinite(payload[key])) {
    throw new Error(`${key} must be a number`);
  }
}
