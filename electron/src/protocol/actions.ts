/** Generated from tagalong.control.actions.CATALOG. Do not edit by hand. */
export const ACTIONS = {
  message_send: "message.send",
  attachment_upload: "attachment.upload",
  transcript_append: "transcript.append",
  session_new: "session.new",
  session_interrupt: "session.interrupt",
  session_quit: "session.quit",
  voice_end_turn: "voice.end_turn",
  microphone_select: "microphone.select",
  microphone_set_muted: "microphone.set_muted",
  audio_stream_select: "audio_stream.select",
  audio_stream_set_muted: "audio_stream.set_muted",
  response_policy_set: "response_policy.set",
  tts_set_enabled: "tts.set_enabled",
  tts_set_provider: "tts.set_provider",
  tts_set_voice: "tts.set_voice",
  speech_read_selection: "speech.read_selection",
  codex_set_model: "codex.set_model",
  codex_set_reasoning: "codex.set_reasoning",
  turn_silence_set: "turn_silence.set",
  transcript_save: "transcript.save",
} as const;

export type ActionId = (typeof ACTIONS)[keyof typeof ACTIONS];
