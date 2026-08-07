export type SelectionLike = {
  desired?: string | null;
  effective?: string | null;
};

export type VoicePickerState = {
  tts_provider?: SelectionLike | string;
  tts_voice?: SelectionLike;
  piper_voice?: string;
};

export type SpeechVoice = {
  id: string;
  label: string;
  downloaded: boolean;
};

export function selectionEffectiveText(selection: SelectionLike | undefined): string;

export function selectedVoiceId(state: VoicePickerState): string;

export function voiceEffectiveText(state: VoicePickerState): string;

export function selectedProviderId(state: VoicePickerState): string;

export function effectiveProviderId(state: VoicePickerState): string;

export function voicePickerActive(state: VoicePickerState): boolean;

export function syncVoicePicker(
  field: { hidden: boolean },
  select: {
    disabled: boolean;
    replaceChildren: (...nodes: object[]) => void;
    appendChild: (node: object) => object;
    options: ArrayLike<{ value: string }>;
    value: string;
  },
  state: VoicePickerState,
  voices: SpeechVoice[],
  createOption: (tag: string) => { value: string; textContent: string },
  optionsFor: (
    voices: SpeechVoice[],
    current: string,
  ) => { value: string; label: string }[],
): void;

export function voiceChangePayload(
  applying: boolean,
  voice: string,
): { voice: string } | null;
