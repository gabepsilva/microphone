export type SpeechVoice = {
  id: string;
  label: string;
  downloaded: boolean;
};

export function parseSpeechCatalog(value: unknown): SpeechVoice[];

export function voiceOptionsIncluding(
  voices: SpeechVoice[],
  current: string,
): { value: string; label: string }[];
