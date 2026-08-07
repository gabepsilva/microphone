/**
 * The Piper voice catalog the Speech panel picker is drawn from.
 *
 * Which voices the session will accept and which are already on disk is the
 * CLI's answer (`speech.catalog` over the socket), not this client's guess —
 * hard-coding the shortlist here would drift from `tagalong/piper_voices.py`.
 */

/**
 * @typedef {object} SpeechVoice
 * @property {string} id
 * @property {string} label
 * @property {boolean} downloaded
 */

/**
 * Read a `speech.catalog` reply, dropping rows that cannot be drawn.
 * @param {unknown} value
 * @returns {SpeechVoice[]}
 */
export function parseSpeechCatalog(value) {
  const rows =
    value !== null &&
    typeof value === "object" &&
    Array.isArray(/** @type {Record<string, unknown>} */ (value).voices)
      ? /** @type {Record<string, unknown>} */ (value).voices
      : value;
  if (!Array.isArray(rows)) {
    return [];
  }
  const voices = [];
  for (const row of rows) {
    if (row === null || typeof row !== "object") {
      continue;
    }
    const record = /** @type {Record<string, unknown>} */ (row);
    if (typeof record.id !== "string" || record.id === "") {
      continue;
    }
    const label =
      typeof record.label === "string" && record.label !== ""
        ? record.label
        : record.id;
    voices.push({
      id: record.id,
      label,
      downloaded: Boolean(record.downloaded),
    });
  }
  return voices;
}

/**
 * Options for the voice picker, with the running voice guaranteed present.
 * @param {SpeechVoice[]} voices
 * @param {string} current
 * @returns {{ value: string, label: string }[]}
 */
export function voiceOptionsIncluding(voices, current) {
  const options = voices.map((voice) => ({
    value: voice.id,
    label: voice.downloaded ? voice.label : `${voice.label} (download)`,
  }));
  if (current && !options.some((option) => option.value === current)) {
    options.unshift({ value: current, label: current });
  }
  return options;
}
