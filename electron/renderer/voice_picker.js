/**
 * Piper voice `<select>` drawing for the Speech panel.
 *
 * Kept out of `app.js` so the hide/show, selection precedence, and effective
 * readout can be pinned without booting the whole renderer.
 */

/**
 * @typedef {import("./speech_catalog.js").SpeechVoice} SpeechVoice
 * @typedef {{ desired?: string | null, effective?: string | null }} Selection
 * @typedef {{
 *   tts_provider?: Selection | string,
 *   tts_voice?: Selection,
 *   piper_voice?: string,
 * }} VoicePickerState
 */

/**
 * Which id the picker should show as selected.
 * @param {VoicePickerState} state
 * @returns {string}
 */
export function selectedVoiceId(state) {
  return (
    state.tts_voice?.desired ?? state.tts_voice?.effective ?? state.piper_voice ?? ""
  );
}

/**
 * Effective-voice footnote when it diverges from the desired pick.
 * @param {VoicePickerState} state
 * @returns {string}
 */
export function voiceEffectiveText(state) {
  const desired = state.tts_voice?.desired ?? "";
  const effective = state.tts_voice?.effective ?? "";
  if (!effective || effective === desired) {
    return "";
  }
  return `effective: ${effective}`;
}

/**
 * Running / requested engine id from a Selection or legacy string.
 * @param {VoicePickerState} state
 * @returns {string}
 */
export function selectedProviderId(state) {
  const provider = state.tts_provider;
  if (provider && typeof provider === "object") {
    return provider.desired ?? provider.effective ?? "piper";
  }
  return provider || "piper";
}

/**
 * Whether the Piper voice field should be interactive.
 * @param {VoicePickerState} state
 * @returns {boolean}
 */
export function voicePickerActive(state) {
  return selectedProviderId(state) === "piper";
}

/**
 * Draw the voice field from session state and the last speech.catalog reply.
 * @param {{ hidden: boolean }} field
 * @param {{
 *   disabled: boolean,
 *   replaceChildren: (...nodes: object[]) => void,
 *   appendChild: (node: object) => object,
 *   options: ArrayLike<{ value: string }>,
 *   value: string,
 * }} select
 * @param {VoicePickerState} state
 * @param {SpeechVoice[]} voices
 * @param {(tag: string) => { value: string, textContent: string }} createOption
 * @param {(voices: SpeechVoice[], current: string) => { value: string, label: string }[]}
 *   optionsFor
 */
export function syncVoicePicker(
  field,
  select,
  state,
  voices,
  createOption,
  optionsFor,
) {
  const active = voicePickerActive(state);
  field.hidden = !active;
  select.disabled = !active;
  if (!active) {
    return;
  }
  const selected = selectedVoiceId(state);
  const options = optionsFor(voices, selected);
  select.replaceChildren();
  for (const opt of options) {
    const el = createOption("option");
    el.value = opt.value;
    el.textContent = opt.label;
    select.appendChild(el);
  }
  if ([...select.options].some((o) => o.value === selected)) {
    select.value = selected;
  }
}

/**
 * A change from applyState must not re-dispatch the value it just drew.
 * @param {boolean} applying
 * @param {string} voice
 * @returns {{ voice: string } | null}
 */
export function voiceChangePayload(applying, voice) {
  if (applying) {
    return null;
  }
  return { voice };
}
