/**
 * DOM helpers for the live transcript (#102).
 *
 * The shapes here mirror tagalong/tui.py's EntryRow: a stamp column, a
 * speaker column carrying the source palette, and a body column that owns
 * the text plus any command output or cut-off chrome. Keeping the two
 * renderers structurally alike is what lets one screenshot answer for both.
 *
 * Untrusted text must only reach the page via textContent / createElement —
 * never innerHTML. Semgrep bans innerHTML under electron/renderer/.
 */

/** Speakers the transcript colours. Mirrors tui.SOURCE_STYLES. */
const SOURCE_CLASSES = {
  Voice: "source-voice",
  Text: "source-text",
  Audio: "source-audio",
  Agent: "source-agent",
  Taga: "source-taga",
};

/** Chrome for a turn the user interrupted. Mirrors tui.CUT_OFF_LINE. */
export const CUT_OFF_LINE = "cut off: user started speaking";

/**
 * Palette class for a speaker, or the muted default for an unknown one.
 * @param {unknown} source
 * @returns {string}
 */
export function sourceClass(source) {
  if (typeof source !== "string") {
    return "source-unknown";
  }
  return SOURCE_CLASSES[source] ?? "source-unknown";
}

/**
 * The label the speaker column shows. Notes are the interface talking to
 * itself, so they carry a dot instead of a name — same as the TUI.
 * @param {Record<string, unknown>} entry
 * @returns {string}
 */
export function sourceLabel(entry) {
  if (entry.kind === "note") {
    return "·";
  }
  return typeof entry.source === "string" ? entry.source : "";
}

/**
 * The body text for a row, kind by kind. Mirrors tui.render_entry_body:
 * reasoning says only that it is thinking until the cost is known, and a
 * command reads as the shell line that produced the output beneath it.
 * @param {Record<string, unknown>} entry
 * @returns {string}
 */
export function entryBodyText(entry) {
  const text = typeof entry.text === "string" ? entry.text : "";
  if (entry.kind === "command") {
    return `$ ${text}`;
  }
  if (entry.kind === "reasoning") {
    if (entry.streaming) {
      return "thinking ▌";
    }
    const seconds = typeof entry.seconds === "number" ? entry.seconds : null;
    const head = seconds === null ? "thinking" : `thinking · ${seconds.toFixed(1)}s`;
    return text ? `${head}\n${text}` : head;
  }
  if (entry.streaming) {
    return `${text} ▌`;
  }
  return text;
}

/**
 * Trailing lines a command row shows below its output.
 * @param {Record<string, unknown>} entry
 * @returns {string[]}
 */
export function commandOutputLines(entry) {
  const lines = Array.isArray(entry.output)
    ? entry.output.filter((line) => typeof line === "string")
    : [];
  if (entry.kind === "command" && typeof entry.exit_code === "number") {
    return [...lines, `[command exit: ${entry.exit_code}]`];
  }
  return lines;
}

/**
 * @param {DomDocument} document
 * @param {string} className
 * @param {string} text
 * @param {string} [tag]
 */
function textNode(document, className, text, tag = "div") {
  const el = document.createElement(tag);
  el.className = className;
  el.textContent = text;
  return el;
}

/**
 * @param {Document} document
 * @param {{ id: number, entry: Record<string, unknown> }} row
 * @returns {HTMLElement}
 */
export function buildTranscriptRowElement(document, row) {
  const entry = row.entry ?? {};
  const article = document.createElement("article");
  article.className =
    entry.kind === "command" ? "transcript-row command" : "transcript-row";
  article.dataset.id = String(row.id);

  const stamp = textNode(
    document,
    "entry-stamp",
    typeof entry.stamp === "string" ? entry.stamp : "",
    "span",
  );
  const label = sourceLabel(entry);
  const source = textNode(
    document,
    `entry-source ${sourceClass(entry.source)}`,
    label,
    "span",
  );

  const main = document.createElement("div");
  main.className = "entry-main";

  const bodyClass =
    entry.kind === "note" || entry.kind === "reasoning"
      ? `transcript-text kind-${String(entry.kind)}`
      : "transcript-text";
  main.appendChild(textNode(document, bodyClass, entryBodyText(entry)));

  for (const line of commandOutputLines(entry)) {
    main.appendChild(textNode(document, "transcript-output", line, "pre"));
  }
  if (entry.interrupted) {
    main.appendChild(textNode(document, "entry-cutoff", `⊥ ${CUT_OFF_LINE}`));
  }

  article.appendChild(stamp);
  article.appendChild(source);
  article.appendChild(main);
  return article;
}

/**
 * Replace the list with a full accepted snapshot.
 * @param {HTMLElement} list
 * @param {Document} document
 * @param {Array<{ id: number, entry: Record<string, unknown> }>} rows
 */
export function renderTranscriptSnapshot(list, document, rows) {
  const nodes = rows.map((row) => buildTranscriptRowElement(document, row));
  list.replaceChildren(...nodes);
}

/**
 * Apply an incremental transcript event to a list already on screen.
 * @param {HTMLElement} list
 * @param {Document} document
 * @param {{ name: string, row?: { id: number, entry: Record<string, unknown> } }} event
 */
export function applyTranscriptDomEvent(list, document, event) {
  if (event.name === "transcript.cleared") {
    list.replaceChildren();
    return;
  }
  const row = event.row;
  if (row === undefined) {
    return;
  }
  const next = buildTranscriptRowElement(document, row);
  const wanted = String(row.id);
  const existing = [...list.children].find((child) => child.dataset?.id === wanted);
  if (existing !== undefined) {
    existing.replaceWith(next);
    return;
  }
  list.appendChild(next);
}

/**
 * What the line under the transcript says when nothing is being recognised.
 * Mirrors tui.VoiceCodexScreen._sync_partial: the reason the line is quiet is
 * more useful than the silence itself.
 * @param {{ microphone_muted?: boolean, audio_stream_muted?: boolean }} state
 * @returns {string}
 */
export function idlePartialText(state) {
  const micMuted = Boolean(state.microphone_muted);
  const audioMuted = Boolean(state.audio_stream_muted);
  if (micMuted && audioMuted) {
    return "mic and speaker muted, nothing transcribing";
  }
  if (micMuted) {
    return "mic muted, Audio still transcribing";
  }
  if (audioMuted) {
    return "speaker muted, mic still hot";
  }
  return "silence: mic hot, nothing pending";
}

/**
 * Show the live recognition line from AppState partial fields.
 * @param {HTMLElement} el
 * @param {{ partial_source?: string, partial_text?: string }} state
 */
export function renderPartialLine(el, state) {
  const source = typeof state.partial_source === "string" ? state.partial_source : "";
  const text = typeof state.partial_text === "string" ? state.partial_text : "";
  const doc = el.ownerDocument;
  const dot = doc.createElement("span");
  dot.className = "partial-dot";
  dot.textContent = "◌ ";
  if (!text) {
    const idle = doc.createElement("span");
    idle.className = "partial-idle";
    idle.textContent = idlePartialText(state);
    el.hidden = false;
    el.replaceChildren(dot, idle);
    return;
  }
  el.hidden = false;
  const label = doc.createElement("span");
  label.className = `partial-source ${sourceClass(source)}`;
  label.textContent = source ? `${source}  ` : "";
  const body = doc.createElement("span");
  body.className = "partial-text";
  body.textContent = text;
  el.replaceChildren(dot, label, body);
}
