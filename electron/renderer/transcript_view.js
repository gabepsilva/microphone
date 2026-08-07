/**
 * DOM helpers for the live transcript (#102).
 *
 * The transcript reads as a chat: who spoke and when sits in a small label
 * above the message, and the message itself gets the width. What a row is
 * still comes from the TUI's entry model (tagalong/tui.py render_entry_body)
 * — the same kinds, the same cut-off chrome, the same speaker palette —
 * so the two clients never disagree about what happened, only about how
 * much room it gets.
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

/** Speakers whose messages are drawn as a sent bubble rather than an answer. */
const INBOUND_SOURCES = new Set(["Voice", "Text", "Audio", "Agent"]);

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
 * How a row is laid out: a bubble from the room, an answer from Taga, or
 * full-width chrome for the interface's own notes and command echoes.
 * @param {Record<string, unknown>} entry
 * @returns {"note" | "command" | "inbound" | "answer"}
 */
export function rowLayout(entry) {
  if (entry.kind === "note") {
    return "note";
  }
  if (entry.kind === "command") {
    return "command";
  }
  return INBOUND_SOURCES.has(entry.source) ? "inbound" : "answer";
}

/**
 * The label above a message: who said it. Notes and commands are the
 * interface talking, so they name what they are instead.
 * @param {Record<string, unknown>} entry
 * @returns {string}
 */
export function sourceLabel(entry) {
  if (entry.kind === "note") {
    return "note";
  }
  if (entry.kind === "command") {
    return "command";
  }
  if (entry.kind === "reasoning") {
    return "thinking";
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
      return "thinking…";
    }
    return text;
  }
  if (entry.streaming) {
    return `${text} ▍`;
  }
  return text;
}

/**
 * The small trailing note on a row: how long the thinking took, or how the
 * command ended. Empty when there is nothing to say.
 * @param {Record<string, unknown>} entry
 * @returns {string}
 */
export function entryFootnote(entry) {
  if (entry.kind === "reasoning" && !entry.streaming) {
    return typeof entry.seconds === "number" ? `${entry.seconds.toFixed(1)}s` : "";
  }
  if (entry.kind === "command" && typeof entry.exit_code === "number") {
    return `exit ${entry.exit_code}`;
  }
  return "";
}

/**
 * Output lines a row shows below its body.
 * @param {Record<string, unknown>} entry
 * @returns {string[]}
 */
export function commandOutputLines(entry) {
  return Array.isArray(entry.output)
    ? entry.output.filter((line) => typeof line === "string")
    : [];
}

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
  const layout = rowLayout(entry);
  const article = document.createElement("article");
  article.className = `msg msg-${layout} ${sourceClass(entry.source)}`;
  article.dataset.id = String(row.id);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.appendChild(textNode(document, "msg-source", sourceLabel(entry), "span"));
  const stamp = typeof entry.stamp === "string" ? entry.stamp : "";
  if (stamp) {
    meta.appendChild(textNode(document, "msg-stamp", stamp, "span"));
  }
  const footnote = entryFootnote(entry);
  if (footnote) {
    meta.appendChild(textNode(document, "msg-footnote", footnote, "span"));
  }
  article.appendChild(meta);

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.appendChild(
    textNode(
      document,
      `msg-body kind-${String(entry.kind ?? "")}`,
      entryBodyText(entry),
    ),
  );
  for (const line of commandOutputLines(entry)) {
    bubble.appendChild(textNode(document, "msg-output", line, "pre"));
  }
  if (entry.interrupted) {
    bubble.appendChild(textNode(document, "msg-cutoff", CUT_OFF_LINE));
  }
  article.appendChild(bubble);
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
    return "mic muted, audio still transcribing";
  }
  if (audioMuted) {
    return "speaker muted, mic still hot";
  }
  return "listening — nothing pending";
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
  dot.className = text ? "partial-dot live" : "partial-dot";
  dot.textContent = "";
  el.classList?.toggle("live", Boolean(text));
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
  label.textContent = source;
  const body = doc.createElement("span");
  body.className = "partial-text";
  body.textContent = text;
  el.replaceChildren(dot, label, body);
}
