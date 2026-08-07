/**
 * DOM helpers for the live transcript (#102).
 *
 * Untrusted text must only reach the page via textContent / createElement —
 * never innerHTML. Semgrep bans innerHTML under electron/renderer/.
 */

/**
 * @param {Document} document
 * @param {{ id: number, entry: Record<string, unknown> }} row
 * @returns {HTMLElement}
 */
export function buildTranscriptRowElement(document, row) {
  const entry = row.entry ?? {};
  const article = document.createElement("article");
  article.className = "transcript-row";
  article.dataset.id = String(row.id);

  const meta = document.createElement("div");
  meta.className = "transcript-meta";
  const kind = document.createElement("span");
  kind.className = "transcript-kind";
  kind.textContent = typeof entry.kind === "string" ? entry.kind : "";
  const source = document.createElement("span");
  source.className = "transcript-source";
  source.textContent = typeof entry.source === "string" ? entry.source : "";
  meta.appendChild(kind);
  meta.appendChild(source);

  const body = document.createElement("div");
  body.className = "transcript-text";
  body.textContent = typeof entry.text === "string" ? entry.text : "";

  article.appendChild(meta);
  article.appendChild(body);

  if (Array.isArray(entry.output)) {
    for (const line of entry.output) {
      if (typeof line !== "string") {
        continue;
      }
      const out = document.createElement("pre");
      out.className = "transcript-output";
      out.textContent = line;
      article.appendChild(out);
    }
  }
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
 * Show the live recognition line from AppState partial fields.
 * @param {HTMLElement} el
 * @param {{ partial_source?: string, partial_text?: string }} state
 */
export function renderPartialLine(el, state) {
  const source = typeof state.partial_source === "string" ? state.partial_source : "";
  const text = typeof state.partial_text === "string" ? state.partial_text : "";
  if (!text) {
    el.replaceChildren();
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const doc = el.ownerDocument;
  const label = doc.createElement("span");
  label.className = "partial-source";
  label.textContent = source ? `${source}: ` : "";
  const body = doc.createElement("span");
  body.className = "partial-text";
  body.textContent = text;
  el.replaceChildren(label, body);
}
