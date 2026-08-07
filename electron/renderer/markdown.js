/**
 * A small Markdown reader for Taga's answers.
 *
 * The TUI mounts Textual's Markdown for a finished Taga turn
 * (tui.uses_markdown_body); this is the renderer's equivalent. It is
 * deliberately not a full CommonMark implementation — it covers what a model
 * actually writes into a transcript (fenced code, headings, lists, quotes,
 * tables, and inline emphasis/code/links) and treats anything it does not
 * recognise as literal text.
 *
 * The parse is separate from the DOM build for one reason: every node is
 * created with createElement and filled with textContent, so model output
 * can never become markup. Semgrep bans innerHTML under electron/renderer/,
 * and this file is exactly the place someone would reach for it.
 */

const FENCE = /^\s*(?:```|~~~)\s*([\w+-]*)\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const RULE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
const QUOTE = /^\s*>\s?(.*)$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const TABLE_DIVIDER = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/;

/**
 * @typedef {{ type: "text" | "code" | "strong" | "em", text: string }
 *   | { type: "link", text: string, url: string }} Span
 */

// Emphasis needs its delimiters tight against the text (`2 * 3 * 4` is
// arithmetic, not italics) and `_` needs word boundaries as well, or every
// snake_case identifier in an answer turns into prose.
const INLINE = new RegExp(
  [
    "(`+)([^`]+?)\\1",
    "\\*\\*(?!\\s)([^*]+?)(?<!\\s)\\*\\*",
    "(?<![A-Za-z0-9_])__(?!\\s)([^_]+?)(?<!\\s)__(?![A-Za-z0-9_])",
    "\\*(?!\\s)([^*\\n]+?)(?<!\\s)\\*",
    "(?<![A-Za-z0-9_])_(?!\\s)([^_\\n]+?)(?<!\\s)_(?![A-Za-z0-9_])",
    "\\[([^\\]\\n]*?)\\]\\(([^)\\s]+)\\)",
  ].join("|"),
  "g",
);

/**
 * Split one line of Markdown into styled spans.
 * @param {string} text
 * @returns {Span[]}
 */
export function parseInline(text) {
  const spans = [];
  let last = 0;
  INLINE.lastIndex = 0;
  for (let match = INLINE.exec(text); match !== null; match = INLINE.exec(text)) {
    if (match.index > last) {
      spans.push({ type: "text", text: text.slice(last, match.index) });
    }
    const [, , code, strongStar, strongScore, emStar, emScore, linkText, url] = match;
    if (code !== undefined) {
      spans.push({ type: "code", text: code });
    } else if (strongStar !== undefined || strongScore !== undefined) {
      spans.push({ type: "strong", text: strongStar ?? strongScore ?? "" });
    } else if (emStar !== undefined || emScore !== undefined) {
      spans.push({ type: "em", text: emStar ?? emScore ?? "" });
    } else {
      spans.push({ type: "link", text: linkText || (url ?? ""), url: url ?? "" });
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    spans.push({ type: "text", text: text.slice(last) });
  }
  return spans;
}

/** Split a table row on unescaped pipes, dropping the outer ones. */
function tableCells(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

/**
 * @typedef {{ type: "code", lang: string, text: string }
 *   | { type: "heading", level: number, text: string }
 *   | { type: "paragraph", text: string }
 *   | { type: "quote", text: string }
 *   | { type: "list", ordered: boolean, items: string[] }
 *   | { type: "table", header: string[], rows: string[][] }
 *   | { type: "rule" }} Block
 */

/**
 * Read Markdown into blocks. Unrecognised input becomes paragraph text.
 * @param {string} source
 * @returns {Block[]}
 */
export function parseMarkdown(source) {
  const lines = String(source ?? "").split("\n");
  const blocks = [];
  let paragraph = [];

  const flush = () => {
    if (paragraph.length > 0) {
      blocks.push({ type: "paragraph", text: paragraph.join("\n") });
      paragraph = [];
    }
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const fence = FENCE.exec(line);
    if (fence !== null) {
      flush();
      const body = [];
      i += 1;
      // An unclosed fence runs to the end: a streaming answer is cut mid-block
      // far more often than a model forgets to close one.
      for (; i < lines.length && FENCE.exec(lines[i]) === null; i += 1) {
        body.push(lines[i]);
      }
      blocks.push({ type: "code", lang: fence[1] ?? "", text: body.join("\n") });
      continue;
    }
    if (line.trim() === "") {
      flush();
      continue;
    }
    if (RULE.test(line)) {
      flush();
      blocks.push({ type: "rule" });
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading !== null) {
      flush();
      blocks.push({
        type: "heading",
        level: (heading[1] ?? "#").length,
        text: heading[2] ?? "",
      });
      continue;
    }
    if (QUOTE.test(line)) {
      flush();
      const body = [];
      for (; i < lines.length; i += 1) {
        const quoted = QUOTE.exec(lines[i]);
        if (quoted === null) {
          break;
        }
        body.push(quoted[1] ?? "");
      }
      i -= 1;
      blocks.push({ type: "quote", text: body.join("\n") });
      continue;
    }
    const bullet = BULLET.exec(line);
    const numbered = NUMBERED.exec(line);
    if (bullet !== null || numbered !== null) {
      flush();
      const ordered = bullet === null;
      const items = [];
      for (; i < lines.length; i += 1) {
        const item = ordered ? NUMBERED.exec(lines[i]) : BULLET.exec(lines[i]);
        if (item === null) {
          break;
        }
        items.push(item[1] ?? "");
      }
      i -= 1;
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    // A table needs its divider on the next line, else these are just pipes.
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      lines[i + 1].includes("-") &&
      TABLE_DIVIDER.test(lines[i + 1])
    ) {
      flush();
      const header = tableCells(line);
      const rows = [];
      i += 2;
      for (; i < lines.length && lines[i].includes("|"); i += 1) {
        rows.push(tableCells(lines[i]));
      }
      i -= 1;
      blocks.push({ type: "table", header, rows });
      continue;
    }
    paragraph.push(line);
  }
  flush();
  return blocks;
}

/**
 * Only these schemes become a real anchor. A link the model wrote is still
 * untrusted input, so `javascript:` and friends stay plain text.
 *
 * Reject whitespace and controls after trim: a prefix check alone would
 * accept `https://a.com\\njavascript:…`, which matters once an
 * open-externally channel starts using this gate.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isSafeUrl(url) {
  const value = String(url ?? "").trim();
  if (value === "" || /[\s\u0000-\u001f\u007f]/.test(value)) {
    return false;
  }
  return /^(?:https?:\/\/|mailto:)/i.test(value);
}

function appendInline(document, host, text) {
  for (const span of parseInline(text)) {
    if (span.type === "text") {
      host.appendChild(document.createTextNode(span.text));
      continue;
    }
    const { tag, className } = {
      code: { tag: "code", className: "md-code-inline" },
      strong: { tag: "strong", className: "md-strong" },
      em: { tag: "em", className: "md-em" },
      link: { tag: "a", className: "md-link" },
    }[span.type];
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = span.text;
    if (span.type === "link") {
      // No href: nothing in this window may navigate away from the app. The
      // target is shown on hover until an open-externally channel exists.
      el.title = isSafeUrl(span.url) ? span.url : "";
      if (!isSafeUrl(span.url)) {
        el.className = "md-link md-link-blocked";
      }
    }
    host.appendChild(el);
  }
}

function appendCells(document, row, cells, tag) {
  for (const cell of cells) {
    const el = document.createElement(tag);
    el.className = "md-cell";
    appendInline(document, el, cell);
    row.appendChild(el);
  }
}

/**
 * Build one block's element.
 * @param {Document} document
 * @param {Block} block
 * @returns {HTMLElement}
 */
export function buildBlockElement(document, block) {
  if (block.type === "code") {
    const pre = document.createElement("pre");
    pre.className = "md-code";
    if (block.lang) {
      pre.dataset.lang = block.lang;
    }
    const code = document.createElement("code");
    code.textContent = block.text;
    pre.appendChild(code);
    return pre;
  }
  if (block.type === "heading") {
    const el = document.createElement(`h${Math.min(block.level + 2, 6)}`);
    el.className = `md-heading md-h${block.level}`;
    appendInline(document, el, block.text);
    return el;
  }
  if (block.type === "rule") {
    const el = document.createElement("hr");
    el.className = "md-rule";
    return el;
  }
  if (block.type === "quote") {
    const el = document.createElement("blockquote");
    el.className = "md-quote";
    appendInline(document, el, block.text);
    return el;
  }
  if (block.type === "list") {
    const list = document.createElement(block.ordered ? "ol" : "ul");
    list.className = "md-list";
    for (const item of block.items) {
      const li = document.createElement("li");
      li.className = "md-item";
      appendInline(document, li, item);
      list.appendChild(li);
    }
    return list;
  }
  if (block.type === "table") {
    const table = document.createElement("table");
    table.className = "md-table";
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    appendCells(document, headRow, block.header, "th");
    head.appendChild(headRow);
    table.appendChild(head);
    const body = document.createElement("tbody");
    for (const cells of block.rows) {
      const row = document.createElement("tr");
      appendCells(document, row, cells, "td");
      body.appendChild(row);
    }
    table.appendChild(body);
    return table;
  }
  const el = document.createElement("p");
  el.className = "md-paragraph";
  appendInline(document, el, block.text);
  return el;
}

/**
 * Render Markdown into *host*, one element per block.
 * @param {Document} document
 * @param {HTMLElement} host
 * @param {string} source
 */
export function renderMarkdownInto(document, host, source) {
  for (const block of parseMarkdown(source)) {
    host.appendChild(buildBlockElement(document, block));
  }
}
