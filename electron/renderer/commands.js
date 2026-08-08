/**
 * Slash-command palette matching, ported from tagalong/commands.py.
 *
 * The session owns the catalog (`commands.list` over the socket); the client
 * owns only which rows a query shows and in what order. That ranking is
 * duplicated here rather than asked for, because a keystroke cannot afford a
 * round trip — so the rules below must stay the ones in `commands.py`:
 * `command_query`, `_name_score`, `_spec_score`, `match_commands`.
 */

/**
 * @typedef {object} CommandSpec
 * @property {string} name
 * @property {string} summary
 * @property {string[]} aliases
 * @property {string | null} action_id
 */

/**
 * The filter fragment when the palette should be open, else null.
 *
 * Open only for a single-token slash query: leading `/`, no newline, no
 * whitespace after the slash. Once arguments start (`/new keep`) the menu
 * closes so free-form args are unobstructed.
 * @param {string} text
 * @returns {string | null}
 */
export function commandQuery(text) {
  if (typeof text !== "string" || !text.startsWith("/")) {
    return null;
  }
  if (text.includes("\n") || text.includes("\r")) {
    return null;
  }
  const rest = text.slice(1);
  if (/\s/.test(rest)) {
    return null;
  }
  return rest.toLowerCase();
}

/** True when every character of `query` appears in order inside `name`. */
function isSubsequence(query, name) {
  let position = 0;
  for (const character of name) {
    if (position < query.length && character === query[position]) {
      position += 1;
    }
  }
  return position === query.length;
}

/**
 * Rank how well a name matches. Lower sorts first; null is no match.
 * Tier 0 prefix, 1 substring, 2 subsequence — then earliest, then shortest.
 * @returns {number[] | null}
 */
function nameScore(name, query) {
  if (name.startsWith(query)) {
    return [0, 0, name.length];
  }
  const index = name.indexOf(query);
  if (index >= 0) {
    return [1, index, name.length];
  }
  if (isSubsequence(query, name)) {
    return [2, 0, name.length];
  }
  return null;
}

function lower(value) {
  return String(value ?? "").toLowerCase();
}

function compare(left, right) {
  for (let i = 0; i < Math.max(left.length, right.length); i += 1) {
    const a = left[i] ?? 0;
    const b = right[i] ?? 0;
    if (a !== b) {
      return a - b;
    }
  }
  return 0;
}

/**
 * Best score across a command's name and aliases, then its description.
 * @param {CommandSpec} spec
 * @param {string} query
 * @returns {number[] | null}
 */
export function specScore(spec, query) {
  let best = null;
  for (const candidate of [spec.name, ...(spec.aliases ?? [])]) {
    const score = nameScore(lower(candidate), query);
    if (score !== null && (best === null || compare(score, best) < 0)) {
      best = score;
    }
  }
  const name = String(spec.name ?? "");
  if (best !== null) {
    return [...best, name.length];
  }
  if (query !== "" && lower(spec.summary).includes(query)) {
    return [3, 0, name.length, name.length];
  }
  return null;
}

/**
 * Filter and rank a catalog for a palette query. An empty query lists
 * everything in registration order.
 * @param {CommandSpec[]} specs
 * @param {string} query
 * @returns {CommandSpec[]}
 */
export function matchCommands(specs, query) {
  const catalog = Array.isArray(specs) ? specs : [];
  if (!query) {
    return [...catalog];
  }
  return catalog
    .map((spec, index) => ({ spec, index, score: specScore(spec, query) }))
    .filter((row) => row.score !== null)
    .sort((a, b) => compare(a.score, b.score) || a.index - b.index)
    .map((row) => row.spec);
}

/**
 * Validate one `commands.list` row. A malformed row is dropped rather than
 * rendered half-built.
 * @param {unknown} value
 * @returns {CommandSpec | null}
 */
export function parseCommandSpec(value) {
  if (value === null || typeof value !== "object") {
    return null;
  }
  const record = /** @type {Record<string, unknown>} */ (value);
  if (typeof record.name !== "string" || record.name === "") {
    return null;
  }
  return {
    name: record.name,
    summary: typeof record.summary === "string" ? record.summary : "",
    aliases: Array.isArray(record.aliases)
      ? record.aliases.filter((alias) => typeof alias === "string")
      : [],
    action_id: typeof record.action_id === "string" ? record.action_id : null,
  };
}

/**
 * Read a `commands.list` reply into specs, tolerating either the bare array
 * or the `{commands: [...]}` envelope the socket sends.
 * @param {unknown} value
 * @returns {CommandSpec[]}
 */
export function parseCommandList(value) {
  const rows =
    Array.isArray(value) ||
    value === null ||
    typeof value !== "object" ||
    !Array.isArray(/** @type {Record<string, unknown>} */ (value).commands)
      ? value
      : /** @type {Record<string, unknown>} */ (value).commands;
  if (!Array.isArray(rows)) {
    return [];
  }
  const specs = [];
  for (const row of rows) {
    const spec = parseCommandSpec(row);
    if (spec !== null) {
      specs.push(spec);
    }
  }
  return specs;
}

/** Keep a selection inside a list that just changed under it. */
export function clampIndex(index, length) {
  if (length <= 0) {
    return 0;
  }
  return ((index % length) + length) % length;
}

/**
 * Resolve a typed `/name …args` against the session catalog.
 *
 * Only the first token is the command; anything after is arguments. The
 * palette closes once arguments appear (`commandQuery`), but Enter on a
 * finished line still has to find the same entry.
 *
 * @param {CommandSpec[]} specs
 * @param {string} text
 * @returns {CommandSpec | null}
 */
export function findCommand(specs, text) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed.startsWith("/")) {
    return null;
  }
  const token = trimmed.slice(1).split(/\s+/)[0]?.toLowerCase() ?? "";
  if (token === "") {
    return null;
  }
  return (
    specs.find(
      (spec) =>
        spec.name.toLowerCase() === token ||
        spec.aliases.some((alias) => alias.toLowerCase() === token),
    ) ?? null
  );
}

/**
 * Whitespace-delimited arguments after the command name.
 *
 * @param {string} text
 * @returns {string[]}
 */
export function slashArguments(text) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed.startsWith("/")) {
    return [];
  }
  const parts = trimmed.slice(1).split(/\s+/).filter(Boolean);
  return parts.slice(1);
}

/**
 * One-line description for `/help <name>`, matching CommandSpec.detail_line.
 *
 * @param {CommandSpec} spec
 * @returns {string}
 */
export function detailLine(spec) {
  const description = spec.summary || "no description";
  const aliases = Array.isArray(spec.aliases) ? spec.aliases : [];
  const aliasPart =
    aliases.length === 0
      ? ""
      : ` (aliases: ${aliases.map((alias) => `/${alias}`).join(", ")})`;
  return `/${spec.name}${aliasPart}: ${description}`;
}

/**
 * @typedef {{ kind: "command", spec: CommandSpec, args: string[] }
 *   | { kind: "message", text: string }
 *   | { kind: "info", text: string }
 *   | { kind: "help" }
 *   | { kind: "error", text: string }} SubmitDecision
 */

/**
 * Decide what Enter on a compose line should do.
 *
 * Strips once here so classification and the message wire payload share one
 * value (TUI parity: `event.message.text.strip()` before both branches).
 * Leading-space slash lines become commands; trailing spaces do not ride
 * `message.send`.
 *
 * @param {string} rawText
 * @param {CommandSpec[]} catalog
 * @returns {SubmitDecision}
 */
export function decideSubmit(rawText, catalog) {
  const text = String(rawText ?? "").trim();
  if (!text.startsWith("/")) {
    return { kind: "message", text };
  }
  const specs = Array.isArray(catalog) ? catalog : [];
  // No bare-`/` case on purpose. A `/` always leaves the menu open, and Enter
  // then takes the highlighted row, so this function is not on that journey --
  // the TUI does the same (tui.py on_prompt_input_submitted), which is the
  // parity this issue is about.
  const spec = findCommand(specs, text);
  if (spec === null) {
    return { kind: "error", text: `unknown command: ${text}` };
  }
  const args = slashArguments(text);
  // No action_id means the command is local help (today: `/help` / `/?`).
  // With a topic, answer in-band. Palette-on-topic is wrong: it would arm a
  // highlighted row. Bare help lists every command, which is what the palette
  // already draws -- the TUI prints the same three fields per row.
  if (spec.action_id === null) {
    if (args.length === 0) {
      return { kind: "help" };
    }
    const token = String(args[0]).replace(/^\//, "");
    const topicSpec = findCommand(specs, `/${token}`);
    if (topicSpec === null) {
      return { kind: "error", text: `unknown command: /${token}` };
    }
    return { kind: "info", text: detailLine(topicSpec) };
  }
  if (args.length > 0) {
    return { kind: "error", text: `usage: /${spec.name}` };
  }
  return { kind: "command", spec, args };
}
