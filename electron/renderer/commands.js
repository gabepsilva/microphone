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
