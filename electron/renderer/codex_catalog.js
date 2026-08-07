/**
 * The Codex model catalog the sidebar pickers are drawn from.
 *
 * Which models exist and what efforts each one accepts is the CLI's answer
 * (`codex.catalog` over the socket), not this client's guess — offering an
 * effort a model does not support is offering a switch the session refuses.
 *
 * The rules here mirror tui.options_including and tui.adopt_efforts_for so
 * both clients offer the same thing for the same catalog.
 */

/**
 * @typedef {object} CodexModel
 * @property {string} slug
 * @property {string} label
 * @property {string[]} efforts
 * @property {string} default_effort
 */

/**
 * Read a `codex.catalog` reply, dropping rows that cannot be drawn.
 * @param {unknown} value
 * @returns {CodexModel[]}
 */
export function parseCodexCatalog(value) {
  const rows =
    value !== null &&
    typeof value === "object" &&
    Array.isArray(/** @type {Record<string, unknown>} */ (value).models)
      ? /** @type {Record<string, unknown>} */ (value).models
      : value;
  if (!Array.isArray(rows)) {
    return [];
  }
  const models = [];
  for (const row of rows) {
    if (row === null || typeof row !== "object") {
      continue;
    }
    const record = /** @type {Record<string, unknown>} */ (row);
    if (typeof record.slug !== "string" || record.slug === "") {
      continue;
    }
    const efforts = Array.isArray(record.efforts)
      ? record.efforts.filter((effort) => typeof effort === "string" && effort !== "")
      : [];
    // A model with no efforts cannot answer the effort picker, and the
    // Python catalog already drops those; refuse them here too.
    if (efforts.length === 0) {
      continue;
    }
    const fallback = efforts[0];
    const preferred =
      typeof record.default_effort === "string" &&
      efforts.includes(record.default_effort)
        ? record.default_effort
        : fallback;
    models.push({
      slug: record.slug,
      label:
        typeof record.label === "string" && record.label !== ""
          ? record.label
          : record.slug,
      efforts,
      default_effort: preferred,
    });
  }
  return models;
}

/**
 * Options for the model picker, with the running model guaranteed present.
 *
 * A model configured before this catalog was discovered — or one the CLI has
 * since stopped listing — must still be selectable, or the picker would show
 * a session running something it does not offer. Mirrors
 * tui.options_including.
 * @param {CodexModel[]} models
 * @param {string} current
 * @returns {{ value: string, label: string }[]}
 */
export function modelOptions(models, current) {
  const options = models.map((model) => ({ value: model.slug, label: model.label }));
  if (current && !options.some((option) => option.value === current)) {
    return [{ value: current, label: current }, ...options];
  }
  return options;
}

/**
 * The efforts a model accepts, or the running effort alone when the catalog
 * has nothing to say about it.
 * @param {CodexModel[]} models
 * @param {string} model
 * @param {string} current
 * @returns {string[]}
 */
export function effortOptions(models, model, current) {
  const found = models.find((entry) => entry.slug === model);
  const efforts = found === undefined ? [] : [...found.efforts];
  if (efforts.length === 0) {
    return current ? [current] : [];
  }
  if (current && !efforts.includes(current)) {
    return [current, ...efforts];
  }
  return efforts;
}

/**
 * The effort to switch to when a model is chosen: keep the running one when
 * the new model accepts it, else the model's default. Null means no change
 * is needed. Mirrors tui.adopt_efforts_for.
 * @param {CodexModel[]} models
 * @param {string} model
 * @param {string} current
 * @returns {string | null}
 */
export function effortForModel(models, model, current) {
  const found = models.find((entry) => entry.slug === model);
  if (found === undefined || found.efforts.length === 0) {
    return null;
  }
  if (found.efforts.includes(current)) {
    return null;
  }
  return found.default_effort;
}
