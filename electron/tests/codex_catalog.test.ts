import { describe, expect, it } from "bun:test";

import {
  effortForModel,
  effortOptions,
  modelOptions,
  parseCodexCatalog,
  type CodexModel,
} from "../renderer/codex_catalog.js";

const LUNA: CodexModel = {
  slug: "gpt-5.6-luna",
  label: "Luna",
  efforts: ["low", "medium", "high"],
  default_effort: "medium",
};
const SOL: CodexModel = {
  slug: "gpt-5.6-sol",
  label: "Sol",
  efforts: ["high"],
  default_effort: "high",
};
const CATALOG = [LUNA, SOL];

describe("codex.catalog parsing", () => {
  it("reads the socket envelope and a bare array alike", () => {
    const row = {
      slug: "gpt-5.6-luna",
      label: "Luna",
      efforts: ["low", "high"],
      default_effort: "high",
    };
    expect(parseCodexCatalog({ models: [row] })).toEqual([
      {
        slug: "gpt-5.6-luna",
        label: "Luna",
        efforts: ["low", "high"],
        default_effort: "high",
      },
    ]);
    expect(parseCodexCatalog([row])).toHaveLength(1);
    expect(parseCodexCatalog(null)).toEqual([]);
    expect(parseCodexCatalog({ models: "nope" })).toEqual([]);
  });

  it("drops a model that cannot answer the effort picker", () => {
    // The Python catalog already refuses these; refusing them here keeps the
    // picker from offering a model with no efforts behind it.
    expect(parseCodexCatalog([{ slug: "x", efforts: [] }])).toEqual([]);
    expect(parseCodexCatalog([{ efforts: ["low"] }])).toEqual([]);
    expect(parseCodexCatalog([{ slug: "", efforts: ["low"] }])).toEqual([]);
  });

  it("falls back to the slug for a label and to a real effort for the default", () => {
    expect(
      parseCodexCatalog([{ slug: "x", efforts: ["low", 7], default_effort: "nope" }]),
    ).toEqual([{ slug: "x", label: "x", efforts: ["low"], default_effort: "low" }]);
  });
});

describe("model picker options", () => {
  it("offers the catalog by label", () => {
    expect(modelOptions(CATALOG, "gpt-5.6-luna")).toEqual([
      { value: "gpt-5.6-luna", label: "Luna" },
      { value: "gpt-5.6-sol", label: "Sol" },
    ]);
  });

  it("keeps a running model the catalog does not list", () => {
    // Mirrors tui.options_including: a session running something unlisted
    // must still show what it is running.
    expect(modelOptions(CATALOG, "gpt-5.5-legacy")[0]).toEqual({
      value: "gpt-5.5-legacy",
      label: "gpt-5.5-legacy",
    });
    expect(modelOptions([], "")).toEqual([]);
  });
});

describe("effort picker options", () => {
  it("offers what the chosen model accepts", () => {
    expect(effortOptions(CATALOG, "gpt-5.6-luna", "low")).toEqual([
      "low",
      "medium",
      "high",
    ]);
    expect(effortOptions(CATALOG, "gpt-5.6-sol", "high")).toEqual(["high"]);
  });

  it("keeps the running effort when the catalog is silent", () => {
    expect(effortOptions([], "gpt-5.6-luna", "low")).toEqual(["low"]);
    expect(effortOptions(CATALOG, "unknown-model", "low")).toEqual(["low"]);
    expect(effortOptions(CATALOG, "gpt-5.6-sol", "low")).toEqual(["low", "high"]);
    expect(effortOptions([], "unknown", "")).toEqual([]);
  });
});

describe("effort after a model switch", () => {
  it("keeps an effort the new model accepts", () => {
    expect(effortForModel(CATALOG, "gpt-5.6-luna", "high")).toBeNull();
  });

  it("moves to the model's default when the running effort is refused", () => {
    // Mirrors tui.adopt_efforts_for — Sol accepts only "high".
    expect(effortForModel(CATALOG, "gpt-5.6-sol", "low")).toBe("high");
    expect(effortForModel(CATALOG, "gpt-5.6-luna", "")).toBe("medium");
  });

  it("changes nothing for a model the catalog does not describe", () => {
    expect(effortForModel(CATALOG, "unknown-model", "low")).toBeNull();
    expect(effortForModel([], "gpt-5.6-luna", "low")).toBeNull();
  });
});
