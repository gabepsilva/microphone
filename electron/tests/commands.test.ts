import { describe, expect, it } from "bun:test";

import {
  clampIndex,
  commandQuery,
  matchCommands,
  parseCommandList,
  parseCommandSpec,
  type CommandSpec,
} from "../renderer/commands.js";

function spec(
  name: string,
  summary = "",
  aliases: string[] = [],
  action_id: string | null = null,
): CommandSpec {
  return { name, summary, aliases, action_id };
}

const CATALOG: CommandSpec[] = [
  spec("new", "Start a fresh session", ["clear"], "session.new"),
  spec("help", "List available slash commands", ["?"]),
  spec("newer", "Something else"),
];

describe("command palette query", () => {
  it("opens only for a single-token slash query", () => {
    // Mirrors commands.command_query: arguments close the menu so free-form
    // args are unobstructed.
    expect(commandQuery("/")).toBe("");
    expect(commandQuery("/ne")).toBe("ne");
    expect(commandQuery("/NE")).toBe("ne");
    expect(commandQuery("/new keep")).toBeNull();
    expect(commandQuery("/new\n")).toBeNull();
    expect(commandQuery("hello")).toBeNull();
    expect(commandQuery("")).toBeNull();
  });
});

describe("command palette ranking", () => {
  it("lists the whole catalog in registration order for an empty query", () => {
    expect(matchCommands(CATALOG, "").map((row) => row.name)).toEqual([
      "new",
      "help",
      "newer",
    ]);
  });

  it("prefers a prefix, then a substring, then a subsequence", () => {
    expect(matchCommands(CATALOG, "ne").map((row) => row.name)).toEqual([
      "new",
      "newer",
    ]);
    // `nw` is no prefix and no substring of either, but is a subsequence.
    expect(matchCommands(CATALOG, "nw").map((row) => row.name)).toEqual([
      "new",
      "newer",
    ]);
    expect(matchCommands(CATALOG, "elp").map((row) => row.name)).toEqual(["help"]);
  });

  it("matches aliases and, last, descriptions", () => {
    expect(matchCommands(CATALOG, "clear").map((row) => row.name)).toEqual(["new"]);
    expect(matchCommands(CATALOG, "slash").map((row) => row.name)).toEqual(["help"]);
  });

  it("returns nothing when nothing matches", () => {
    expect(matchCommands(CATALOG, "zzz")).toEqual([]);
    expect(matchCommands([], "x")).toEqual([]);
  });
});

describe("commands.list parsing", () => {
  it("reads the socket envelope and the bare array alike", () => {
    const row = {
      name: "new",
      summary: "s",
      aliases: ["clear"],
      action_id: "session.new",
    };
    expect(parseCommandList({ commands: [row] })).toEqual([
      { name: "new", summary: "s", aliases: ["clear"], action_id: "session.new" },
    ]);
    expect(parseCommandList([row])).toHaveLength(1);
    expect(parseCommandList(null)).toEqual([]);
    expect(parseCommandList({})).toEqual([]);
  });

  it("drops a malformed row instead of rendering it half-built", () => {
    expect(parseCommandSpec({ name: "" })).toBeNull();
    expect(parseCommandSpec({ summary: "no name" })).toBeNull();
    expect(parseCommandSpec(null)).toBeNull();
    expect(parseCommandSpec({ name: "help" })).toEqual({
      name: "help",
      summary: "",
      aliases: [],
      action_id: null,
    });
    expect(parseCommandSpec({ name: "help", aliases: ["?", 7] })?.aliases).toEqual([
      "?",
    ]);
  });
});

describe("palette selection", () => {
  it("wraps in both directions and survives an empty list", () => {
    expect(clampIndex(0, 3)).toBe(0);
    expect(clampIndex(3, 3)).toBe(0);
    expect(clampIndex(-1, 3)).toBe(2);
    expect(clampIndex(5, 0)).toBe(0);
  });
});
