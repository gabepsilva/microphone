import { describe, expect, it } from "bun:test";

import {
  clampIndex,
  commandQuery,
  decideSubmit,
  detailLine,
  findCommand,
  matchCommands,
  parseCommandList,
  parseCommandSpec,
  slashArguments,
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

describe("slash line resolution", () => {
  it("resolves the first token against name and aliases", () => {
    expect(findCommand(CATALOG, "/new")?.name).toBe("new");
    expect(findCommand(CATALOG, "/clear")?.name).toBe("new");
    expect(findCommand(CATALOG, "/NEW keep")?.name).toBe("new");
    expect(findCommand(CATALOG, "/missing")).toBeNull();
    expect(findCommand(CATALOG, "new")).toBeNull();
  });

  it("exposes arguments after the command name", () => {
    expect(slashArguments("/new")).toEqual([]);
    expect(slashArguments("/new keep")).toEqual(["keep"]);
    expect(slashArguments("  /help one two  ")).toEqual(["one", "two"]);
    expect(slashArguments("plain")).toEqual([]);
  });
});

describe("detail line (TUI detail_line parity)", () => {
  it("matches CommandSpec.detail_line shape", () => {
    // Mirrors tests/test_commands.py::test_command_spec_formats_help_lines.
    expect(detailLine(spec("status", ""))).toBe("/status: no description");
    expect(detailLine(spec("new", "Fresh session", ["clear"]))).toBe(
      "/new (aliases: /clear): Fresh session",
    );
  });
});

describe("submit decision", () => {
  it("routes plain text as a message", () => {
    expect(decideSubmit("hello", CATALOG)).toEqual({
      kind: "message",
      text: "hello",
    });
    expect(decideSubmit("", CATALOG)).toEqual({ kind: "message", text: "" });
  });

  it("strips once so leading-space slash and trailing spaces match the TUI", () => {
    // Acceptance #134: " /new" and "/new " are commands; messages ship trimmed.
    expect(decideSubmit(" /new", CATALOG).kind).toBe("command");
    expect(decideSubmit("/new ", CATALOG).kind).toBe("command");
    expect(decideSubmit("  /clear  ", CATALOG).kind).toBe("command");
    expect(decideSubmit(" hi ", CATALOG)).toEqual({
      kind: "message",
      text: "hi",
    });
  });

  it("dispatches action-backed commands without leftover args", () => {
    const decision = decideSubmit("/new", CATALOG);
    expect(decision.kind).toBe("command");
    if (decision.kind === "command") {
      expect(decision.spec.name).toBe("new");
      expect(decision.args).toEqual([]);
    }
    expect(decideSubmit("/clear", CATALOG).kind).toBe("command");
  });

  it("refuses leftover args on action-backed commands", () => {
    expect(decideSubmit("/new keep", CATALOG)).toEqual({
      kind: "error",
      text: "usage: /new",
    });
  });

  it("asks for the full listing on bare /help and its alias", () => {
    // Bare help is not a dispatchable command: it has no action_id, and
    // routing it as one drove completeSelectedCommand, which rewrote the
    // prompt to `/help` and collapsed the menu to the `/help` row itself.
    expect(decideSubmit("/help", CATALOG)).toEqual({ kind: "help" });
    expect(decideSubmit("/?", CATALOG)).toEqual({ kind: "help" });
  });

  it("does not special-case a bare /, which never reaches here", () => {
    // A `/` keeps the menu open, and app.js prompt.send prefers the
    // highlighted row, so Enter runs `/new` and decideSubmit is not called.
    // The TUI does the same -- driving its prompt with "/" then Enter runs
    // `/new` -- so this is the parity the issue asks for, not a gap to patch.
    // Reachable only with an empty catalog, where an error beats an empty menu.
    expect(decideSubmit("/", CATALOG)).toEqual({
      kind: "error",
      text: "unknown command: /",
    });
  });

  it("lists the whole catalog for the query bare help parks on", () => {
    // The decision is only useful if `/` is the query that matches
    // everything: showBareHelp writes `/`, and syncPalette recomputes rows
    // from commandQuery of the prompt.
    const query = commandQuery("/");
    expect(query).toBe("");
    expect(matchCommands(CATALOG, query ?? "x").map((row) => row.name)).toEqual([
      "new",
      "help",
      "newer",
    ]);
    // What the old routing collapsed to, for contrast.
    expect(matchCommands(CATALOG, "help").map((row) => row.name)).toEqual(["help"]);
  });

  it("answers /help <topic> with a detail line (info, not palette)", () => {
    expect(decideSubmit("/help new", CATALOG)).toEqual({
      kind: "info",
      text: "/new (aliases: /clear): Start a fresh session",
    });
    // Alias and leading slash on the topic both resolve.
    expect(decideSubmit("/help clear", CATALOG)).toEqual({
      kind: "info",
      text: "/new (aliases: /clear): Start a fresh session",
    });
    expect(decideSubmit("/help /new", CATALOG)).toEqual({
      kind: "info",
      text: "/new (aliases: /clear): Start a fresh session",
    });
  });

  it("reports unknown slash and unknown help topics as errors", () => {
    expect(decideSubmit("/missing", CATALOG)).toEqual({
      kind: "error",
      text: "unknown command: /missing",
    });
    expect(decideSubmit("/help missing", CATALOG)).toEqual({
      kind: "error",
      text: "unknown command: /missing",
    });
  });
});
