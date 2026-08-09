import { describe, expect, it } from "bun:test";

import {
  buildBlockElement,
  isSafeUrl,
  parseInline,
  parseMarkdown,
  renderMarkdownInto,
  type Block,
} from "../renderer/markdown.js";

import { flatText, makeDocument, type FakeNode } from "./fake_dom.js";

describe("markdown blocks", () => {
  it("reads a fenced code block whole, language and all", () => {
    const blocks = parseMarkdown(
      ["before", "```python", "def f():", "    return 1", "```", "after"].join("\n"),
    );
    expect(blocks).toEqual([
      { type: "paragraph", text: "before" },
      { type: "code", lang: "python", text: "def f():\n    return 1" },
      { type: "paragraph", text: "after" },
    ]);
  });

  it("keeps markdown syntax inside a fence literal", () => {
    const blocks = parseMarkdown(
      ["```", "# not a heading", "- not a list", "```"].join("\n"),
    );
    expect(blocks).toEqual([
      { type: "code", lang: "", text: "# not a heading\n- not a list" },
    ]);
  });

  it("closes an unterminated fence at the end of the answer", () => {
    // A streaming turn is cut mid-block far more often than a model forgets
    // to close a fence.
    expect(parseMarkdown("```sh\nls -l")).toEqual([
      { type: "code", lang: "sh", text: "ls -l" },
    ]);
  });

  it("reads headings, lists, quotes, and rules", () => {
    const blocks = parseMarkdown(
      [
        "## Title",
        "- one",
        "- two",
        "",
        "1. first",
        "2. second",
        "> quoted",
        "---",
      ].join("\n"),
    );
    expect(blocks).toEqual([
      { type: "heading", level: 2, text: "Title" },
      { type: "list", ordered: false, items: ["one", "two"] },
      { type: "list", ordered: true, items: ["first", "second"] },
      { type: "quote", text: "quoted" },
      { type: "rule" },
    ]);
  });

  it("reads a pipe table, and leaves lone pipes as prose", () => {
    expect(
      parseMarkdown(["| a | b |", "| --- | --- |", "| 1 | 2 |"].join("\n")),
    ).toEqual([{ type: "table", header: ["a", "b"], rows: [["1", "2"]] }]);
    expect(parseMarkdown("cat foo | grep bar")).toEqual([
      { type: "paragraph", text: "cat foo | grep bar" },
    ]);
  });

  it("keeps a paragraph's own line breaks", () => {
    expect(parseMarkdown("one\ntwo")).toEqual([
      { type: "paragraph", text: "one\ntwo" },
    ]);
  });
});

describe("markdown inline", () => {
  it("splits code, emphasis, and links", () => {
    expect(parseInline("a `x` **b** _c_ [d](https://e.f) end")).toEqual([
      { type: "text", text: "a " },
      { type: "code", text: "x" },
      { type: "text", text: " " },
      { type: "strong", text: "b" },
      { type: "text", text: " " },
      { type: "em", text: "c" },
      { type: "text", text: " " },
      { type: "link", text: "d", url: "https://e.f" },
      { type: "text", text: " end" },
    ]);
  });

  it("leaves loose punctuation and snake_case alone", () => {
    // Emphasis needs its delimiters tight against the text, and `_` needs
    // word boundaries, or arithmetic and identifiers turn into prose.
    expect(parseInline("2 * 3 * 4 and _")).toEqual([
      { type: "text", text: "2 * 3 * 4 and _" },
    ]);
    expect(parseInline("call thread_fork_now(x)")).toEqual([
      { type: "text", text: "call thread_fork_now(x)" },
    ]);
    expect(parseInline("_yes_ and snake_case")).toEqual([
      { type: "em", text: "yes" },
      { type: "text", text: " and snake_case" },
    ]);
  });

  it("accepts only http(s) and mailto for a link target", () => {
    expect(isSafeUrl("https://example.com")).toBe(true);
    expect(isSafeUrl("mailto:a@b.c")).toBe(true);
    expect(isSafeUrl("javascript:alert(1)")).toBe(false);
    expect(isSafeUrl("data:text/html,<script>")).toBe(false);
    expect(isSafeUrl("file:///etc/passwd")).toBe(false);
  });
});

describe("markdown rendering is text, never markup", () => {
  const hostile = '<img src=x onerror="window.__xss=1">';

  it("puts hostile source in code and prose on textContent", () => {
    const { document, root } = makeDocument();
    renderMarkdownInto(
      document,
      root,
      ["```html", hostile, "```", "", `and inline ${hostile}`].join("\n"),
    );
    expect(flatText(root)).toBe(`${hostile}and inline ${hostile}`);
    const pre = root.children[0]!;
    expect(pre.tag).toBe("pre");
    expect(pre.children[0]!.tag).toBe("code");
    expect(pre.children[0]!.textContent).toBe(hostile);
  });

  it("never gives a link an href, and marks an unsafe target", () => {
    const { document, root } = makeDocument();
    renderMarkdownInto(
      document,
      root,
      "see [here](https://example.com) and [bad](javascript:alert(1))",
    );
    const anchors: FakeNode[] = root.children[0]!.children.filter(
      (child) => child.tag === "a",
    );
    expect(anchors).toHaveLength(2);
    // No href anywhere: nothing in this window may navigate away from the app.
    expect(anchors.every((anchor) => !("href" in anchor))).toBe(true);
    expect(anchors[0]!.title).toBe("https://example.com");
    expect(anchors[0]!.className).toBe("md-link");
    expect(anchors[1]!.title).toBe("");
    expect(anchors[1]!.className).toBe("md-link md-link-blocked");
    // Link text renders inside the anchor (as a run, not HTML).
    expect(flatText(anchors[1]!)).toBe("bad");
  });

  it("tags a code block with its language for the label", () => {
    const { document } = makeDocument();
    const block: Block = { type: "code", lang: "rust", text: "fn main() {}" };
    const el = buildBlockElement(document, block) as FakeNode;
    expect(el.dataset.lang).toBe("rust");
    const plain = buildBlockElement(document, {
      type: "code",
      lang: "",
      text: "x",
    }) as FakeNode;
    expect(plain.dataset.lang).toBeUndefined();
  });

  it("builds a table out of cells, not a string", () => {
    const { document } = makeDocument();
    const el = buildBlockElement(document, {
      type: "table",
      header: ["a", "b"],
      rows: [["1", hostile]],
    }) as FakeNode;
    expect(el.tag).toBe("table");
    expect(flatText(el)).toBe(`ab1${hostile}`);
  });
});

describe("isSafeUrl", () => {
  it("allows http(s) and mailto after trim", () => {
    expect(isSafeUrl("https://example.com")).toBe(true);
    expect(isSafeUrl("  http://example.com/path  ")).toBe(true);
    expect(isSafeUrl("mailto:a@b.c")).toBe(true);
  });

  it("rejects javascript, data, and scheme-smuggling whitespace", () => {
    expect(isSafeUrl("javascript:alert(1)")).toBe(false);
    expect(isSafeUrl("data:text/html,x")).toBe(false);
    expect(isSafeUrl("https://a.com\njavascript:alert(1)")).toBe(false);
    expect(isSafeUrl("https://a.com\tjavascript:alert(1)")).toBe(false);
    expect(isSafeUrl("")).toBe(false);
  });
});
