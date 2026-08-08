import { describe, expect, it } from "bun:test";

import {
  applyTranscriptDomEvent,
  buildTranscriptRowElement,
  commandOutputLines,
  entryBodyText,
  entryFootnote,
  idlePartialText,
  isTailing,
  TAIL_SLACK_PX,
  renderPartialLine,
  renderTranscriptSnapshot,
  rowLayout,
  sourceClass,
  sourceLabel,
  usesMarkdownBody,
} from "../renderer/transcript_view.js";

import { flatText, makeDocument, type FakeNode } from "./fake_dom.js";

/** The message bubble, where every piece of entry text lands. */
function bubble(row: FakeNode): FakeNode {
  const found = row.children.find((child) => child.className === "msg-bubble");
  if (found === undefined) {
    throw new Error("row has no msg-bubble");
  }
  return found;
}

function meta(row: FakeNode): FakeNode {
  const found = row.children.find((child) => child.className === "msg-meta");
  if (found === undefined) {
    throw new Error("row has no msg-meta");
  }
  return found;
}

function body(row: FakeNode): FakeNode {
  const found = bubble(row).children.find((child) =>
    child.className.startsWith("msg-body"),
  );
  if (found === undefined) {
    throw new Error("row has no msg-body");
  }
  return found;
}

/** What the body would put on screen, markdown children included. */
function bodyText(row: FakeNode): string {
  return flatText(body(row));
}

function textsIn(node: FakeNode): string[] {
  return node.children.map((child) => child.textContent ?? "");
}

describe("transcript_view hostile text", () => {
  const hostile =
    '<img src=x onerror="window.__xss=1"> <script>window.__xss=1</script>';

  it("renders transcript text as literal textContent, not HTML nodes", () => {
    const { document, root } = makeDocument();
    const row = buildTranscriptRowElement(document, {
      id: 7,
      entry: {
        kind: "speech",
        source: "Voice",
        stamp: "02:59:02",
        text: hostile,
        output: [`out:${hostile}`],
      },
    }) as FakeNode;
    root.appendChild(row);

    expect(row.children.map((child) => child.className)).toEqual([
      "msg-meta",
      "msg-bubble",
    ]);
    expect(textsIn(meta(row))).toEqual(["Voice", "02:59:02"]);
    expect(bodyText(row)).toBe(hostile);
    const output = bubble(row).children.find(
      (child) => child.className === "msg-output",
    );
    expect(output?.textContent).toBe(`out:${hostile}`);
  });

  it("keeps snapshot / update / partial paths on textContent", () => {
    const { document, root } = makeDocument();
    renderTranscriptSnapshot(root, document, [
      {
        id: 1,
        entry: { kind: "speech", source: "Agent", text: hostile },
      },
    ]);
    expect(root.children).toHaveLength(1);
    expect(bodyText(root.children[0]!)).toBe(hostile);

    applyTranscriptDomEvent(root, document, {
      name: "transcript.entry_updated",
      row: {
        id: 1,
        entry: { kind: "speech", source: "Agent", text: `${hostile} more` },
      },
    });
    expect(bodyText(root.children[0]!)).toBe(`${hostile} more`);

    const partial = document.createElement("div");
    renderPartialLine(partial, {
      partial_source: "Voice",
      partial_text: hostile,
    });
    expect(partial.hidden).toBe(false);
    expect(textsIn(partial as FakeNode).join("")).toBe(`Voice${hostile}`);
  });

  it("clears the list on transcript.cleared", () => {
    const { document, root } = makeDocument();
    renderTranscriptSnapshot(root, document, [
      { id: 1, entry: { kind: "speech", source: "Text", text: "hi" } },
    ]);
    applyTranscriptDomEvent(root, document, { name: "transcript.cleared" });
    expect(root.children).toHaveLength(0);
  });

  it("removes a row on transcript.entry_removed", () => {
    const { document, root } = makeDocument();
    renderTranscriptSnapshot(root, document, [
      { id: 1, entry: { kind: "speech", source: "Voice", text: "kept" } },
      { id: 2, entry: { kind: "speech", source: "Voice", text: "echo" } },
    ]);
    applyTranscriptDomEvent(root, document, {
      name: "transcript.entry_removed",
      id: 2,
    });
    expect(root.children).toHaveLength(1);
    expect(bodyText(root.children[0]!)).toBe("kept");
  });
});

describe("transcript_view entry model", () => {
  it("colours known speakers and leaves unknown ones muted", () => {
    expect(sourceClass("Voice")).toBe("source-voice");
    expect(sourceClass("Taga")).toBe("source-taga");
    expect(sourceClass("Audio")).toBe("source-audio");
    expect(sourceClass(undefined)).toBe("source-unknown");
    expect(sourceClass("Whoever")).toBe("source-unknown");
  });

  it("bubbles the room and gives Taga the full column", () => {
    expect(rowLayout({ kind: "speech", source: "Voice" })).toBe("inbound");
    expect(rowLayout({ kind: "speech", source: "Audio" })).toBe("inbound");
    expect(rowLayout({ kind: "speech", source: "Agent" })).toBe("inbound");
    expect(rowLayout({ kind: "speech", source: "Taga" })).toBe("answer");
    expect(rowLayout({ kind: "reasoning", source: "Taga" })).toBe("answer");
    expect(rowLayout({ kind: "note", source: "Voice" })).toBe("note");
    expect(rowLayout({ kind: "command", source: "Taga" })).toBe("command");
  });

  it("labels interface rows by what they are, not by a speaker", () => {
    expect(sourceLabel({ kind: "note", source: "Taga" })).toBe("note");
    expect(sourceLabel({ kind: "command", source: "Taga" })).toBe("command");
    expect(sourceLabel({ kind: "reasoning", source: "Taga" })).toBe("thinking");
    expect(sourceLabel({ kind: "speech", source: "Taga" })).toBe("Taga");
    expect(sourceLabel({ kind: "speech" })).toBe("");
  });

  it("renders each entry kind the way the TUI does", () => {
    expect(entryBodyText({ kind: "speech", text: "hi" })).toBe("hi");
    expect(entryBodyText({ kind: "speech", text: "hi", streaming: true })).toBe("hi ▍");
    expect(entryBodyText({ kind: "command", text: "ls -l" })).toBe("$ ls -l");
    expect(entryBodyText({ kind: "reasoning", streaming: true })).toBe("thinking…");
    expect(entryBodyText({ kind: "reasoning", text: "why" })).toBe("why");
  });

  it("puts the thinking cost and the command exit in the meta line", () => {
    expect(entryFootnote({ kind: "reasoning", seconds: 1.24 })).toBe("1.2s");
    expect(entryFootnote({ kind: "reasoning", seconds: 1.24, streaming: true })).toBe(
      "",
    );
    expect(entryFootnote({ kind: "command", exit_code: 1 })).toBe("exit 1");
    expect(entryFootnote({ kind: "command" })).toBe("");
    expect(entryFootnote({ kind: "speech", exit_code: 0 })).toBe("");
  });

  it("keeps only string output lines", () => {
    expect(commandOutputLines({ kind: "command", output: ["a", 3, null] })).toEqual([
      "a",
    ]);
    expect(commandOutputLines({ kind: "speech" })).toEqual([]);
  });

  it("marks an interrupted answer with cut-off chrome", () => {
    const { document } = makeDocument();
    const row = buildTranscriptRowElement(document, {
      id: 3,
      entry: { kind: "speech", source: "Taga", text: "half", interrupted: true },
    }) as FakeNode;
    const cutoff = bubble(row).children.find(
      (child) => child.className === "msg-cutoff",
    );
    expect(cutoff?.textContent).toBe("cut off: user started speaking");
  });

  it("tags a row with its layout and its speaker palette", () => {
    const { document } = makeDocument();
    const row = buildTranscriptRowElement(document, {
      id: 4,
      entry: { kind: "command", source: "Taga", text: "ls", exit_code: 0 },
    }) as FakeNode;
    expect(row.className).toBe("msg msg-command source-taga");
    expect(textsIn(meta(row))).toEqual(["command", "exit 0"]);
  });

  it("reads only a finished Taga answer as Markdown", () => {
    // Mirrors tui.uses_markdown_body — a streaming turn and anything
    // transcribed from the room stay literal.
    expect(usesMarkdownBody({ kind: "speech", source: "Taga", text: "**hi**" })).toBe(
      true,
    );
    expect(
      usesMarkdownBody({
        kind: "speech",
        source: "Taga",
        text: "**hi**",
        streaming: true,
      }),
    ).toBe(false);
    expect(usesMarkdownBody({ kind: "speech", source: "Voice", text: "**hi**" })).toBe(
      false,
    );
    expect(usesMarkdownBody({ kind: "reasoning", source: "Taga", text: "x" })).toBe(
      false,
    );
    expect(usesMarkdownBody({ kind: "speech", source: "Taga", text: "" })).toBe(false);
  });

  it("builds a Taga answer's markdown as elements, and leaves a voice line alone", () => {
    const { document } = makeDocument();
    const answer = buildTranscriptRowElement(document, {
      id: 8,
      entry: { kind: "speech", source: "Taga", text: "run `ls`\n\n```sh\nls -l\n```" },
    }) as FakeNode;
    expect(body(answer).className).toBe("msg-body kind-speech md");
    const blocks = body(answer).children.map((child) => child.className);
    expect(blocks).toEqual(["md-paragraph", "md-code"]);

    const spoken = buildTranscriptRowElement(document, {
      id: 9,
      entry: { kind: "speech", source: "Voice", text: "run `ls`" },
    }) as FakeNode;
    expect(body(spoken).children).toHaveLength(0);
    expect(body(spoken).textContent).toBe("run `ls`");
  });

  it("says why the partial line is quiet instead of hiding it", () => {
    expect(idlePartialText({})).toBe("Listening...");
    expect(idlePartialText({ microphone_muted: true })).toBe(
      "mic muted, audio still transcribing",
    );
    expect(idlePartialText({ audio_stream_muted: true })).toBe(
      "speaker muted, mic still hot",
    );
    expect(idlePartialText({ microphone_muted: true, audio_stream_muted: true })).toBe(
      "mic and speaker muted, nothing transcribing",
    );

    const { document } = makeDocument();
    const partial = document.createElement("div");
    renderPartialLine(partial, { microphone_muted: true });
    expect(partial.hidden).toBe(false);
    expect(textsIn(partial as FakeNode).join("")).toBe(
      "mic muted, audio still transcribing",
    );
  });
});

describe("transcript_view tail following", () => {
  // A chat surface that scrolls on every event cannot be read during a
  // streaming answer: the TUI gates the same behaviour on `_tailing`.
  const area = (scrollHeight: number, scrollTop: number, clientHeight: number) => ({
    scrollHeight,
    scrollTop,
    clientHeight,
  });

  it("follows at the live end and while within a row or two of it", () => {
    expect(isTailing(area(0, 0, 0))).toBe(true);
    expect(isTailing(area(2000, 1400, 600))).toBe(true);
    expect(isTailing(area(2000, 1400 - TAIL_SLACK_PX, 600))).toBe(true);
  });

  it("stops following once the reader is held back in history", () => {
    expect(isTailing(area(2000, 1400 - TAIL_SLACK_PX - 1, 600))).toBe(false);
    expect(isTailing(area(2000, 0, 600))).toBe(false);
  });

  it("reads as following only before the row lands, not after", () => {
    // Sitting at the end of a 2000px transcript.
    const before = area(2000, 1400, 600);
    expect(isTailing(before)).toBe(true);
    // The same view once a 400px answer has been appended and nothing has
    // scrolled yet: measuring here would decide the reader had walked away.
    expect(isTailing(area(2400, 1400, 600))).toBe(false);
  });
});
