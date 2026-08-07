import { describe, expect, it } from "bun:test";

import {
  applyTranscriptDomEvent,
  buildTranscriptRowElement,
  renderPartialLine,
  renderTranscriptSnapshot,
  type DomDocument,
  type DomElement,
} from "../renderer/transcript_view.js";

type FakeNode = DomElement & {
  nodeType: number;
  children: FakeNode[];
  dataset: Record<string, string>;
  ownerDocument: FakeDocument;
};

type FakeDocument = DomDocument & {
  createElement(tag: string): FakeNode;
};

function makeDocument(): { document: FakeDocument; root: FakeNode } {
  const document = {} as FakeDocument;
  const makeNode = (): FakeNode => {
    const node: FakeNode = {
      nodeType: 1,
      textContent: "",
      className: "",
      hidden: false,
      dataset: {},
      children: [],
      ownerDocument: document,
      appendChild(child: DomElement) {
        this.children.push(child as FakeNode);
        return child;
      },
      replaceChildren(...nodes: DomElement[]) {
        this.children = nodes as FakeNode[];
      },
      replaceWith(_next: DomElement) {
        void _next;
      },
    };
    return node;
  };
  document.createElement = (_tag: string) => makeNode();
  const root = makeNode();
  const patchTree = (parent: FakeNode): void => {
    for (const child of parent.children) {
      child.replaceWith = (next: DomElement) => {
        const index = parent.children.indexOf(child);
        if (index >= 0) {
          parent.children[index] = next as FakeNode;
        }
        patchTree(parent);
      };
      patchTree(child);
    }
  };
  const originalAppend = root.appendChild.bind(root);
  root.appendChild = (child: DomElement) => {
    const added = originalAppend(child);
    patchTree(root);
    return added;
  };
  const originalReplace = root.replaceChildren.bind(root);
  root.replaceChildren = (...nodes: DomElement[]) => {
    originalReplace(...nodes);
    patchTree(root);
  };
  return { document, root };
}

describe("transcript_view hostile text", () => {
  const hostile =
    '<img src=x onerror="window.__xss=1"> <script>window.__xss=1</script>';

  it("renders transcript text as literal textContent, not HTML nodes", () => {
    const { document, root } = makeDocument();
    const row = buildTranscriptRowElement(document, {
      id: 7,
      entry: {
        kind: "Voice",
        source: "mic",
        text: hostile,
        output: [`out:${hostile}`],
      },
    }) as FakeNode;
    root.appendChild(row);

    expect(row.children.map((child) => child.className).sort()).toEqual(
      ["transcript-meta", "transcript-output", "transcript-text"].sort(),
    );
    const textNode = row.children.find(
      (child) => child.className === "transcript-text",
    );
    expect(textNode?.textContent).toBe(hostile);
  });

  it("keeps snapshot / update / partial paths on textContent", () => {
    const { document, root } = makeDocument();
    renderTranscriptSnapshot(root, document, [
      {
        id: 1,
        entry: { kind: "Agent", source: "electron", text: hostile },
      },
    ]);
    expect(root.children).toHaveLength(1);
    const body = root.children[0]!.children.find(
      (child) => child.className === "transcript-text",
    );
    expect(body?.textContent).toBe(hostile);

    applyTranscriptDomEvent(root, document, {
      name: "transcript.entry_updated",
      row: {
        id: 1,
        entry: { kind: "Agent", source: "electron", text: `${hostile} more` },
      },
    });
    const updated = root.children[0]!.children.find(
      (child) => child.className === "transcript-text",
    );
    expect(updated?.textContent).toBe(`${hostile} more`);

    const partial = document.createElement("div");
    renderPartialLine(partial, {
      partial_source: "Voice",
      partial_text: hostile,
    });
    expect(partial.hidden).toBe(false);
    expect(
      [...(partial.children as FakeNode[])].map((child) => child.textContent).join(""),
    ).toBe(`Voice: ${hostile}`);
  });

  it("clears the list on transcript.cleared", () => {
    const { document, root } = makeDocument();
    renderTranscriptSnapshot(root, document, [
      { id: 1, entry: { kind: "Text", source: "tui", text: "hi" } },
    ]);
    applyTranscriptDomEvent(root, document, { name: "transcript.cleared" });
    expect(root.children).toHaveLength(0);
  });
});
