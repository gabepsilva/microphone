/**
 * A DOM double for the renderer helpers.
 *
 * Deliberately dumb: nodes only remember what was set on them. That is the
 * point — a helper that reached for innerHTML or built markup out of a string
 * would have nowhere to put it here, so these tests fail on the mechanism
 * rather than on a rendered result that happens to look right.
 */

import type { DomDocument, DomElement } from "../renderer/transcript_view.js";

export type FakeNode = DomElement & {
  nodeType: number;
  tag: string;
  children: FakeNode[];
  dataset: Record<string, string>;
  ownerDocument: FakeDocument;
};

export type FakeDocument = DomDocument & {
  createElement(tag: string): FakeNode;
  createTextNode(text: string): FakeNode;
};

export function makeDocument(): { document: FakeDocument; root: FakeNode } {
  const document = {} as FakeDocument;
  const makeNode = (tag: string): FakeNode => {
    const node: FakeNode = {
      nodeType: tag === "#text" ? 3 : 1,
      tag,
      textContent: "",
      className: "",
      title: "",
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
  document.createElement = (tag: string) => makeNode(tag);
  document.createTextNode = (text: string) => {
    const node = makeNode("#text");
    node.textContent = text;
    return node;
  };
  const root = makeNode("div");
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

/** Every piece of text a subtree would put on screen, in document order. */
export function flatText(node: FakeNode): string {
  if (node.children.length === 0) {
    return node.textContent ?? "";
  }
  return node.children.map(flatText).join("");
}
