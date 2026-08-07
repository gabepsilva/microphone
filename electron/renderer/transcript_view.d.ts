export type TranscriptEntryLike = Record<string, unknown>;

export type TranscriptRowLike = {
  id: number;
  entry: TranscriptEntryLike;
};

export type TranscriptDomEvent = {
  name: string;
  row?: TranscriptRowLike;
};

/** Minimal DOM surface used by the renderer helpers (no DOM lib in tsc). */
export type DomDocument = {
  createElement(tag: string): DomElement;
};

export type DomElement = {
  className: string;
  hidden: boolean;
  dataset: Record<string, string>;
  textContent: string | null;
  children: ArrayLike<DomElement>;
  ownerDocument: DomDocument;
  appendChild(child: DomElement): DomElement;
  replaceChildren(...nodes: DomElement[]): void;
  replaceWith(next: DomElement): void;
};

export function buildTranscriptRowElement(
  document: DomDocument,
  row: TranscriptRowLike,
): DomElement;

export function renderTranscriptSnapshot(
  list: DomElement,
  document: DomDocument,
  rows: TranscriptRowLike[],
): void;

export function applyTranscriptDomEvent(
  list: DomElement,
  document: DomDocument,
  event: TranscriptDomEvent,
): void;

export function renderPartialLine(
  el: DomElement,
  state: { partial_source?: string; partial_text?: string },
): void;
