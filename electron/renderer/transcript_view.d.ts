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

export const CUT_OFF_LINE: string;

export function sourceClass(source: unknown): string;

export function sourceLabel(entry: TranscriptEntryLike): string;

export function entryBodyText(entry: TranscriptEntryLike): string;

export function commandOutputLines(entry: TranscriptEntryLike): string[];

export function idlePartialText(state: {
  microphone_muted?: boolean;
  audio_stream_muted?: boolean;
}): string;

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
  state: {
    partial_source?: string;
    partial_text?: string;
    microphone_muted?: boolean;
    audio_stream_muted?: boolean;
  },
): void;
