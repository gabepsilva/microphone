export type TranscriptEntryLike = Record<string, unknown>;

export type TranscriptRowLike = {
  id: number;
  entry: TranscriptEntryLike;
};

export type TranscriptDomEvent = {
  name: string;
  id?: number;
  row?: TranscriptRowLike;
};

/** Minimal DOM surface used by the renderer helpers (no DOM lib in tsc). */
export type DomDocument = {
  createElement(tag: string): DomElement;
  createTextNode(text: string): DomElement;
};

export type DomElement = {
  className: string;
  /** Absent in the test double; the renderer guards every use. */
  classList?: { toggle(name: string, force?: boolean): void };
  hidden: boolean;
  dataset: Record<string, string>;
  textContent: string | null;
  title?: string;
  children: ArrayLike<DomElement>;
  ownerDocument: DomDocument;
  appendChild(child: DomElement): DomElement;
  replaceChildren(...nodes: DomElement[]): void;
  replaceWith(next: DomElement): void;
  remove(): void;
};

export const CUT_OFF_LINE: string;

export function sourceClass(source: unknown): string;

export function sourceLabel(entry: TranscriptEntryLike): string;

export function entryBodyText(entry: TranscriptEntryLike): string;

export function entryFootnote(entry: TranscriptEntryLike): string;

export function usesMarkdownBody(entry: TranscriptEntryLike): boolean;

export function rowLayout(
  entry: TranscriptEntryLike,
): "note" | "command" | "inbound" | "answer";

export function commandOutputLines(entry: TranscriptEntryLike): string[];

export function idlePartialText(state: {
  microphone_muted?: boolean;
  audio_stream_muted?: boolean;
}): string;

export const TAIL_SLACK_PX: number;

export function isTailing(area: {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}): boolean;

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
