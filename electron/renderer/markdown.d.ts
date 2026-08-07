import type { DomDocument, DomElement } from "./transcript_view.js";

export type Span =
  | { type: "text" | "code" | "strong" | "em"; text: string }
  | { type: "link"; text: string; url: string };

export type Block =
  | { type: "code"; lang: string; text: string }
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "quote"; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "rule" };

export function parseInline(text: string): Span[];

export function parseMarkdown(source: string): Block[];

export function isSafeUrl(url: string): boolean;

export function buildBlockElement(document: DomDocument, block: Block): DomElement;

export function renderMarkdownInto(
  document: DomDocument,
  host: DomElement,
  source: string,
): void;
