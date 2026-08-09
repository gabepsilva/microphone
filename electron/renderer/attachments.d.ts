import type { DomDocument, DomElement } from "./transcript_view.js";

export const MAX_IMAGE_BYTES: number;
export const MAX_IMAGE_LABEL: string;
export const IMAGE_TOKEN_RE: RegExp;

export function imageToken(number: number): string;

export function parseImageNumbers(text: string): number[];

export type ImageTokenOccurrence = {
  number: number;
  token: string;
  start: number;
  end: number;
};

export function parseImageTokens(text: string): ImageTokenOccurrence[];

export function looksLikeImage(bytes: Uint8Array): boolean;

export function oversizeRefusal(file: { name?: string; size?: number }): string | null;

export function magicRefusal(name: string, bytes: Uint8Array): string | null;

export function base64FromBytes(bytes: Uint8Array): string;

export class DraftAttachments {
  ids: string[];
  add(attachmentId: string): string;
  resolve(text: string): string[];
  clear(): void;
}

export function renderTokenTextInto(
  document: DomDocument,
  host: DomElement,
  text: string,
): void;
