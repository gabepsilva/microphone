/**
 * Image attachments for the compose surface: tokens, draft, and preflight.
 *
 * Mirrors TagAlong's Python side (tagalong/attachments.py) so the two clients
 * agree on what the record says:
 *
 *   - IMAGE_TOKEN_RE parses the same `[Image #N]` tokens the TUI drafts and
 *     codex.py consumes (attachments.py:32).
 *   - MAX_IMAGE_BYTES is the same 20 MiB cap (attachments.py:52), enforced
 *     *before* any bytes ride the socket — above it the transport would hang
 *     up the connection rather than refuse politely (transport.py:284).
 *   - looksLikeImage() sniffs the same eight magic bytes as the Python side
 *     (attachments.py:121-133).
 *
 * The token is the canonical human-readable record: chips are a rendering of
 * the token scan, never a separate model, and nothing ever strips tokens from
 * entry text. electron/tests/fixtures/token_parity.json pins the JS and Python
 * parsers to the same corpus.
 *
 * Untrusted text must only reach the page via textContent / createElement —
 * never innerHTML. Semgrep bans innerHTML under electron/renderer/.
 */

/** A 20 MiB image upload cap, mirroring attachments.MAX_IMAGE_BYTES. */
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

/** Labels a size cap in copy without drifting from MAX_IMAGE_BYTES. */
export const MAX_IMAGE_LABEL = "20 MiB";

/** Mirrors attachments.IMAGE_TOKEN_RE — `[Image #N]` with a 1-based number. */
export const IMAGE_TOKEN_RE = /\[Image #(\d+)\]/g;

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const JPEG_MAGIC = [0xff, 0xd8, 0xff];
const GIF_MAGIC = [0x47, 0x49, 0x46, 0x38];
const WEBP_RIFF = [0x52, 0x49, 0x46, 0x46];
const WEBP_MARK_TEXT = "WEBP";

/**
 * Render the human-facing marker for attachment `number` (1-based).
 * @param {number} number
 * @returns {string}
 */
export function imageToken(number) {
  if (number < 1) {
    throw new RangeError("image numbers are 1-based");
  }
  return `[Image #${number}]`;
}

/**
 * Image numbers mentioned in `text`, in first-seen order without repeats.
 * Mirrors attachments.parse_image_numbers: a hand-edited draft cannot invent
 * ids or attach the same file twice.
 * @param {string} text
 * @returns {number[]}
 */
export function parseImageNumbers(text) {
  const seen = [];
  IMAGE_TOKEN_RE.lastIndex = 0;
  for (
    let match = IMAGE_TOKEN_RE.exec(text);
    match !== null;
    match = IMAGE_TOKEN_RE.exec(text)
  ) {
    const number = Number(match[1]);
    if (!seen.includes(number)) {
      seen.push(number);
    }
  }
  return seen;
}

/**
 * Every token occurrence in `text`, with its span — the scan the chip view
 * draws from.
 * @param {string} text
 * @returns {Array<{ number: number, token: string, start: number, end: number }>}
 */
export function parseImageTokens(text) {
  const tokens = [];
  IMAGE_TOKEN_RE.lastIndex = 0;
  for (
    let match = IMAGE_TOKEN_RE.exec(text);
    match !== null;
    match = IMAGE_TOKEN_RE.exec(text)
  ) {
    tokens.push({
      number: Number(match[1]),
      token: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return tokens;
}

/**
 * True when `bytes` begins with a recognised image magic — the same set as
 * attachments.looks_like_image: PNG, JPEG, GIF, and RIFF/WEBP. Size is the
 * caller's concern; only the first bytes are inspected.
 * @param {Uint8Array} bytes
 * @returns {boolean}
 */
export function looksLikeImage(bytes) {
  if (!bytes || bytes.length === 0) {
    return false;
  }
  const startsWith = (magic) =>
    bytes.length >= magic.length && magic.every((byte, index) => bytes[index] === byte);
  if (startsWith(PNG_MAGIC) || startsWith(JPEG_MAGIC) || startsWith(GIF_MAGIC)) {
    return true;
  }
  if (!startsWith(WEBP_RIFF) || bytes.length < 12) {
    return false;
  }
  for (let index = 0; index < WEBP_MARK_TEXT.length; index += 1) {
    if (String.fromCharCode(bytes[8 + index]) !== WEBP_MARK_TEXT[index]) {
      return false;
    }
  }
  return true;
}

/** A name a refusal can cite, with a fallback for unnamed files. */
function fileName(file) {
  return typeof file?.name === "string" && file.name !== "" ? file.name : "this file";
}

/**
 * Why a file cannot be staged on size alone, or null when the size is fine.
 * @param {{ name?: string, size?: number }} file
 * @returns {string | null}
 */
export function oversizeRefusal(file) {
  const size = Number(file?.size ?? 0);
  if (size <= MAX_IMAGE_BYTES) {
    return null;
  }
  const mib = (size / (1024 * 1024)).toFixed(1);
  return `${fileName(file)} is ${mib} MiB — images are capped at ${MAX_IMAGE_LABEL}`;
}

/**
 * Why a file cannot be staged when its leading bytes are not a recognised
 * image — or null when they are. Only the magic is judged here.
 * @param {string} name
 * @param {Uint8Array} bytes
 * @returns {string | null}
 */
export function magicRefusal(name, bytes) {
  if (looksLikeImage(bytes)) {
    return null;
  }
  return `${name} is not an image — only images can be attached right now`;
}

/**
 * Base64-encode bytes without the per-byte String.fromCharCode loop that
 * cost ~346 ms for a 5 MiB upload (measured in #139, F5). Chunking keeps the
 * spread bounded; btoa does the native conversion.
 * @param {Uint8Array} bytes
 * @returns {string}
 */
export function base64FromBytes(bytes) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
}

/**
 * A JS mirror of attachments.DraftAttachments: images staged for the current
 * prompt draft, keyed by `[Image #N]`. Numbering is 1-based and stable for the
 * life of the draft; ids are opaque, never filesystem paths. Deleting a token
 * from the text drops that image from the submit set.
 */
export class DraftAttachments {
  constructor() {
    this.ids = [];
  }

  /** Register an uploaded image id and return the token to insert. */
  add(attachmentId) {
    this.ids.push(attachmentId);
    return imageToken(this.ids.length);
  }

  /** Attachment ids for tokens still present in `text`, in number order. */
  resolve(text) {
    const resolved = [];
    for (const number of parseImageNumbers(text)) {
      const attachmentId = this.ids[number - 1];
      if (attachmentId !== undefined && !resolved.includes(attachmentId)) {
        resolved.push(attachmentId);
      }
    }
    return resolved;
  }

  clear() {
    this.ids = [];
  }
}

/**
 * Draw `text` into `host` as literal text runs with `[Image #N]` token
 * segments as chips. The chip is a rendering of the token in the text, not
 * evidence that an attachment exists — any text can contain a token, so the
 * chip echoes rather than asserts.
 * @param {DomDocument} document
 * @param {DomElement} host
 * @param {string} text
 */
export function renderTokenTextInto(document, host, text) {
  let last = 0;
  IMAGE_TOKEN_RE.lastIndex = 0;
  for (
    let match = IMAGE_TOKEN_RE.exec(text);
    match !== null;
    match = IMAGE_TOKEN_RE.exec(text)
  ) {
    if (match.index > last) {
      host.appendChild(document.createTextNode(text.slice(last, match.index)));
    }
    const chip = document.createElement("span");
    chip.className = "image-chip";
    chip.textContent = match[0];
    host.appendChild(chip);
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    host.appendChild(document.createTextNode(text.slice(last)));
  }
}
