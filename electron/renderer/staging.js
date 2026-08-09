/**
 * One staging loop for paste, drop, and picker — #139 D2.
 *
 * clipboardData.files and dataTransfer.files are the same construct, so a
 * single loop preflights, uploads, and stages whatever array of File objects
 * an input event carries. Preflight happens *before* upload (D6, F1): an
 * oversized file must never reach the transport, which would hang up the
 * socket rather than refuse politely (transport.py:284).
 *
 * The caller owns the DOM: this module returns per-file results and the
 * caller turns tokens into chips and refusals into banner copy.
 */

import { magicRefusal, oversizeRefusal } from "./attachments.js";

/** How many leading bytes the magic sniff needs (RIFF/WEBP reads 12). */
const MAGIC_READ_BYTES = 12;

/**
 * Why `file` cannot be staged for an upload, or null when it can. Reads only
 * the leading bytes — an oversized payload never touches the sockets.
 *
 * @param {File} file
 * @returns {Promise<string | null>}
 */
export async function refusalForFile(file) {
  const oversize = oversizeRefusal(file);
  if (oversize !== null) {
    return oversize;
  }
  const name = typeof file.name === "string" ? file.name : "";
  const head = new Uint8Array(await file.slice(0, MAGIC_READ_BYTES).arrayBuffer());
  return magicRefusal(name, head);
}

/**
 * Stage every file an input event carried. Each file either uploads and
 * returns its id, or comes back with the reason the caller should show.
 * `id` stays null when the upload declined after bytes were read — the
 * caller's upload hook reports its own error details.
 *
 * @param {ArrayLike<File>} files
 * @param {{ (bytes: Uint8Array): Promise<string | null> }} upload
 * @returns {Promise<Array<{ file: File, id?: string | null, refused?: string | null }>>}
 */
export async function stageFiles(files, upload) {
  const results = [];
  for (const file of files) {
    const refused = await refusalForFile(file);
    if (refused !== null) {
      results.push({ file, refused });
      continue;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    const id = await upload(bytes);
    results.push({ file, id, refused: null });
  }
  return results;
}
