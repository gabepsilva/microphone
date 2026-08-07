/**
 * Sandboxed preload scripts may only require("electron") (and a short Electron
 * allowlist). Relative requires fail at runtime with
 * "module not found: ./…", leaving window.tagalong undefined.
 */
export function hasRelativeRequire(source: string): boolean {
  return /require\s*\(\s*["']\.[^"']*["']\s*\)/.test(source);
}
