/** Frozen preload surface. Types cannot express "do not add a key". */
export const PRELOAD_ALLOWLIST = ["setTts", "snapshot"] as const;

export function allowlistKeys(exposed: object): string[] {
  return Object.keys(exposed).sort();
}

export function matchesAllowlist(exposed: object): boolean {
  const keys = allowlistKeys(exposed);
  return (
    keys.length === PRELOAD_ALLOWLIST.length &&
    keys.every((key, index) => key === [...PRELOAD_ALLOWLIST].sort()[index])
  );
}
