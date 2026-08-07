export type Shortcut = {
  keys: string;
  label: string;
  id?: string;
};

export const POLICY_ORDER: string[];
export const SHORTCUTS_PROMPT: Shortcut[];
export const SHORTCUTS_SESSION: Shortcut[];

export function nextPolicy(current: string): string;

export function commandForEvent(event: {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
}): string | null;
