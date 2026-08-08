export const DEFAULT_SIDEBAR_WIDTH_PX: number;
export const MIN_SIDEBAR_WIDTH_PX: number;
export const MAX_SIDEBAR_WIDTH_PX: number;
export const MIN_CHAT_WIDTH_PX: number;
export const STORAGE_KEY: string;

export function clampSidebarWidth(px: number, viewportWidth: number): number;

export function parseStoredWidth(raw: unknown): number | null;

export function applySidebarWidth(
  root: { style: { setProperty(name: string, value: string): void } },
  px: number,
): void;

export function readStoredWidth(storage: {
  getItem(key: string): string | null;
}): number | null;

export function writeStoredWidth(
  storage: { setItem(key: string, value: string): void },
  px: number,
): void;

export type SidebarResizeRoot = {
  style: {
    setProperty(name: string, value: string): void;
  };
  classList: {
    add(name: string): void;
    remove(name: string): void;
  };
};

export type SidebarResizeHandle = {
  addEventListener(name: string, fn: (event: unknown) => void): void;
  removeEventListener(name: string, fn: (event: unknown) => void): void;
  setPointerCapture?(pointerId: number): void;
  classList: {
    add(name: string): void;
    remove(name: string): void;
  };
};

export function bindSidebarResize(options: {
  body: SidebarResizeRoot;
  handle: SidebarResizeHandle;
  storage?: {
    getItem(key: string): string | null;
    setItem(key: string, value: string): void;
  };
  getViewportWidth?: () => number;
}): () => void;
