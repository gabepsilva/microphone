/**
 * Drag-to-resize the settings sidebar.
 *
 * Width is a CSS variable shared by the grid track and the fixed panel so the
 * chat column never runs underneath. Seat-local preference only — nothing on
 * the socket.
 */

/** 22rem at a 16px root — matches the original fixed track. */
export const DEFAULT_SIDEBAR_WIDTH_PX = 352;
export const MIN_SIDEBAR_WIDTH_PX = 240;
export const MAX_SIDEBAR_WIDTH_PX = 560;
/** Leave at least this much for the chat column. */
export const MIN_CHAT_WIDTH_PX = 320;
export const STORAGE_KEY = "tagalong.sidebarWidth";

/**
 * Clamp a candidate width to the panel's min/max and the viewport.
 * @param {number} px
 * @param {number} viewportWidth
 * @returns {number}
 */
export function clampSidebarWidth(px, viewportWidth) {
  if (!Number.isFinite(px) || !Number.isFinite(viewportWidth)) {
    return DEFAULT_SIDEBAR_WIDTH_PX;
  }
  const maxForViewport = Math.max(
    MIN_SIDEBAR_WIDTH_PX,
    viewportWidth - MIN_CHAT_WIDTH_PX,
  );
  const max = Math.min(MAX_SIDEBAR_WIDTH_PX, maxForViewport);
  return Math.round(Math.min(max, Math.max(MIN_SIDEBAR_WIDTH_PX, px)));
}

/**
 * Read a stored width string into pixels, or null when absent/invalid.
 * @param {unknown} raw
 * @returns {number | null}
 */
export function parseStoredWidth(raw) {
  if (typeof raw !== "string" || raw === "") {
    return null;
  }
  const px = Number(raw);
  if (!Number.isFinite(px) || px <= 0) {
    return null;
  }
  return px;
}

/**
 * @param {HTMLElement} root
 * @param {number} px
 */
export function applySidebarWidth(root, px) {
  root.style.setProperty("--sidebar-width", `${px}px`);
}

/**
 * @param {{ getItem(key: string): string | null, setItem(key: string, value: string): void }} storage
 * @returns {number | null}
 */
export function readStoredWidth(storage) {
  try {
    return parseStoredWidth(storage.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

/**
 * @param {{ setItem(key: string, value: string): void }} storage
 * @param {number} px
 */
export function writeStoredWidth(storage, px) {
  try {
    storage.setItem(STORAGE_KEY, String(px));
  } catch {
    // Quota / private mode — resize still works for the session.
  }
}

/**
 * Wire a drag handle to the body root that owns `--sidebar-width`.
 *
 * @param {{
 *   body: HTMLElement,
 *   handle: HTMLElement,
 *   storage?: { getItem(key: string): string | null, setItem(key: string, value: string): void },
 *   getViewportWidth?: () => number,
 * }} options
 * @returns {() => void} unbind
 */
export function bindSidebarResize(options) {
  const {
    body,
    handle,
    storage = globalThis.localStorage,
    getViewportWidth = () => globalThis.innerWidth,
  } = options;

  const setWidth = (px) => {
    const next = clampSidebarWidth(px, getViewportWidth());
    applySidebarWidth(body, next);
    return next;
  };

  const stored = storage ? readStoredWidth(storage) : null;
  setWidth(stored ?? DEFAULT_SIDEBAR_WIDTH_PX);

  let dragging = false;
  let pointerId = null;

  const onPointerMove = (event) => {
    if (!dragging || event.pointerId !== pointerId) {
      return;
    }
    // Sidebar is docked on the right: distance from the pointer to the
    // window's right edge is the panel width.
    setWidth(getViewportWidth() - event.clientX);
  };

  const endDrag = (event) => {
    if (!dragging || (event && event.pointerId !== pointerId)) {
      return;
    }
    dragging = false;
    pointerId = null;
    body.classList.remove("sidebar-resizing");
    handle.classList.remove("active");
    if (storage) {
      const raw = body.style.getPropertyValue("--sidebar-width").replace("px", "");
      const px = Number(raw);
      if (Number.isFinite(px)) {
        writeStoredWidth(storage, px);
      }
    }
  };

  const onPointerDown = (event) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    dragging = true;
    pointerId = event.pointerId;
    handle.setPointerCapture?.(event.pointerId);
    body.classList.add("sidebar-resizing");
    handle.classList.add("active");
    setWidth(getViewportWidth() - event.clientX);
  };

  const onDoubleClick = (event) => {
    event.preventDefault();
    const next = setWidth(DEFAULT_SIDEBAR_WIDTH_PX);
    if (storage) {
      writeStoredWidth(storage, next);
    }
  };

  handle.addEventListener("pointerdown", onPointerDown);
  handle.addEventListener("pointermove", onPointerMove);
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
  handle.addEventListener("dblclick", onDoubleClick);

  return () => {
    handle.removeEventListener("pointerdown", onPointerDown);
    handle.removeEventListener("pointermove", onPointerMove);
    handle.removeEventListener("pointerup", endDrag);
    handle.removeEventListener("pointercancel", endDrag);
    handle.removeEventListener("dblclick", onDoubleClick);
    endDrag();
  };
}
