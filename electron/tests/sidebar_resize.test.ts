import { describe, expect, it } from "bun:test";

import {
  DEFAULT_SIDEBAR_WIDTH_PX,
  MAX_SIDEBAR_WIDTH_PX,
  MIN_CHAT_WIDTH_PX,
  MIN_SIDEBAR_WIDTH_PX,
  STORAGE_KEY,
  applySidebarWidth,
  bindSidebarResize,
  clampSidebarWidth,
  parseStoredWidth,
  readStoredWidth,
  writeStoredWidth,
} from "../renderer/sidebar_resize.js";

describe("clampSidebarWidth", () => {
  it("keeps widths inside the panel min/max", () => {
    expect(clampSidebarWidth(100, 1200)).toBe(MIN_SIDEBAR_WIDTH_PX);
    expect(clampSidebarWidth(900, 1200)).toBe(MAX_SIDEBAR_WIDTH_PX);
    expect(clampSidebarWidth(400, 1200)).toBe(400);
  });

  it("leaves room for the chat column on a narrow window", () => {
    const viewport = MIN_CHAT_WIDTH_PX + 280;
    expect(clampSidebarWidth(500, viewport)).toBe(280);
  });

  it("falls back to the default when the numbers are not finite", () => {
    expect(clampSidebarWidth(Number.NaN, 1200)).toBe(DEFAULT_SIDEBAR_WIDTH_PX);
    expect(clampSidebarWidth(400, Number.NaN)).toBe(DEFAULT_SIDEBAR_WIDTH_PX);
  });
});

describe("parseStoredWidth", () => {
  it("accepts a positive numeric string and rejects the rest", () => {
    expect(parseStoredWidth("400")).toBe(400);
    expect(parseStoredWidth("")).toBeNull();
    expect(parseStoredWidth("nope")).toBeNull();
    expect(parseStoredWidth(null)).toBeNull();
    expect(parseStoredWidth("-1")).toBeNull();
  });
});

describe("storage helpers", () => {
  it("round-trips a width through a map-backed store", () => {
    const data = new Map<string, string>();
    const storage = {
      getItem: (key: string): string | null =>
        data.has(key) ? (data.get(key) ?? null) : null,
      setItem: (key: string, value: string): void => {
        data.set(key, value);
      },
    };
    writeStoredWidth(storage, 400);
    expect(data.get(STORAGE_KEY)).toBe("400");
    expect(readStoredWidth(storage)).toBe(400);
  });

  it("survives a throwing store", () => {
    const broken = {
      getItem: (): string | null => {
        throw new Error("blocked");
      },
      setItem: (_key: string, _value: string): void => {
        throw new Error("blocked");
      },
    };
    expect(readStoredWidth(broken)).toBeNull();
    writeStoredWidth(broken, 400);
  });
});

describe("applySidebarWidth", () => {
  it("writes the CSS variable in pixels", () => {
    const props = new Map<string, string>();
    applySidebarWidth(
      {
        style: {
          setProperty: (name: string, value: string): void => {
            props.set(name, value);
          },
        },
      },
      400,
    );
    expect(props.get("--sidebar-width")).toBe("400px");
  });
});

describe("bindSidebarResize", () => {
  type Listener = (event: {
    button?: number;
    pointerId?: number;
    clientX?: number;
    preventDefault?: () => void;
  }) => void;

  function fakeElement() {
    const listeners = new Map<string, Listener>();
    const classes = new Set<string>();
    const props = new Map<string, string>();
    return {
      style: {
        setProperty: (name: string, value: string): void => {
          props.set(name, value);
        },
        getPropertyValue: (name: string): string => props.get(name) ?? "",
      },
      classList: {
        add: (name: string): void => {
          classes.add(name);
        },
        remove: (name: string): void => {
          classes.delete(name);
        },
        has: (name: string): boolean => classes.has(name),
      },
      addEventListener: (name: string, fn: Listener): void => {
        listeners.set(name, fn);
      },
      removeEventListener: (name: string): void => {
        listeners.delete(name);
      },
      setPointerCapture: (_pointerId?: number): void => undefined,
      emit: (name: string, event: Parameters<Listener>[0]): void => {
        listeners.get(name)?.(event);
      },
      props,
      classes,
    };
  }

  it("restores a stored width, drags from the right edge, and persists", () => {
    const body = fakeElement();
    const handle = fakeElement();
    const data = new Map<string, string>([[STORAGE_KEY, "400"]]);
    const storage = {
      getItem: (key: string): string | null =>
        data.has(key) ? (data.get(key) ?? null) : null,
      setItem: (key: string, value: string): void => {
        data.set(key, value);
      },
    };
    const unbind = bindSidebarResize({
      body,
      handle,
      storage,
      getViewportWidth: () => 1000,
    });
    expect(body.props.get("--sidebar-width")).toBe("400px");

    handle.emit("pointerdown", {
      button: 0,
      pointerId: 1,
      clientX: 550,
      preventDefault() {},
    });
    expect(body.classes.has("sidebar-resizing")).toBe(true);
    expect(body.props.get("--sidebar-width")).toBe("450px");

    handle.emit("pointermove", { pointerId: 1, clientX: 500 });
    expect(body.props.get("--sidebar-width")).toBe("500px");

    handle.emit("pointerup", { pointerId: 1 });
    expect(body.classes.has("sidebar-resizing")).toBe(false);
    expect(data.get(STORAGE_KEY)).toBe("500");

    handle.emit("dblclick", { preventDefault() {} });
    expect(body.props.get("--sidebar-width")).toBe(`${DEFAULT_SIDEBAR_WIDTH_PX}px`);
    expect(data.get(STORAGE_KEY)).toBe(String(DEFAULT_SIDEBAR_WIDTH_PX));

    unbind();
  });
});
