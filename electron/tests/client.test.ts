import { EventEmitter } from "node:events";
import { describe, expect, it } from "bun:test";
import type net from "node:net";

import {
  SessionEvents,
  TagAlongClient,
  darwinRuntimeRoot,
  socketPath,
} from "../src/client";
import {
  APP_STATE_KEYS,
  applyStateFragment,
  emptyAppState,
  parseAppState,
  parseTranscriptRows,
  type AppState,
  type TranscriptRow,
} from "../src/state";

class FakeSocket extends EventEmitter {
  destroyed = false;
  readonly written: string[] = [];
  private readonly answered = new Set<number>();

  write(data: string): boolean {
    this.written.push(data);
    return true;
  }

  destroy(): void {
    this.destroyed = true;
    this.emit("close");
  }

  /** Reply to the next pending JSON-RPC request with a result. */
  respond(result: unknown): void {
    const last = this.written.at(-1);
    if (last === undefined) {
      throw new Error("no request to respond to");
    }
    const request = JSON.parse(last) as { id: number };
    this._reply(request.id, result);
  }

  respondTo(method: string, result: unknown): void {
    for (let index = this.written.length - 1; index >= 0; index -= 1) {
      const request = JSON.parse(this.written[index]!) as {
        id: number;
        method: string;
      };
      if (request.method === method && !this.answered.has(request.id)) {
        this._reply(request.id, result);
        return;
      }
    }
    throw new Error(`no pending ${method}`);
  }

  rejectTo(method: string, message: string): void {
    for (let index = this.written.length - 1; index >= 0; index -= 1) {
      const request = JSON.parse(this.written[index]!) as {
        id: number;
        method: string;
      };
      if (request.method === method && !this.answered.has(request.id)) {
        this.answered.add(request.id);
        this.emit(
          "data",
          Buffer.from(
            `${JSON.stringify({
              jsonrpc: "2.0",
              id: request.id,
              error: { message },
            })}\n`,
          ),
        );
        return;
      }
    }
    throw new Error(`no pending ${method}`);
  }

  private _reply(id: number, result: unknown): void {
    this.answered.add(id);
    this.emit(
      "data",
      Buffer.from(`${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`),
    );
  }
}

function connectOnce(fake: FakeSocket): TagAlongClient {
  return new TagAlongClient(
    () => fake as unknown as net.Socket,
    () => "/run/user/1000/tagalong/tagalong.sock",
  );
}

function countMethod(fake: FakeSocket, method: string): number {
  return fake.written.filter((line) => {
    try {
      return (JSON.parse(line) as { method: string }).method === method;
    } catch {
      return false;
    }
  }).length;
}

async function waitForMethodCount(
  fake: FakeSocket,
  method: string,
  count: number,
): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (countMethod(fake, method) >= count) {
      return;
    }
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  throw new Error(`timed out waiting for ${method} x${count}`);
}

async function waitForSocket(
  sockets: FakeSocket[],
  index: number,
): Promise<FakeSocket> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const fake = sockets[index];
    if (fake !== undefined) {
      return fake;
    }
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  throw new Error(`timed out waiting for socket ${index}`);
}

describe("socketPath", () => {
  it("joins XDG_RUNTIME_DIR and refuses a missing runtime dir", () => {
    expect(socketPath({ XDG_RUNTIME_DIR: "/run/user/1000" }, "linux")).toBe(
      "/run/user/1000/tagalong/tagalong.sock",
    );
    expect(() => socketPath({}, "linux")).toThrow("XDG_RUNTIME_DIR is unset");
  });

  it("asks macOS for its own runtime dir instead of refusing", () => {
    // macOS never sets XDG_RUNTIME_DIR, so refusing would mean no socket at all.
    expect(socketPath({}, "darwin", () => "/var/folders/x")).toBe(
      "/var/folders/x/tagalong/tagalong.sock",
    );
  });

  it("prefers an explicit XDG_RUNTIME_DIR even on macOS", () => {
    const unreachable = () => {
      throw new Error("darwin root consulted while XDG_RUNTIME_DIR was set");
    };
    expect(
      socketPath({ XDG_RUNTIME_DIR: "/run/user/1000" }, "darwin", unreachable),
    ).toBe("/run/user/1000/tagalong/tagalong.sock");
  });
});

describe("darwinRuntimeRoot", () => {
  it("reads the kernel value rather than $TMPDIR, and strips the trailing slash", () => {
    // $TMPDIR names the same directory but any caller can repoint it; this
    // value is the socket's parent, so it has to come from the OS.
    const calls: Array<[string, string[]]> = [];
    const root = darwinRuntimeRoot((command, args) => {
      calls.push([command, args]);
      return "/var/folders/br/abc/T/\n";
    });

    expect(root).toBe("/var/folders/br/abc/T");
    expect(calls).toEqual([["getconf", ["DARWIN_USER_TEMP_DIR"]]]);
  });

  it("refuses an empty answer rather than joining onto nothing", () => {
    expect(() => darwinRuntimeRoot(() => "  \n")).toThrow(
      "macOS reported no per-user runtime directory",
    );
  });
});

describe("applyStateFragment", () => {
  it("merges changed fields without mutating the previous state", () => {
    const before = emptyAppState();
    const after = applyStateFragment(before, {
      tts_enabled: false,
      microphone: { desired: "Yeti", effective: null },
    });
    expect(before.tts_enabled).toBe(true);
    expect(after.tts_enabled).toBe(false);
    expect(after.microphone).toEqual({ desired: "Yeti", effective: null });
  });

  it("handles every AppState key so partial_* cannot be dropped by default", () => {
    const before = emptyAppState();
    const changed: Record<string, unknown> = {
      microphone: { desired: "Yeti", effective: "Yeti" },
      microphone_muted: true,
      audio_stream: { desired: "Zoom", effective: null },
      audio_stream_muted: true,
      response_policy: "voice",
      tts_enabled: false,
      tts_provider: { desired: "edge", effective: "edge" },
      tts_voice: { desired: "en_US-amy-medium", effective: "en_US-amy-medium" },
      piper_voice: "en_US-amy-medium",
      edge_voice: "en-US-JennyNeural",
      codex_model: "gpt",
      codex_reasoning: "high",
      codex_thread: "thread-9",
      codex_state: "thinking",
      codex_speaking: true,
      turn_silence: 1.5,
      confidence: 0.83,
      language: "fr",
      moonshine: "small-streaming",
      tokens: 42,
      echoes_cut: 3,
      partial_source: "Voice",
      partial_text: "hello",
    };
    for (const key of APP_STATE_KEYS) {
      expect(key in changed).toBe(true);
    }
    const after = applyStateFragment(before, changed);
    expect(after.partial_source).toBe("Voice");
    expect(after.partial_text).toBe("hello");
    expect(after.turn_silence).toBe(1.5);
    expect(after.codex_thread).toBe("thread-9");
    expect(after.codex_state).toBe("thinking");
    expect(after.codex_speaking).toBe(true);
    expect(after.confidence).toBe(0.83);
    expect(after.language).toBe("fr");
    expect(after.moonshine).toBe("small-streaming");
    expect(after.tokens).toBe(42);
    expect(after.echoes_cut).toBe(3);
    expect(after.microphone).toEqual({ desired: "Yeti", effective: "Yeti" });
    // Exhaustiveness: every key differs from empty defaults after the merge.
    const empty = emptyAppState();
    for (const key of APP_STATE_KEYS) {
      expect(after[key as keyof AppState]).not.toEqual(empty[key as keyof AppState]);
    }
  });
});

describe("parseAppState", () => {
  it("falls back per field when the snapshot is malformed", () => {
    const parsed = parseAppState({
      tts_enabled: false,
      microphone: null,
      turn_silence: Number.NaN,
      response_policy: 12,
      codex_thread: 12,
      codex_state: null,
      codex_speaking: "yes",
      confidence: Number.NaN,
      language: 12,
      moonshine: null,
      tokens: "42",
      echoes_cut: Number.NaN,
    });
    expect(parsed.tts_enabled).toBe(false);
    expect(parsed.microphone).toEqual({ desired: null, effective: null });
    expect(parsed.turn_silence).toBe(3.0);
    expect(parsed.response_policy).toBe("both");
    expect(parsed.codex_thread).toBe("none");
    expect(parsed.codex_state).toBe("idle");
    expect(parsed.codex_speaking).toBe(false);
    expect(parsed.confidence).toBe(0.6);
    expect(parsed.language).toBe("en");
    expect(parsed.moonshine).toBe("medium-streaming");
    expect(parsed.tokens).toBe(0);
    expect(parsed.echoes_cut).toBe(0);
  });

  it("returns empty defaults when the snapshot is not an object", () => {
    expect(parseAppState(null)).toEqual(emptyAppState());
    expect(parseAppState("nope")).toEqual(emptyAppState());
  });

  it("ignores selection objects whose desired/effective are not string|null", () => {
    const parsed = parseAppState({
      microphone: { desired: 1, effective: "ok" },
      audio_stream: { desired: "Zoom", effective: false },
    });
    expect(parsed.microphone).toEqual({ desired: null, effective: null });
    expect(parsed.audio_stream).toEqual({ desired: null, effective: null });
  });
});

describe("parseTranscriptRows", () => {
  it("skips non-arrays and malformed row shapes", () => {
    expect(parseTranscriptRows(null)).toEqual([]);
    expect(parseTranscriptRows("x")).toEqual([]);
    expect(
      parseTranscriptRows([
        null,
        { id: "1", entry: { text: "no" } },
        { id: 1, entry: null },
        {
          id: 2,
          entry: {
            kind: "Voice",
            source: "mic",
            text: "ok",
            output: ["a", 1, "b"],
            exit_code: 0,
            seconds: 1.5,
          },
        },
      ]),
    ).toEqual([
      {
        id: 2,
        provisional: false,
        entry: {
          kind: "Voice",
          source: "mic",
          text: "ok",
          stamp: "",
          reply_to: "",
          interrupted: false,
          output: ["a", "b"],
          exit_code: 0,
          streaming: false,
          seconds: 1.5,
        },
      },
    ]);
  });

  it("keeps the provisional flag from the wire envelope", () => {
    expect(
      parseTranscriptRows([
        {
          id: 1,
          provisional: true,
          entry: { kind: "speech", source: "Voice", text: "pending" },
        },
      ]),
    ).toEqual([
      {
        id: 1,
        provisional: true,
        entry: {
          kind: "speech",
          source: "Voice",
          text: "pending",
          stamp: "",
          reply_to: "",
          interrupted: false,
          output: [],
          exit_code: null,
          streaming: false,
          seconds: null,
        },
      },
    ]);
  });
});

describe("TagAlongClient", () => {
  it("initializes then dispatches a call over a fake socket", async () => {
    const fake = new FakeSocket();
    const client = connectOnce(fake);

    queueMicrotask(() => {
      fake.emit("connect");
      fake.respond({});
    });

    const callPromise = client.call("snapshot");
    await Promise.resolve();
    await Promise.resolve();
    fake.respond({ state: { tts_enabled: true } });

    await expect(callPromise).resolves.toEqual({ state: { tts_enabled: true } });
    expect(fake.written[0]).toContain('"method":"initialize"');
    expect(fake.written[1]).toContain('"method":"snapshot"');
  });

  it("rejects pending calls when the socket closes", async () => {
    const fake = new FakeSocket();
    const client = connectOnce(fake);

    queueMicrotask(() => {
      fake.emit("connect");
      fake.respond({});
    });

    const callPromise = client.call("snapshot");
    await Promise.resolve();
    await Promise.resolve();
    fake.emit("close");

    await expect(callPromise).rejects.toThrow("connection closed");
  });

  it("rejects the handshake when the socket errors before connect", async () => {
    const fake = new FakeSocket();
    const client = connectOnce(fake);

    queueMicrotask(() => {
      fake.emit("error", new Error("ECONNREFUSED"));
    });

    await expect(client.call("snapshot")).rejects.toThrow("ECONNREFUSED");
  });

  it("resolves responses by id even if a stray method field is present", async () => {
    const fake = new FakeSocket();
    const client = connectOnce(fake);

    queueMicrotask(() => {
      fake.emit("connect");
      fake.respond({});
    });

    const callPromise = client.call("snapshot");
    await Promise.resolve();
    await Promise.resolve();
    const request = JSON.parse(fake.written.at(-1)!) as { id: number };
    fake.emit(
      "data",
      Buffer.from(
        `${JSON.stringify({
          jsonrpc: "2.0",
          id: request.id,
          method: "event",
          result: { ok: true },
        })}\n`,
      ),
    );

    await expect(callPromise).resolves.toEqual({ ok: true });
  });

  it("reconnects on the next call after close", async () => {
    const sockets: FakeSocket[] = [];
    const client = new TagAlongClient(
      () => {
        const fake = new FakeSocket();
        sockets.push(fake);
        return fake as unknown as net.Socket;
      },
      () => "/run/user/1000/tagalong/tagalong.sock",
    );

    queueMicrotask(() => {
      sockets[0]!.emit("connect");
      sockets[0]!.respond({});
    });
    const first = client.call("snapshot");
    await Promise.resolve();
    await Promise.resolve();
    sockets[0]!.respond({ n: 1 });
    await first;

    client.close();
    expect(sockets[0]!.destroyed).toBe(true);

    queueMicrotask(() => {
      sockets[1]!.emit("connect");
      sockets[1]!.respond({});
    });
    const second = client.call("snapshot");
    await Promise.resolve();
    await Promise.resolve();
    sockets[1]!.respond({ n: 2 });
    await expect(second).resolves.toEqual({ n: 2 });
    expect(sockets).toHaveLength(2);
  });
});

describe("SessionEvents", () => {
  it("applies state.changed from a long-poll and resubscribes after lost", async () => {
    const fake = new FakeSocket();
    const events = connectOnce(fake);
    const states: Array<{ tts_enabled: boolean }> = [];

    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: (state) => {
        states.push({ tts_enabled: state.tts_enabled });
      },
    });

    const started = session.start();
    await Promise.resolve();
    fake.emit("connect");
    await waitForMethodCount(fake, "initialize", 1);
    fake.respondTo("initialize", {});
    await waitForMethodCount(fake, "subscribe", 1);
    fake.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: { ...emptyAppState(), tts_enabled: true },
    });
    await started;
    expect(states.at(-1)?.tts_enabled).toBe(true);
    // Default onError (omitted above) plus public getters used by main.ts.
    expect(session.hasSnapshot).toBe(true);
    expect(session.state.tts_enabled).toBe(true);

    await waitForMethodCount(fake, "poll", 1);
    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 1,
          name: "state.changed",
          payload: { tts_enabled: false },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 2);
    expect(states.at(-1)?.tts_enabled).toBe(false);

    // Overflow: lost is terminal; recovery is subscribe again.
    fake.respondTo("poll", { lost: true, events: [] });
    await waitForMethodCount(fake, "subscribe", 2);
    fake.respondTo("subscribe", {
      instance: "abc",
      sequence: 9,
      protocol_version: 1,
      state: { ...emptyAppState(), tts_enabled: true },
    });
    await waitForMethodCount(fake, "poll", 3);
    expect(states.at(-1)?.tts_enabled).toBe(true);

    session.stop();
  });

  it("surfaces action.failed for async tray failures (#128b)", async () => {
    const fake = new FakeSocket();
    const events = connectOnce(fake);
    const failures: Array<{ action: string; detail: string }> = [];

    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: () => undefined,
      onActionFailed: (event) => {
        failures.push({ action: event.action, detail: event.detail });
      },
    });

    const started = session.start();
    await Promise.resolve();
    fake.emit("connect");
    await waitForMethodCount(fake, "initialize", 1);
    fake.respondTo("initialize", {});
    await waitForMethodCount(fake, "subscribe", 1);
    fake.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: emptyAppState(),
    });
    await started;

    await waitForMethodCount(fake, "poll", 1);
    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 1,
          name: "action.failed",
          payload: {
            request_id: "req-9",
            action: "speech.read_selection",
            actor: "electron-1",
            detail: "Primary selection is empty",
          },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 2);
    expect(failures).toEqual([
      {
        action: "speech.read_selection",
        detail: "Primary selection is empty",
      },
    ]);

    // Malformed payloads must not invent a failure event.
    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 2,
          name: "action.failed",
          payload: { request_id: 9, action: "speech.read_selection" },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 3);
    expect(failures).toHaveLength(1);

    session.stop();
  });

  it("ignores action.failed when no onActionFailed listener is installed", async () => {
    const fake = new FakeSocket();
    const events = connectOnce(fake);
    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: () => undefined,
    });
    const started = session.start();
    await Promise.resolve();
    fake.emit("connect");
    await waitForMethodCount(fake, "initialize", 1);
    fake.respondTo("initialize", {});
    await waitForMethodCount(fake, "subscribe", 1);
    fake.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: emptyAppState(),
    });
    await started;
    await waitForMethodCount(fake, "poll", 1);
    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 1,
          name: "action.failed",
          payload: {
            request_id: "req-1",
            action: "speech.read_selection",
            detail: "ignored without listener",
          },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 2);
    session.stop();
  });

  it("seeds transcript from subscribe and applies entry events", async () => {
    const fake = new FakeSocket();
    const events = connectOnce(fake);
    const snapshots: TranscriptRow[][] = [];
    const wireEvents: Array<{ name: string }> = [];

    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: () => undefined,
      onTranscriptSnapshot: (rows) => {
        snapshots.push(rows);
      },
      onTranscriptEvent: (event) => {
        wireEvents.push({ name: event.name });
      },
    });

    const started = session.start();
    await Promise.resolve();
    fake.emit("connect");
    await waitForMethodCount(fake, "initialize", 1);
    fake.respondTo("initialize", {});
    await waitForMethodCount(fake, "subscribe", 1);
    fake.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: emptyAppState(),
      transcript: [
        {
          id: 1,
          entry: {
            kind: "Voice",
            source: "mic",
            text: "hi",
            stamp: "",
            reply_to: "",
            interrupted: false,
            output: [],
            exit_code: null,
            streaming: false,
            seconds: null,
          },
        },
      ],
    });
    await started;
    expect(snapshots.at(-1)?.map((row) => row.id)).toEqual([1]);
    expect(session.transcript.map((row) => row.id)).toEqual([1]);

    await waitForMethodCount(fake, "poll", 1);
    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 1,
          name: "transcript.entry_added",
          payload: {
            id: 2,
            entry: {
              kind: "Taga",
              source: "codex",
              text: "hello",
              stamp: "",
              reply_to: "",
              interrupted: false,
              output: [],
              exit_code: null,
              streaming: false,
              seconds: null,
            },
          },
        },
        {
          sequence: 2,
          name: "transcript.entry_updated",
          payload: {
            id: 2,
            entry: {
              kind: "Taga",
              source: "codex",
              text: "hello world",
              stamp: "",
              reply_to: "",
              interrupted: false,
              output: [],
              exit_code: null,
              streaming: false,
              seconds: null,
            },
          },
        },
        {
          sequence: 3,
          name: "state.changed",
          payload: { partial_source: "Voice", partial_text: "…" },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 2);
    expect(wireEvents.map((event) => event.name)).toEqual([
      "transcript.entry_added",
      "transcript.entry_updated",
    ]);
    expect(session.transcript.map((row) => row.entry.text)).toEqual([
      "hi",
      "hello world",
    ]);
    expect(session.state.partial_text).toBe("…");

    fake.respondTo("poll", {
      lost: false,
      events: [
        {
          sequence: 4,
          name: "transcript.entry_removed",
          payload: { id: 2 },
        },
      ],
    });
    await waitForMethodCount(fake, "poll", 3);
    expect(session.transcript.map((row) => row.entry.text)).toEqual(["hi"]);
    expect(wireEvents.at(-1)?.name).toBe("transcript.entry_removed");

    fake.respondTo("poll", {
      lost: false,
      events: [{ sequence: 5, name: "transcript.cleared", payload: {} }],
    });
    await waitForMethodCount(fake, "poll", 4);
    expect(session.transcript).toEqual([]);
    expect(wireEvents.at(-1)?.name).toBe("transcript.cleared");

    // Malformed / unknown transcript events are ignored.
    fake.respondTo("poll", {
      lost: false,
      events: [
        { sequence: 6, name: "transcript.entry_added", payload: { id: "bad" } },
        { sequence: 7, name: "transcript.other", payload: {} },
      ],
    });
    await waitForMethodCount(fake, "poll", 5);
    expect(session.transcript).toEqual([]);

    session.stop();
  });

  it("recovers from poll errors and backs off when resubscribe fails", async () => {
    const sockets: FakeSocket[] = [];
    const events = new TagAlongClient(
      () => {
        const fake = new FakeSocket();
        sockets.push(fake);
        queueMicrotask(() => {
          fake.emit("connect");
        });
        return fake as unknown as net.Socket;
      },
      () => "/run/user/1000/tagalong/tagalong.sock",
    );
    const errors: string[] = [];
    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: () => undefined,
      onError: (error) => {
        errors.push(error.message);
      },
    });

    const started = session.start();
    const first = await waitForSocket(sockets, 0);
    await waitForMethodCount(first, "initialize", 1);
    first.respondTo("initialize", {});
    await waitForMethodCount(first, "subscribe", 1);
    first.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: emptyAppState(),
      transcript: [],
    });
    await started;
    await waitForMethodCount(first, "poll", 1);

    // Poll failure closes the socket; reconnect + subscribe recovers.
    first.rejectTo("poll", "poll blew up");
    const second = await waitForSocket(sockets, 1);
    await waitForMethodCount(second, "initialize", 1);
    second.respondTo("initialize", {});
    await waitForMethodCount(second, "subscribe", 1);
    second.respondTo("subscribe", {
      instance: "abc",
      sequence: 1,
      protocol_version: 1,
      state: { ...emptyAppState(), tts_enabled: false },
      transcript: [],
    });
    await waitForMethodCount(second, "poll", 1);
    expect(errors).toContain("poll blew up");
    expect(session.state.tts_enabled).toBe(false);

    // Resubscribe failure hits the backoff delay path.
    second.rejectTo("poll", "again");
    const third = await waitForSocket(sockets, 2);
    await waitForMethodCount(third, "initialize", 1);
    third.respondTo("initialize", {});
    await waitForMethodCount(third, "subscribe", 1);
    third.rejectTo("subscribe", "subscribe denied");
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(errors).toContain("subscribe denied");

    session.stop();
  });

  it("stop then start leaves only one poll loop", async () => {
    const sockets: FakeSocket[] = [];
    const events = new TagAlongClient(
      () => {
        const fake = new FakeSocket();
        sockets.push(fake);
        return fake as unknown as net.Socket;
      },
      () => "/run/user/1000/tagalong/tagalong.sock",
    );
    const session = new SessionEvents(events, {
      timeoutMs: 50,
      onState: () => undefined,
    });

    const first = session.start();
    await Promise.resolve();
    sockets[0]!.emit("connect");
    await waitForMethodCount(sockets[0]!, "initialize", 1);
    sockets[0]!.respondTo("initialize", {});
    await waitForMethodCount(sockets[0]!, "subscribe", 1);
    sockets[0]!.respondTo("subscribe", {
      instance: "abc",
      sequence: 0,
      protocol_version: 1,
      state: emptyAppState(),
    });
    await first;
    await waitForMethodCount(sockets[0]!, "poll", 1);

    session.stop();
    const second = session.start();
    await Promise.resolve();
    sockets[1]!.emit("connect");
    await waitForMethodCount(sockets[1]!, "initialize", 1);
    sockets[1]!.respondTo("initialize", {});
    await waitForMethodCount(sockets[1]!, "subscribe", 1);
    sockets[1]!.respondTo("subscribe", {
      instance: "abc",
      sequence: 1,
      protocol_version: 1,
      state: emptyAppState(),
    });
    await second;
    await waitForMethodCount(sockets[1]!, "poll", 1);

    // Old loop must not keep polling the first socket after stop.
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(countMethod(sockets[0]!, "poll")).toBe(1);
    expect(countMethod(sockets[1]!, "poll")).toBe(1);

    session.stop();
  });
});
