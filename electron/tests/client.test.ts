import { EventEmitter } from "node:events";
import { describe, expect, it } from "bun:test";
import type net from "node:net";

import { SessionEvents, TagAlongClient, socketPath } from "../src/client";
import { applyStateFragment, emptyAppState, parseAppState } from "../src/state";

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

describe("socketPath", () => {
  it("joins XDG_RUNTIME_DIR and refuses a missing runtime dir", () => {
    expect(socketPath({ XDG_RUNTIME_DIR: "/run/user/1000" })).toBe(
      "/run/user/1000/tagalong/tagalong.sock",
    );
    expect(() => socketPath({})).toThrow("XDG_RUNTIME_DIR is unset");
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
});

describe("parseAppState", () => {
  it("falls back per field when the snapshot is malformed", () => {
    const parsed = parseAppState({
      tts_enabled: false,
      microphone: null,
      turn_silence: Number.NaN,
      response_policy: 12,
    });
    expect(parsed.tts_enabled).toBe(false);
    expect(parsed.microphone).toEqual({ desired: null, effective: null });
    expect(parsed.turn_silence).toBe(3.0);
    expect(parsed.response_policy).toBe("both");
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
