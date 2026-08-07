import { EventEmitter } from "node:events";
import { describe, expect, it } from "bun:test";
import type net from "node:net";

import { TagAlongClient, socketPath } from "../src/client";

class FakeSocket extends EventEmitter {
  destroyed = false;
  readonly written: string[] = [];

  write(data: string): boolean {
    this.written.push(data);
    return true;
  }

  destroy(): void {
    this.destroyed = true;
  }

  /** Reply to the next pending JSON-RPC request with a result. */
  respond(result: unknown): void {
    const last = this.written.at(-1);
    if (last === undefined) {
      throw new Error("no request to respond to");
    }
    const request = JSON.parse(last) as { id: number };
    this.emit(
      "data",
      Buffer.from(`${JSON.stringify({ jsonrpc: "2.0", id: request.id, result })}\n`),
    );
  }
}

describe("socketPath", () => {
  it("joins XDG_RUNTIME_DIR and refuses a missing runtime dir", () => {
    expect(socketPath({ XDG_RUNTIME_DIR: "/run/user/1000" })).toBe(
      "/run/user/1000/tagalong/tagalong.sock",
    );
    expect(() => socketPath({})).toThrow("XDG_RUNTIME_DIR is unset");
  });
});

describe("TagAlongClient", () => {
  it("initializes then dispatches a call over a fake socket", async () => {
    const fake = new FakeSocket();
    const client = new TagAlongClient(
      () => fake as unknown as net.Socket,
      () => "/run/user/1000/tagalong/tagalong.sock",
    );

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
    const client = new TagAlongClient(
      () => fake as unknown as net.Socket,
      () => "/run/user/1000/tagalong/tagalong.sock",
    );

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
});
