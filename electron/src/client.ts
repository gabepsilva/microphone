import net from "node:net";
import path from "node:path";

import { applyStateFragment, emptyAppState, type AppState } from "./state";

export function socketPath(env: NodeJS.ProcessEnv = process.env): string {
  const runtime = env.XDG_RUNTIME_DIR;
  if (!runtime) {
    throw new Error("XDG_RUNTIME_DIR is unset; refusing a /tmp socket");
  }
  return path.join(runtime, "tagalong", "tagalong.sock");
}

export type ConnectFn = (socketPath: string) => net.Socket;

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

type PollResult = {
  events: Array<{ sequence: number; name: string; payload: Record<string, unknown> }>;
  lost: boolean;
};

type SnapshotResult = {
  state: AppState;
  sequence: number;
  instance: string;
  protocol_version: number;
};

/** JSON-RPC client for one TagAlong Unix socket. Injectable connect for tests. */
export class TagAlongClient {
  private _socket: net.Socket | null = null;
  private _connecting: Promise<net.Socket> | null = null;
  private _buffer = "";
  private readonly _pending = new Map<number, Pending>();
  private _nextId = 1;

  constructor(
    private readonly connect: ConnectFn = (sockPath) => net.createConnection(sockPath),
    private readonly resolvePath: () => string = () => socketPath(),
  ) {}

  async call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const socket = await this._ensure();
    const id = this._nextId;
    this._nextId += 1;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    });
  }

  /** Drop the socket so the next call opens a fresh connection. */
  close(): void {
    const socket = this._socket;
    this._failAll(new Error("connection closed"));
    this._reset();
    if (socket !== null && !socket.destroyed) {
      socket.destroy();
    }
  }

  private _ensure(): Promise<net.Socket> {
    if (this._socket && !this._socket.destroyed) {
      return Promise.resolve(this._socket);
    }
    if (this._connecting) {
      return this._connecting;
    }
    this._connecting = new Promise((resolve, reject) => {
      const socket = this.connect(this.resolvePath());
      const fail = (error: Error) => {
        this._failAll(error);
        this._reset();
        reject(error);
      };
      socket.on("error", (error: Error) => fail(error));
      socket.on("close", () => {
        this._failAll(new Error("connection closed"));
        this._reset();
      });
      socket.on("data", (chunk: Buffer | string) => this._onData(chunk));
      socket.on("connect", () => {
        this._socket = socket;
        const id = this._nextId;
        this._nextId += 1;
        this._pending.set(id, {
          resolve: () => {
            this._connecting = null;
            resolve(socket);
          },
          reject: fail,
        });
        socket.write(
          `${JSON.stringify({
            jsonrpc: "2.0",
            id,
            method: "initialize",
            params: { client: "electron" },
          })}\n`,
        );
      });
    });
    return this._connecting;
  }

  private _onData(chunk: Buffer | string): void {
    this._buffer += chunk.toString("utf8");
    let newline = this._buffer.indexOf("\n");
    while (newline !== -1) {
      const line = this._buffer.slice(0, newline);
      this._buffer = this._buffer.slice(newline + 1);
      let payload: {
        id?: number;
        error?: { message: string };
        result?: unknown;
      };
      try {
        payload = JSON.parse(line) as typeof payload;
      } catch (error) {
        this._failAll(error instanceof Error ? error : new Error(String(error)));
        return;
      }
      // Transport is poll-only — the server never pushes ``method: "event"``.
      // Match responses by id only (#96).
      if (payload.id !== undefined) {
        const pending = this._pending.get(payload.id);
        if (pending) {
          this._pending.delete(payload.id);
          if (payload.error) {
            pending.reject(new Error(payload.error.message));
          } else {
            pending.resolve(payload.result);
          }
        }
      }
      newline = this._buffer.indexOf("\n");
    }
  }

  private _failAll(error: Error): void {
    for (const pending of this._pending.values()) {
      pending.reject(error);
    }
    this._pending.clear();
  }

  private _reset(): void {
    this._socket = null;
    this._connecting = null;
    this._buffer = "";
  }
}

export type SessionEventsOptions = {
  /** Long-poll budget in ms. Default 30s. */
  timeoutMs?: number;
  onState: (state: AppState) => void;
  onError?: (error: Error) => void;
};

/**
 * Parked long-poll loop on a dedicated connection.
 *
 * Commands must use a second {@link TagAlongClient}: a parked ``poll`` starves
 * its own socket (transport sequential `_handle`, #96 G1).
 */
export class SessionEvents {
  private _state: AppState = emptyAppState();
  private _stopped = true;
  private _loop: Promise<void> | null = null;
  private readonly timeoutMs: number;
  private readonly onState: (state: AppState) => void;
  private readonly onError: (error: Error) => void;

  constructor(
    private readonly events: TagAlongClient,
    options: SessionEventsOptions,
  ) {
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.onState = options.onState;
    this.onError = options.onError ?? (() => undefined);
  }

  get state(): AppState {
    return this._state;
  }

  async start(): Promise<void> {
    if (!this._stopped) {
      return;
    }
    this._stopped = false;
    await this._subscribe();
    this._loop = this._run();
  }

  stop(): void {
    this._stopped = true;
    this.events.close();
  }

  private async _subscribe(): Promise<void> {
    const snapshot = (await this.events.call("subscribe")) as SnapshotResult;
    this._state = snapshot.state;
    this.onState(this._state);
  }

  private async _run(): Promise<void> {
    while (!this._stopped) {
      try {
        const polled = (await this.events.call("poll", {
          timeout_ms: this.timeoutMs,
        })) as PollResult;
        if (this._stopped) {
          return;
        }
        if (polled.lost) {
          // Terminal per subscription — only subscribe mints a fresh one.
          await this._subscribe();
          continue;
        }
        for (const event of polled.events) {
          if (event.name === "state.changed") {
            this._state = applyStateFragment(this._state, event.payload ?? {});
            this.onState(this._state);
          }
        }
      } catch (error) {
        if (this._stopped) {
          return;
        }
        const err = error instanceof Error ? error : new Error(String(error));
        this.onError(err);
        // Drop the dead socket and resubscribe on a fresh connection.
        this.events.close();
        try {
          await this._subscribe();
        } catch (resubscribeError) {
          const nested =
            resubscribeError instanceof Error
              ? resubscribeError
              : new Error(String(resubscribeError));
          this.onError(nested);
          await delay(250);
        }
      }
    }
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
