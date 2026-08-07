import net from "node:net";
import path from "node:path";

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

/** JSON-RPC client for the TagAlong Unix socket. Injectable connect for tests. */
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
        method?: string;
        error?: { message: string };
        result?: unknown;
      };
      try {
        payload = JSON.parse(line) as typeof payload;
      } catch (error) {
        this._failAll(error instanceof Error ? error : new Error(String(error)));
        return;
      }
      if (payload.method !== "event" && payload.id !== undefined) {
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
