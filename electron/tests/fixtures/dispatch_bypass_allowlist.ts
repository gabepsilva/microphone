/**
 * Planted violation: forwards any action string to the socket without the
 * DISPATCH_ALLOWLIST check. The Semgrep table gate and unit tests must reject
 * this pattern outside the validated registrar path.
 */
import type { TagAlongClient } from "../../src/client";

export function plantBypassDispatch(
  client: Pick<TagAlongClient, "call">,
  action: string,
  payload: Record<string, unknown>,
): Promise<unknown> {
  return client.call("dispatch", { action, payload });
}
