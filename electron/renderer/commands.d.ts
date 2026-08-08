export type CommandSpec = {
  name: string;
  summary: string;
  aliases: string[];
  action_id: string | null;
};

export function commandQuery(text: string): string | null;

export function specScore(spec: CommandSpec, query: string): number[] | null;

export function matchCommands(specs: CommandSpec[], query: string): CommandSpec[];

export function parseCommandSpec(value: unknown): CommandSpec | null;

export function parseCommandList(value: unknown): CommandSpec[];

export function clampIndex(index: number, length: number): number;

export function findCommand(specs: CommandSpec[], text: string): CommandSpec | null;

export function slashArguments(text: string): string[];

export function detailLine(spec: CommandSpec): string;

export type SubmitDecision =
  | { kind: "command"; spec: CommandSpec; args: string[] }
  | { kind: "message"; text: string }
  | { kind: "info"; text: string }
  | { kind: "error"; text: string };

export function decideSubmit(rawText: string, catalog: CommandSpec[]): SubmitDecision;
