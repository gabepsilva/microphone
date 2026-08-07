export type CodexModel = {
  slug: string;
  label: string;
  efforts: string[];
  default_effort: string;
};

export function parseCodexCatalog(value: unknown): CodexModel[];

export function modelOptions(
  models: CodexModel[],
  current: string,
): { value: string; label: string }[];

export function effortOptions(
  models: CodexModel[],
  model: string,
  current: string,
): string[];

export function effortForModel(
  models: CodexModel[],
  model: string,
  current: string,
): string | null;
