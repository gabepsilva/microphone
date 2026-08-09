export type StagedFileResult = {
  file: File;
  id?: string | null;
  refused?: string | null;
};

export function refusalForFile(file: File): Promise<string | null>;

export function stageFiles(
  files: ArrayLike<File>,
  upload: (bytes: Uint8Array) => Promise<string | null>,
): Promise<StagedFileResult[]>;
