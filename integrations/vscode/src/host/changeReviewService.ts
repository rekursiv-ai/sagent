import * as path from "node:path";

import * as vscode from "vscode";

import type { SagentFileChange, SagentSessionItem } from "../shared/sagentProtocol";
import type {
  ChangePreview,
  ChangePreviewLine,
} from "../shared/changePreviewModel";

/**
 * A file change that Sagent's typed session projection can identify safely.
 *
 * This deliberately carries the Sagent record identity alongside the file
 * path. The structured file result is the preview source; VS Code's native
 * Git/diff APIs remain responsible for opening the file review on click.
 */
export type SagentChangeReviewTarget = {
  readonly filePath: string;
  readonly uri: vscode.Uri;
  readonly toolName: "Edit" | "Write";
  readonly turnId: string;
  readonly recordId: string;
  readonly fileChange: SagentFileChange;
};

/**
 * Return the review target represented by one Sagent session item.
 *
 * Sagent's Edit and Write tools return typed FileEditResult/FileWriteResult
 * records.  The VS Code protocol projects those records onto the originating
 * tool item, so this function does not infer changes from human-readable
 * messages or shell output.  Bash changes remain intentionally outside this
 * boundary until Sagent exposes structured file results for them.
 */
export function toChangeReviewTarget(
  item: SagentSessionItem,
): SagentChangeReviewTarget | undefined {
  if (
    item.type !== "tool"
    || (item.name !== "Edit" && item.name !== "Write")
    || item.status !== "completed"
    || !item.path
    || !item.fileChange
  ) {
    return undefined;
  }

  return {
    filePath: item.path,
    uri: vscode.Uri.file(item.path),
    toolName: item.name,
    turnId: item.turnId,
    recordId: item.id,
    fileChange: item.fileChange,
  };
}

/**
 * Collect completed, typed Sagent file changes without creating a second
 * change tracker.  The latest record for a path is the useful review target
 * when one turn edits the same file more than once.
 */
export function collectChangeReviewTargets(
  items: readonly SagentSessionItem[],
): SagentChangeReviewTarget[] {
  const targetsByPath = new Map<string, SagentChangeReviewTarget>();
  for (const item of items) {
    const target = toChangeReviewTarget(item);
    if (target) {
      targetsByPath.set(target.uri.fsPath, target);
    }
  }
  return [...targetsByPath.values()];
}

type GitChange = {
  readonly uri: vscode.Uri;
  readonly originalUri: vscode.Uri;
};

type ChangeResources = {
  readonly currentUri: vscode.Uri;
  readonly originalUri?: vscode.Uri;
};

type GitRepository = {
  readonly rootUri: vscode.Uri;
  readonly state: {
    readonly workingTreeChanges: readonly GitChange[];
    readonly indexChanges: readonly GitChange[];
    readonly untrackedChanges: readonly GitChange[];
  };
};

type GitApi = {
  readonly repositories: readonly GitRepository[];
};

type GitExtension = {
  getAPI(version: 1): GitApi;
};

/**
 * Opens a Sagent file change using VS Code's native Git-backed diff editor.
 *
 * The Git extension owns repository discovery and original-resource lookup;
 * this adapter only selects the matching native change and invokes the
 * documented `vscode.diff` command.  It never reads file contents or computes
 * a second diff.  A file with no current Git change is opened normally because
 * the Sagent record can outlive a subsequent user commit or cleanup.
 */
export class SagentChangeReviewService {
  public async open(target: SagentChangeReviewTarget): Promise<void> {
    await this.openFile(target.filePath);
  }

  /**
   * Open a file using VS Code's current native repository state.
   *
   * The caller is still responsible for obtaining the path from Sagent's
   * structured file-result record. This method only owns the VS Code Git and
   * diff-editor adapter, so the webview does not need to know Git details.
   */
  public async openFile(filePath: string): Promise<void> {
    const targetUri = vscode.Uri.file(filePath);
    const resources = await this.findChangeResources(filePath);
    if (!resources?.originalUri) {
      await vscode.commands.executeCommand("vscode.open", targetUri);
      return;
    }

    await vscode.commands.executeCommand(
      "vscode.diff",
      resources.originalUri,
      resources.currentUri,
      `Sagent: ${path.basename(filePath)}`,
    );
  }

  /**
   * Build preview data from Sagent's structured file result. The webview never
   * reads workspace files, parses tool text, or depends on Git for previews.
   */
  public async createPreview(target: SagentChangeReviewTarget): Promise<ChangePreview> {
    // Intentionally do not use Git here. Sagent's typed result is the sole
    // preview source for now; the Git resource path remains below only for the
    // native click-to-open behavior and a possible future fallback.
    return createNativeChangePreview(
      target.recordId,
      target.filePath,
      target.fileChange,
    );
  }

  private async findChangeResources(filePath: string): Promise<ChangeResources | undefined> {
    const targetUri = vscode.Uri.file(filePath);
    const gitExtension = vscode.extensions.getExtension<GitExtension>("vscode.git");
    if (!gitExtension) {
      return undefined;
    }

    const git = gitExtension.isActive
      ? gitExtension.exports
      : await gitExtension.activate();
    const repository = git
      ? findRepository(git.getAPI(1).repositories, targetUri.fsPath)
      : undefined;
    if (!repository) {
      return undefined;
    }
    if (findUntrackedChange(repository, targetUri.fsPath)) {
      return { currentUri: targetUri };
    }
    const change = findTrackedChange(repository, targetUri.fsPath);
    return change
      ? { currentUri: change.uri, originalUri: change.originalUri }
      : undefined;
  }
}

function createNativeChangePreview(
  changeId: string,
  filePath: string,
  fileChange: SagentFileChange,
): ChangePreview {
  const lines: ChangePreviewLine[] = [];
  let addedLines = 0;
  let removedLines = 0;
  let oldLineNumber = 1;
  let newLineNumber = 1;

  if (fileChange.kind === "write") {
    for (const text of splitLines(fileChange.content ?? "")) {
      lines.push({ kind: "added", text, newLineNumber });
      newLineNumber += 1;
      addedLines += 1;
    }
  } else {
    for (const edit of fileChange.edits) {
      const start = edit.start ?? newLineNumber;
      oldLineNumber = start;
      newLineNumber = start;
      for (const text of splitLines(edit.before ?? "")) {
        lines.push({ kind: "removed", text, oldLineNumber });
        oldLineNumber += 1;
        removedLines += 1;
      }
      for (const text of splitLines(edit.after ?? "")) {
        lines.push({ kind: "added", text, newLineNumber });
        newLineNumber += 1;
        addedLines += 1;
      }
    }
  }

  return {
    status: "ready",
    changeId,
    path: filePath,
    lines,
    addedLines,
    removedLines,
  };
}

function splitLines(value: string): string[] {
  const lines = value.split(/\r?\n/);
  return lines[lines.length - 1] === "" ? lines.slice(0, -1) : lines;
}

function findRepository(
  repositories: readonly GitRepository[],
  filePath: string,
): GitRepository | undefined {
  return repositories
    .filter((repository) => isWithinRepository(repository.rootUri.fsPath, filePath))
    .sort((left, right) => right.rootUri.fsPath.length - left.rootUri.fsPath.length)[0];
}

function findUntrackedChange(
  repository: GitRepository,
  filePath: string,
): boolean {
  const normalizedFilePath = normalizeFilePath(filePath);
  return repository.state.untrackedChanges.some(
    (change) => normalizeFilePath(change.uri.fsPath) === normalizedFilePath,
  );
}

function findTrackedChange(
  repository: GitRepository,
  filePath: string,
): GitChange | undefined {
  const changes = [
    ...repository.state.workingTreeChanges,
    ...repository.state.indexChanges,
  ];
  const normalizedFilePath = normalizeFilePath(filePath);
  return changes.find(
    (change) => normalizeFilePath(change.uri.fsPath) === normalizedFilePath,
  );
}

function isWithinRepository(repositoryPath: string, filePath: string): boolean {
  const normalizedRepositoryPath = normalizeFilePath(repositoryPath);
  const normalizedFilePath = normalizeFilePath(filePath);
  const relativePath = path.relative(normalizedRepositoryPath, normalizedFilePath);
  return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

function normalizeFilePath(filePath: string): string {
  const normalized = path.normalize(path.resolve(filePath));
  return process.platform === "win32" ? normalized.toLowerCase() : normalized;
}
