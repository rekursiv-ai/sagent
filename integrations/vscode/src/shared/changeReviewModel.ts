/**
 * Serializable presentation data for native Sagent file changes.
 *
 * Sagent owns the edit/write result and its record identity. VS Code owns the
 * file and diff editor. This module only turns the native result projection
 * into data a webview can render; it does not read files or calculate diffs.
 */

export type ChangeReviewToolName = "Edit" | "Write";

/**
 * The structural part of SagentChangeReviewTarget needed by presentation.
 * Keeping this shape free of VS Code types lets the webview consume the model
 * without importing the extension host or recreating the native service.
 */
export type NativeChangeReviewTarget = {
  readonly recordId: string;
  readonly filePath: string;
  readonly toolName: ChangeReviewToolName;
  readonly turnId: string;
};

export type ChangeReviewPresentationTarget = {
  readonly id: string;
  readonly path: string;
  readonly toolName: ChangeReviewToolName;
  readonly turnId: string;
  readonly agentLabel: string;
};

export type ChangeReviewFile = ChangeReviewPresentationTarget;

export type ChangeReviewGroup = {
  readonly id: string;
  readonly agentLabel: string;
  readonly turnId: string;
  readonly files: readonly ChangeReviewFile[];
};

type MutableChangeReviewGroup = {
  readonly id: string;
  readonly agentLabel: string;
  readonly turnId: string;
  readonly files: ChangeReviewFile[];
};

export type ChangeReviewModel =
  | {
      readonly status: "review";
      readonly fileCount: number;
      readonly groups: readonly ChangeReviewGroup[];
    }
  | {
      readonly status: "empty";
      readonly message: string;
    }
  | {
      readonly status: "error";
      readonly message: string;
      readonly retryable: boolean;
    };

/**
 * Adapt one native Sagent change target to the webview's serializable shape.
 * The caller supplies the agent label from Sagent's selected session/team
 * context because a file-result record itself does not own team membership.
 */
export function toChangeReviewPresentationTarget(
  target: NativeChangeReviewTarget,
  agentLabel: string,
): ChangeReviewPresentationTarget {
  return {
    id: target.recordId,
    path: target.filePath,
    toolName: target.toolName,
    turnId: target.turnId,
    agentLabel,
  };
}

/**
 * Build the review read model while preserving Sagent's record order.
 *
 * A nested map keeps agent and turn as separate identities. This matters when
 * two agents use the same turn label in their own journal; no delimiter-based
 * composite key or string parsing is needed to keep those groups distinct.
 */
export function createChangeReviewModel(
  targets: readonly ChangeReviewPresentationTarget[],
): ChangeReviewModel {
  if (targets.length === 0) {
    return {
      status: "empty",
      message: "No file changes are available to review.",
    };
  }

  const groupsByAgent = new Map<string, Map<string, MutableChangeReviewGroup>>();
  const groups: MutableChangeReviewGroup[] = [];

  for (const target of targets) {
    let groupsByTurn = groupsByAgent.get(target.agentLabel);
    if (!groupsByTurn) {
      groupsByTurn = new Map<string, MutableChangeReviewGroup>();
      groupsByAgent.set(target.agentLabel, groupsByTurn);
    }

    let group = groupsByTurn.get(target.turnId);
    if (!group) {
      group = {
        id: `change-group-${groups.length}`,
        agentLabel: target.agentLabel,
        turnId: target.turnId,
        files: [],
      };
      groupsByTurn.set(target.turnId, group);
      groups.push(group);
    }

    group.files.push(target);
  }

  return {
    status: "review",
    fileCount: targets.length,
    groups,
  };
}

export function createChangeReviewError(
  message: string,
  retryable = false,
): ChangeReviewModel {
  return {
    status: "error",
    message,
    retryable,
  };
}
