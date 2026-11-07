import type { ReactElement } from "react";

import type {
  ChangeReviewGroup,
  ChangeReviewModel,
} from "../../../../src/shared/changeReviewModel";
import type { ChangePreview } from "../../../../src/shared/changePreviewModel";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";
import type {
  ChangeReviewAction,
  ChangeReviewPanelProps,
} from "./changeReviewContract";
import { DiffHoverPreview } from "./DiffHoverPreview";

/**
 * Reusable presentation for Sagent's structured file-result records.
 *
 * This component never opens files and never computes a diff. Its only output
 * is a typed action for the extension host, where VS Code's native Git/diff
 * service remains authoritative.
 */
export function ChangeReviewPanel({
  model,
  onAction,
  onRequestPreview,
  previews,
}: ChangeReviewPanelProps): ReactElement {
  switch (model.status) {
    case "review":
      return (
        <ReviewState
          model={model}
          onAction={onAction}
          onRequestPreview={onRequestPreview}
          previews={previews}
        />
      );
    case "empty":
      return <EmptyState message={model.message} />;
    case "error":
      return (
        <ErrorState
          message={model.message}
          onAction={onAction}
          retryable={model.retryable}
        />
      );
  }
}

function ReviewState({
  model,
  onAction,
  onRequestPreview,
  previews,
}: {
  readonly model: Extract<ChangeReviewModel, { status: "review" }>;
  readonly onAction: (action: ChangeReviewAction) => void;
  readonly onRequestPreview: (changeId: string) => void;
  readonly previews: Readonly<Record<string, ChangePreview>>;
}): ReactElement {
  return (
    <section
      aria-label="Sagent changes"
      className="flex flex-col gap-2 border-b border-sagent-border p-3"
    >
      <header className="flex items-center gap-2 text-sm font-semibold">
        <Icon name="diff" />
        <span>Changes</span>
        <span className="ml-auto text-xs font-normal text-sagent-muted-foreground">
          {model.fileCount} {model.fileCount === 1 ? "file" : "files"}
        </span>
      </header>
      <div className="flex flex-col gap-2">
        {model.groups.map((group) => (
          <ChangeGroup
            key={group.id}
            group={group}
            onAction={onAction}
            onRequestPreview={onRequestPreview}
            previews={previews}
          />
        ))}
      </div>
    </section>
  );
}

function ChangeGroup({
  group,
  onAction,
  onRequestPreview,
  previews,
}: {
  readonly group: ChangeReviewGroup;
  readonly onAction: (action: ChangeReviewAction) => void;
  readonly onRequestPreview: (changeId: string) => void;
  readonly previews: Readonly<Record<string, ChangePreview>>;
}): ReactElement {
  return (
    <section
      aria-label={`${group.agentLabel} ${group.turnId}`}
      className="flex flex-col gap-1"
    >
      <div className="flex items-center gap-2 text-xs font-semibold text-sagent-muted-foreground">
        <Icon name="organization" />
        <span>{group.agentLabel}</span>
        <span aria-hidden="true">·</span>
        <span>{group.turnId}</span>
      </div>
      <ul className="m-0 flex list-none flex-col gap-1 p-0">
        {group.files.map((file) => (
          <li key={file.id}>
            <DiffHoverPreview
              onRequest={() => onRequestPreview(file.id)}
              preview={previews[file.id]}
            >
              <Button
                aria-label={`Open diff for ${file.path}`}
                className="w-full justify-start text-left"
                onClick={() =>
                  onAction({ type: "open-diff", changeId: file.id })
                }
                variant="ghost"
              >
                <Icon name="diff" />
                <span className="min-w-0 flex-1 truncate">{file.path}</span>
                <span className="text-xs text-sagent-muted-foreground">
                  {file.toolName}
                </span>
              </Button>
            </DiffHoverPreview>
          </li>
        ))}
      </ul>
    </section>
  );
}

function EmptyState({ message }: { readonly message: string }): ReactElement {
  return (
    <section
      aria-label="No changes to review"
      className="flex items-center gap-2 p-3 text-sm text-sagent-muted-foreground"
    >
      <Icon name="diff" />
      <span>{message}</span>
    </section>
  );
}

function ErrorState({
  message,
  onAction,
  retryable,
}: {
  readonly message: string;
  readonly onAction: (action: ChangeReviewAction) => void;
  readonly retryable: boolean;
}): ReactElement {
  return (
    <section
      aria-label="Change review error"
      className="flex flex-col gap-2 border-b border-sagent-border p-3"
      role="alert"
    >
      <div className="flex items-start gap-2 text-sm text-[var(--vscode-errorForeground)]">
        <Icon name="diff" />
        <span>{message}</span>
      </div>
      {retryable && (
        <Button
          className="self-start"
          onClick={() => onAction({ type: "retry" })}
          variant="secondary"
        >
          <Icon name="refresh" />
          Retry
        </Button>
      )}
    </section>
  );
}
