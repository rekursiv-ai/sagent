import type { ChangeReviewModel } from "../../../../src/shared/changeReviewModel";
import type { ChangePreview } from "../../../../src/shared/changePreviewModel";

/**
 * Actions emitted by the reusable review panel.
 *
 * The panel intentionally emits a Sagent record ID instead of a path. The
 * extension host must resolve that ID to the native change-review target and
 * invoke SagentChangeReviewService, keeping Git and diff ownership in VS Code.
 */
export type ChangeReviewAction =
  | { readonly type: "open-diff"; readonly changeId: string }
  | { readonly type: "retry" };

export type ChangeReviewPanelProps = {
  readonly model: ChangeReviewModel;
  readonly onAction: (action: ChangeReviewAction) => void;
  readonly onRequestPreview: (changeId: string) => void;
  readonly previews: Readonly<Record<string, ChangePreview>>;
};
