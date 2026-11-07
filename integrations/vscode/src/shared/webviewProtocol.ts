import type { ConversationState } from "./conversationTypes";
import type { ChangePreview } from "./changePreviewModel";
import type {
  ModelSettingsAction,
  ModelSettingsScope,
} from "./modelSettings/modelSettingsContract";

export type WebviewToExtensionMessage =
  | { type: "ready" }
  | { type: "send"; text: string }
  | { type: "command"; line: string }
  | { type: "remove-command-output"; id: string }
  | { type: "cancel" }
  | { type: "retry" }
  | { type: "select-session"; path: string }
  | { type: "new-session" }
  | { type: "refresh-sessions" }
  | { type: "select-agent"; label: string }
  | { type: "refresh-team" }
  | { type: "refresh-model-settings" }
  | {
      type: "model-settings";
      scope: ModelSettingsScope;
      action: ModelSettingsAction;
    }
  | { type: "open-change"; changeId: string }
  | { type: "preview-change"; changeId: string }
  | { type: "open-file"; path: string }
  | { type: "review"; path: string };

export type ExtensionToWebviewMessage =
  | { type: "state"; state: ConversationState }
  | { type: "change-preview"; preview: ChangePreview };
