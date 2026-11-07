/**
 * Typed data and intents for the provider/model settings surface.
 *
 * Sagent owns the catalog, authentication state, model capabilities, and
 * mutations. These types describe the native-shaped snapshot that a future
 * server method can serialize; they do not contain a provider/model registry
 * or a second settings implementation.
 */

export type SagentThinkingEffort =
  | "none"
  | "min"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max";

export type SagentThinkingBudget = "none" | "auto" | "fixed";

export type SagentThinkingOutput = "none" | "text" | "redacted";

export type SagentServiceTier =
  | "auto"
  | "default"
  | "flex"
  | "priority"
  | "ultrafast";

export type SagentContextSize = {
  /** The native context name, such as `fullcontext` or `258k`. */
  name: string;
  /** The resolved token count returned by Sagent. */
  tokens: number;
  /** The display name Sagent uses for this context size. */
  displayName: string;
  isDefault: boolean;
};

export type SagentModelSettingsValues = {
  context: SagentContextSize;
  thinkingEffort: SagentThinkingEffort;
  thinkingBudget: SagentThinkingBudget;
  thinkingOutput: SagentThinkingOutput;
  serviceTier: SagentServiceTier;
  cacheTtlSeconds: number;
  manageContextServerSide: boolean;
};

export type SagentModelCapability = {
  /** Sagent's exact model identifier, not a display-only label. */
  modelId: string;
  contextSizes: readonly SagentContextSize[];
  thinkingEfforts: readonly SagentThinkingEffort[];
  thinkingBudgets: readonly SagentThinkingBudget[];
  thinkingOutputs: readonly SagentThinkingOutput[];
  serviceTiers: readonly SagentServiceTier[];
  cacheTtlSeconds: readonly number[];
  manageContextServerSide: readonly boolean[];
  defaults: SagentModelSettingsValues;
  limits: SagentModelLimits;
};

export type SagentModelLimits = {
  maxRequestTokens: number;
  maxResponseTokens: number;
  maxContextTokens: number;
};

export type SagentProviderState = {
  /** Exact names from `ModelCatalog.factories`, in Sagent's order. */
  providers: readonly string[];
  currentProvider: string;
  /** Accounts exposed by the current provider's native `Logins` object. */
  accounts: readonly string[];
  currentAccount: string;
};

export type SagentModelState = {
  /** Exact model IDs yielded by the selected native provider. */
  models: readonly string[];
  currentModel: string;
};

export type SagentModelDiagnostics = {
  label: string;
  model: string;
  context: string;
  thinking: {
    budget: SagentThinkingBudget;
    output: SagentThinkingOutput;
    effort: SagentThinkingEffort;
  };
  serviceTier: SagentServiceTier;
  cacheTtlSeconds: number;
  limits: SagentModelLimits;
  cwd: string;
  tools: readonly string[];
};

export type ModelSettingsScope = "current-agent" | "new-session-default";

export type ModelSettingsSnapshot = {
  status: "loading" | "ready" | "error";
  errorMessage?: string;
  /** The live agent label whose model/settings are represented. */
  agentLabel: string;
  /** Scopes actually supported by the server response. */
  availableScopes: readonly ModelSettingsScope[];
  selectedScope: ModelSettingsScope;
  provider: SagentProviderState;
  model: SagentModelState;
  capability: SagentModelCapability;
  settings: SagentModelSettingsValues;
  /** Settings the current native host can apply through AgentSelf. */
  writableSettings: readonly ModelSettingName[];
  diagnostics?: SagentModelDiagnostics;
};

export type ModelSettingName =
  | "context"
  | "thinking.effort"
  | "thinking.budget"
  | "thinking.output"
  | "service_tier"
  | "cache_ttl_sec"
  | "manage_context_server_side";

/**
 * User intent only. The extension host will later translate these intents to
 * one Sagent-owned operation; the webview never changes runtime state itself.
 */
export type ModelSettingsAction =
  | {
      type: "select-scope";
      scope: ModelSettingsScope;
    }
  | {
      type: "select-provider";
      scope: ModelSettingsScope;
      provider: string;
    }
  | {
      type: "select-account";
      scope: ModelSettingsScope;
      account: string;
    }
  | {
      type: "select-model";
      scope: ModelSettingsScope;
      model: string;
    }
  | {
      type: "set-setting";
      scope: ModelSettingsScope;
      name: ModelSettingName;
      value: string | number | boolean;
    }
  | { type: "refresh" };

export type ModelSettingsPanelContract = {
  snapshot: ModelSettingsSnapshot;
  onAction: (action: ModelSettingsAction) => void;
};
