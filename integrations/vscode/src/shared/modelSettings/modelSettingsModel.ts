import type {
  ModelSettingName,
  ModelSettingsSnapshot,
  SagentContextSize,
  SagentModelDiagnostics,
} from "./modelSettingsContract";

export type ModelSettingsSelectOption = {
  value: string;
  label: string;
  isDefault?: boolean;
};

export type ModelSettingsControl = {
  name: ModelSettingName;
  label: string;
  description: string;
  value: string;
  options: readonly ModelSettingsSelectOption[];
};

export type ModelSettingsViewModel = {
  status: ModelSettingsSnapshot["status"];
  errorMessage?: string;
  agentLabel: string;
  availableScopes: ModelSettingsSnapshot["availableScopes"];
  selectedScope: ModelSettingsSnapshot["selectedScope"];
  providerOptions: readonly ModelSettingsSelectOption[];
  acpServerOptions: readonly ModelSettingsSelectOption[];
  accountOptions: readonly ModelSettingsSelectOption[];
  modelOptions: readonly ModelSettingsSelectOption[];
  controls: readonly ModelSettingsControl[];
  diagnostics?: SagentModelDiagnostics;
};

/**
 * Project Sagent's native capability/options snapshot into presentation data.
 * The arrays are copied from Sagent; this function deliberately has no
 * provider/model inventory of its own.
 */
export function createModelSettingsViewModel(
  snapshot: ModelSettingsSnapshot,
): ModelSettingsViewModel {
  const { capability, provider, model, settings } = snapshot;
  return {
    status: snapshot.status,
    errorMessage: snapshot.errorMessage,
    agentLabel: snapshot.agentLabel,
    availableScopes: snapshot.availableScopes,
    selectedScope: snapshot.selectedScope,
    providerOptions: options(provider.providers),
    acpServerOptions: options(
      provider.acpServers && provider.acpServers.length > 0
        ? provider.acpServers
        : (provider.currentProvider === "acp" ? ["agy"] : []),
      provider.currentAcpServer ?? "agy",
      (server) => (server === "agy" ? "Google Antigravity (agy)" : String(server)),
    ),
    accountOptions: options(provider.accounts),
    modelOptions: options(model.models),
    controls: [
      selectControl(
        "context",
        "Context",
        "Context window before Sagent compacts the conversation.",
        settings.context.name,
        capability.contextSizes.map(contextOption),
      ),
      selectControl(
        "thinking.effort",
        "Thinking effort",
        "How hard the model reasons.",
        settings.thinkingEffort,
        options(capability.thinkingEfforts, capability.defaults.thinkingEffort),
      ),
      selectControl(
        "thinking.budget",
        "Thinking budget",
        "Whether reasoning is disabled, automatic, or fixed.",
        settings.thinkingBudget,
        options(capability.thinkingBudgets, capability.defaults.thinkingBudget),
      ),
      selectControl(
        "thinking.output",
        "Thinking output",
        "What part of the model reasoning Sagent returns.",
        settings.thinkingOutput,
        options(capability.thinkingOutputs, capability.defaults.thinkingOutput),
      ),
      selectControl(
        "service_tier",
        "Service tier",
        "Serving tier, balancing speed and price.",
        settings.serviceTier,
        options(capability.serviceTiers, capability.defaults.serviceTier),
      ),
      selectControl(
        "cache_ttl_sec",
        "Prompt cache lifetime",
        "How long the provider keeps the prompt cache.",
        String(settings.cacheTtlSeconds),
        options(capability.cacheTtlSeconds, capability.defaults.cacheTtlSeconds, formatSeconds),
      ),
      selectControl(
        "manage_context_server_side",
        "Server-managed context",
        "Whether the provider rolls the conversation history.",
        String(settings.manageContextServerSide),
        options(
          capability.manageContextServerSide,
          capability.defaults.manageContextServerSide,
          (enabled) => (enabled ? "Enabled" : "Disabled"),
        ),
      ),
    ].filter((control) => snapshot.writableSettings.includes(control.name)),
    diagnostics: snapshot.diagnostics,
  };
}

function options<T extends string | number | boolean>(
  values: readonly T[],
  defaultValue?: T,
  formatValue: (value: T) => string = String,
): ModelSettingsSelectOption[] {
  const normalizedDefault = defaultValue === undefined ? undefined : String(defaultValue);
  return values.map((value) => ({
    value: String(value),
    label:
      String(value) === normalizedDefault
        ? `${formatValue(value)} (default)`
        : formatValue(value),
    ...(String(value) === normalizedDefault ? { isDefault: true } : {}),
  }));
}

function contextOption(context: SagentContextSize): ModelSettingsSelectOption {
  return {
    // Sagent's setter accepts the context name; displayName is presentation only.
    value: context.name,
    label: context.isDefault
      ? `${context.displayName} (default)`
      : context.displayName,
  };
}

function selectControl(
  name: ModelSettingName,
  label: string,
  description: string,
  value: string,
  controlOptions: readonly ModelSettingsSelectOption[],
): ModelSettingsControl {
  return { name, label, description, value, options: controlOptions };
}

function formatSeconds(seconds: number): string {
  if (seconds === 0) {
    return "Disabled";
  }
  return `${seconds} seconds`;
}
