/**
 * Wire types for the Sagent-specific VS Code process protocol.
 *
 * The Python server remains authoritative for session records and runtime
 * state. These types describe only the stable data crossing the process
 * boundary; they do not recreate Sagent's session or team implementations.
 */

import type {
  ModelSettingName,
  ModelSettingsScope,
  ModelSettingsSnapshot,
  SagentContextSize,
  SagentModelCapability,
  SagentModelDiagnostics,
  SagentModelLimits,
  SagentModelSettingsValues,
  SagentProviderState,
  SagentServiceTier,
  SagentThinkingBudget,
  SagentThinkingEffort,
  SagentThinkingOutput,
} from "./modelSettings/modelSettingsContract";
import type { SagentCommandDescriptor } from "./commandSurface/commandSurface";

export type { SagentCommandDescriptor } from "./commandSurface/commandSurface";

export type SagentEvent = {
  type: string;
  [key: string]: unknown;
};

export type SagentSessionItemType =
  | "user"
  | "agent"
  | "assistant"
  | "thinking"
  | "tool";

export type SagentToolStatus =
  | "running"
  | "completed"
  | "failed"
  | "detached"
  | "cancelled";

export type SagentFileChange =
  | {
      kind: "edit";
      edits: SagentFileEdit[];
    }
  | {
      kind: "write";
      content?: string;
    };

export type SagentFileEdit = {
  before?: string;
  after?: string;
  start?: number;
  count?: number;
};

export type SagentSessionItem = {
  id: string;
  type: SagentSessionItemType;
  turnId: string;
  model?: string;
  text?: string;
  callId?: string;
  name?: string;
  summary?: string;
  arguments?: Record<string, unknown>;
  status?: SagentToolStatus;
  isError?: boolean;
  path?: string;
  sender?: string;
  recipient?: string;
  contextId?: number;
  timestamp?: string;
  fileChange?: SagentFileChange;
};

export type SagentSessionSnapshot = {
  version: number;
  items: SagentSessionItem[];
};

export type SagentSessionState = {
  open: boolean;
  path?: string;
  sessionId?: string;
  tools: string[];
  running: boolean;
  snapshot: SagentSessionSnapshot;
};

export type SagentSessionInfo = {
  id: string;
  path: string;
  mtime: number;
  title: string;
  records: number;
  model: string;
};

export type SagentSessionList = {
  activePath?: string;
  sessions: SagentSessionInfo[];
};

export type SagentTeamAgent = {
  label: string;
  parent?: string;
  above?: string;
  children: string[];
  watching?: string;
  persistent: boolean;
  running: boolean;
  conversationId?: string;
  sessionPath?: string;
  model?: string;
  provider?: string;
  account?: string;
  tools: string[];
  allowedTools?: string[];
  spendUsd?: number;
  maxSpendUsd?: number;
  owedReplies: Record<string, string>;
};

export type SagentTeamState = {
  open: boolean;
  agents: SagentTeamAgent[];
};

export type SagentAgentSessionState = {
  label: string;
  sessionId?: string;
  path?: string;
  snapshot: SagentSessionSnapshot;
};

export type SagentCommandList = {
  agentLabel: string;
  commands: SagentCommandDescriptor[];
};

export type SagentSkillInfo = {
  name: string;
  description: string;
  source: "project" | "user" | "builtin";
};

export type SagentRekursivSkillCandidate = {
  id: string;
  label: string;
  description: string;
  path: string;
};

export type SagentRekursivContext = {
  mode: "standard" | "rekursiv";
  orientation: boolean;
  skillDirectories: string[];
  skills: SagentSkillInfo[];
  environment: Record<string, {
    status: string;
    url?: string;
    checks?: Array<{ name: string; status: string; detail: string }>;
  }>;
};

export function parseSagentSessionState(
  response: Record<string, unknown>,
): SagentSessionState {
  return {
    open: response.open === true,
    path: optionalString(response.path),
    sessionId: optionalString(response.sessionId),
    tools: stringArray(response.tools),
    running: response.running === true,
    snapshot: parseSessionSnapshot(response.snapshot),
  };
}

export function parseSagentSessionList(
  response: Record<string, unknown>,
): SagentSessionList {
  const sessions = Array.isArray(response.sessions)
    ? response.sessions.flatMap((value) => {
        const parsed = parseSagentSessionInfo(value);
        return parsed ? [parsed] : [];
      })
    : [];
  return {
    activePath: optionalString(response.activePath),
    sessions,
  };
}

export function parseSagentTeamState(
  response: Record<string, unknown>,
): SagentTeamState {
  const agents = Array.isArray(response.agents)
    ? response.agents.flatMap((value) => {
        const parsed = parseSagentTeamAgent(value);
        return parsed ? [parsed] : [];
      })
    : [];
  return { open: response.open === true, agents };
}

export function parseSagentAgentSessionState(
  response: Record<string, unknown>,
): SagentAgentSessionState {
  return {
    label: requiredString(response.label) ?? "",
    sessionId: optionalString(response.sessionId),
    path: optionalString(response.path),
    snapshot: parseSessionSnapshot(response.snapshot),
  };
}

export function parseSagentCommandList(
  response: Record<string, unknown>,
): SagentCommandList {
  const commands = Array.isArray(response.commands)
    ? response.commands.flatMap((value) => {
        const parsed = parseSagentCommandDescriptor(value);
        return parsed ? [parsed] : [];
      })
    : [];
  return {
    agentLabel: requiredString(response.agentLabel, "agentLabel"),
    commands,
  };
}

export function parseSagentSkillList(
  response: Record<string, unknown>,
): SagentSkillInfo[] {
  if (!Array.isArray(response.skills)) return [];
  return response.skills.flatMap((value) => {
    if (!isRecord(value) || typeof value.name !== "string") return [];
    return [{
      name: value.name,
      description: typeof value.description === "string" ? value.description : "",
      source: value.source === "project" || value.source === "builtin" ? value.source : "user",
    } satisfies SagentSkillInfo];
  });
}

export function parseSagentRekursivContext(
  response: Record<string, unknown>,
): SagentRekursivContext {
  const mode = response.mode === "rekursiv" ? "rekursiv" : "standard";
  const skills = Array.isArray(response.skills)
    ? response.skills.flatMap((value) => {
        if (!isRecord(value) || typeof value.name !== "string") return [];
        return [{
          name: value.name,
          description: typeof value.description === "string" ? value.description : "",
          source: value.source === "project" || value.source === "builtin" ? value.source : "user",
        } satisfies SagentSkillInfo];
      })
    : [];
  const environment: SagentRekursivContext["environment"] = {};
  if (isRecord(response.environment)) {
    for (const [name, value] of Object.entries(response.environment)) {
      if (isRecord(value) && typeof value.status === "string") {
        const checks = Array.isArray(value.checks)
          ? value.checks.flatMap((check) => {
              if (!isRecord(check)
                || typeof check.name !== "string"
                || typeof check.status !== "string"
                || typeof check.detail !== "string") {
                return [];
              }
              return [{ name: check.name, status: check.status, detail: check.detail }];
            })
          : undefined;
        environment[name] = {
          status: value.status,
          ...(typeof value.url === "string" ? { url: value.url } : {}),
          ...(checks ? { checks } : {}),
        };
      }
    }
  }
  return {
    mode,
    orientation: response.orientation === true,
    skillDirectories: stringArray(response.skillDirectories),
    skills,
    environment,
  };
}

function parseSagentCommandDescriptor(
  value: unknown,
): SagentCommandDescriptor | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const name = requiredString(value.name);
  const kind = value.kind === "action" || value.kind === "setting" ? value.kind : undefined;
  const scopes = ["session", "agent", "team", "provider", "runtime", "tool", "screen", "mcp"] as const;
  const scope = scopes.includes(value.scope as (typeof scopes)[number])
    ? (value.scope as (typeof scopes)[number])
    : undefined;
  const helpShort = requiredString(value.helpShort);
  if (!name || !kind || !scope || !helpShort) {
    return undefined;
  }
  return {
    name,
    kind,
    scope,
    helpShort,
    ...(typeof value.helpLong === "string" ? { helpLong: value.helpLong } : {}),
    ...(typeof value.usage === "string" ? { usage: value.usage } : {}),
    options: stringArray(value.options),
    readable: value.readable === true,
    operation: kind === "action" ? "run-action" : "apply-setting",
  };
}

export function parseModelSettingsSnapshot(
  response: Record<string, unknown>,
): ModelSettingsSnapshot {
  const provider = record(response.provider, "provider");
  const model = record(response.model, "model");
  const capability = record(response.capability, "capability");
  const settings = record(response.settings, "settings");
  const status = response.status;
  if (status !== "loading" && status !== "ready" && status !== "error") {
    throw new Error("invalid model settings status");
  }
  const scope = response.selectedScope;
  if (scope !== "current-agent" && scope !== "new-session-default") {
    throw new Error("invalid model settings scope");
  }
  return {
    status,
    ...(typeof response.errorMessage === "string" ? { errorMessage: response.errorMessage } : {}),
    agentLabel: requiredString(response.agentLabel, "agentLabel"),
    availableScopes: stringArray(response.availableScopes).filter(isModelSettingsScope),
    selectedScope: scope,
    provider: parseProviderState(provider),
    model: parseModelState(model),
    capability: parseModelCapability(capability),
    settings: parseModelSettingsValues(settings),
    writableSettings: stringArray(response.writableSettings).filter(isModelSettingName),
    ...(response.diagnostics === undefined
      ? {}
      : { diagnostics: parseModelDiagnostics(record(response.diagnostics, "diagnostics")) }),
  };
}

function parseSagentTeamAgent(value: unknown): SagentTeamAgent | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const label = requiredString(value.label);
  if (!label) {
    return undefined;
  }
  const agent: SagentTeamAgent = {
    label,
    children: stringArray(value.children),
    persistent: value.persistent === true,
    running: value.running === true,
    tools: stringArray(value.tools),
    owedReplies: recordOfStrings(value.owedReplies),
  };
  assignOptionalString(agent, "parent", value.parent);
  assignOptionalString(agent, "above", value.above);
  assignOptionalString(agent, "watching", value.watching);
  assignOptionalString(agent, "conversationId", value.conversationId);
  assignOptionalString(agent, "sessionPath", value.sessionPath);
  assignOptionalString(agent, "model", value.model);
  assignOptionalString(agent, "provider", value.provider);
  assignOptionalString(agent, "account", value.account);
  if (Array.isArray(value.allowedTools)) {
    agent.allowedTools = stringArray(value.allowedTools);
  }
  assignOptionalNumber(agent, "spendUsd", value.spendUsd);
  assignOptionalNumber(agent, "maxSpendUsd", value.maxSpendUsd);
  return agent;
}

function parseProviderState(value: Record<string, unknown>): SagentProviderState {
  return {
    providers: stringArray(value.providers),
    currentProvider: requiredString(value.currentProvider, "currentProvider"),
    accounts: stringArray(value.accounts),
    currentAccount: requiredString(value.currentAccount, "currentAccount"),
    ...(Array.isArray(value.acpServers) ? { acpServers: stringArray(value.acpServers) } : {}),
    ...(typeof value.currentAcpServer === "string" ? { currentAcpServer: value.currentAcpServer } : {}),
  };
}

function parseModelState(value: Record<string, unknown>): ModelSettingsSnapshot["model"] {
  return {
    models: stringArray(value.models),
    currentModel: requiredString(value.currentModel, "currentModel"),
  };
}

function parseModelCapability(value: Record<string, unknown>): SagentModelCapability {
  const defaults = record(value.defaults, "capability.defaults");
  return {
    modelId: requiredString(value.modelId, "capability.modelId"),
    contextSizes: array(value.contextSizes, "capability.contextSizes")
      .map((entry) => parseContextSize(record(entry, "context size"))),
    thinkingEfforts: stringArray(value.thinkingEfforts).filter(isThinkingEffort),
    thinkingBudgets: stringArray(value.thinkingBudgets).filter(isThinkingBudget),
    thinkingOutputs: stringArray(value.thinkingOutputs).filter(isThinkingOutput),
    serviceTiers: stringArray(value.serviceTiers).filter(isServiceTier),
    cacheTtlSeconds: numberArray(value.cacheTtlSeconds),
    manageContextServerSide: value.manageContextServerSide instanceof Array
      ? value.manageContextServerSide.filter((entry): entry is boolean => typeof entry === "boolean")
      : [],
    defaults: parseModelSettingsValues(defaults),
    limits: parseModelLimits(record(value.limits, "capability.limits")),
  };
}

function parseModelSettingsValues(value: Record<string, unknown>): SagentModelSettingsValues {
  return {
    context: parseContextSize(record(value.context, "settings.context")),
    thinkingEffort: enumValue(value.thinkingEffort, isThinkingEffort, "thinkingEffort"),
    thinkingBudget: enumValue(value.thinkingBudget, isThinkingBudget, "thinkingBudget"),
    thinkingOutput: enumValue(value.thinkingOutput, isThinkingOutput, "thinkingOutput"),
    serviceTier: enumValue(value.serviceTier, isServiceTier, "serviceTier"),
    cacheTtlSeconds: numberValue(value.cacheTtlSeconds, "cacheTtlSeconds"),
    manageContextServerSide: booleanValue(value.manageContextServerSide, "manageContextServerSide"),
  };
}

function parseContextSize(value: Record<string, unknown>): SagentContextSize {
  return {
    name: requiredString(value.name, "context.name"),
    tokens: numberValue(value.tokens, "context.tokens"),
    displayName: requiredString(value.displayName, "context.displayName"),
    isDefault: value.isDefault === true,
  };
}

function parseModelLimits(value: Record<string, unknown>): SagentModelLimits {
  return {
    maxRequestTokens: numberValue(value.maxRequestTokens, "maxRequestTokens"),
    maxResponseTokens: numberValue(value.maxResponseTokens, "maxResponseTokens"),
    maxContextTokens: numberValue(value.maxContextTokens, "maxContextTokens"),
  };
}

function parseModelDiagnostics(value: Record<string, unknown>): SagentModelDiagnostics {
  const thinking = record(value.thinking, "diagnostics.thinking");
  return {
    label: requiredString(value.label, "diagnostics.label"),
    model: requiredString(value.model, "diagnostics.model"),
    context: requiredString(value.context, "diagnostics.context"),
    thinking: {
      budget: enumValue(thinking.budget, isThinkingBudget, "diagnostics.thinking.budget"),
      output: enumValue(thinking.output, isThinkingOutput, "diagnostics.thinking.output"),
      effort: enumValue(thinking.effort, isThinkingEffort, "diagnostics.thinking.effort"),
    },
    serviceTier: enumValue(value.serviceTier, isServiceTier, "diagnostics.serviceTier"),
    cacheTtlSeconds: numberValue(value.cacheTtlSeconds, "diagnostics.cacheTtlSeconds"),
    limits: parseModelLimits(record(value.limits, "diagnostics.limits")),
    cwd: requiredString(value.cwd, "diagnostics.cwd"),
    tools: stringArray(value.tools),
  };
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`invalid ${field}`);
  }
  return value;
}

function array(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`invalid ${field}`);
  }
  return value;
}

function numberArray(value: unknown): number[] {
  return array(value, "number array").map((entry, index) =>
    numberValue(entry, `number array[${index}]`),
  );
}

function numberValue(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`invalid ${field}`);
  }
  return value;
}

function booleanValue(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`invalid ${field}`);
  }
  return value;
}

function enumValue<T>(
  value: unknown,
  predicate: (candidate: unknown) => candidate is T,
  field: string,
): T {
  if (!predicate(value)) {
    throw new Error(`invalid ${field}`);
  }
  return value;
}

function isModelSettingsScope(value: string): value is ModelSettingsScope {
  return value === "current-agent" || value === "new-session-default";
}

function isModelSettingName(value: string): value is ModelSettingName {
  return value === "context"
    || value === "thinking.effort"
    || value === "thinking.budget"
    || value === "thinking.output"
    || value === "service_tier"
    || value === "cache_ttl_sec"
    || value === "manage_context_server_side";
}

function isThinkingEffort(value: unknown): value is SagentThinkingEffort {
  return value === "none"
    || value === "min"
    || value === "low"
    || value === "medium"
    || value === "high"
    || value === "xhigh"
    || value === "max";
}

function isThinkingBudget(value: unknown): value is SagentThinkingBudget {
  return value === "none" || value === "auto" || value === "fixed";
}

function isThinkingOutput(value: unknown): value is SagentThinkingOutput {
  return value === "none" || value === "text" || value === "redacted";
}

function isServiceTier(value: unknown): value is SagentServiceTier {
  return value === "auto"
    || value === "default"
    || value === "flex"
    || value === "priority"
    || value === "ultrafast";
}

function parseSagentSessionInfo(value: unknown): SagentSessionInfo | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const id = requiredString(value.id);
  const path = requiredString(value.path);
  if (!id || !path || typeof value.mtime !== "number") {
    return undefined;
  }
  return {
    id,
    path,
    mtime: value.mtime,
    title: optionalString(value.title) ?? "Untitled session",
    records: typeof value.records === "number" ? value.records : 0,
    model: optionalString(value.model) ?? "",
  };
}

function parseSessionSnapshot(value: unknown): SagentSessionSnapshot {
  if (!isRecord(value)) {
    return { version: 1, items: [] };
  }

  const items = Array.isArray(value.items)
    ? value.items.flatMap((item) => {
        const parsed = parseSessionItem(item);
        return parsed ? [parsed] : [];
      })
    : [];

  return {
    version: typeof value.version === "number" ? value.version : 1,
    items,
  };
}

function parseSessionItem(value: unknown): SagentSessionItem | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const id = requiredString(value.id);
  const turnId = requiredString(value.turnId);
  const type = sessionItemType(value.type);
  if (!id || !turnId || !type) {
    return undefined;
  }

  const item: SagentSessionItem = {
    id,
    type,
    turnId,
  };
  addStringProperty(item, value, "model");
  addStringProperty(item, value, "text");
  addStringProperty(item, value, "callId");
  addStringProperty(item, value, "name");
  addStringProperty(item, value, "summary");
  addStringProperty(item, value, "path");
  addStringProperty(item, value, "sender");
  addStringProperty(item, value, "recipient");
  addStringProperty(item, value, "timestamp");
  if (typeof value.contextId === "number") {
    item.contextId = value.contextId;
  }
  if (isRecord(value.arguments)) {
    item.arguments = value.arguments;
  }
  const status = toolStatus(value.status);
  if (status) {
    item.status = status;
  }
  if (typeof value.isError === "boolean") {
    item.isError = value.isError;
  }
  const fileChange = parseSagentFileChange(value.fileChange);
  if (fileChange) {
    item.fileChange = fileChange;
  }
  return item;
}

export function parseSagentFileChange(value: unknown): SagentFileChange | undefined {
  if (!isRecord(value) || (value.kind !== "edit" && value.kind !== "write")) {
    return undefined;
  }
  if (value.kind === "write") {
    return {
      kind: "write",
      ...(typeof value.content === "string" ? { content: value.content } : {}),
    };
  }
  if (!Array.isArray(value.edits)) {
    return undefined;
  }
  const edits = value.edits.flatMap((edit) => {
    if (!isRecord(edit)) {
      return [];
    }
    return [{
      ...(typeof edit.before === "string" ? { before: edit.before } : {}),
      ...(typeof edit.after === "string" ? { after: edit.after } : {}),
      ...(typeof edit.start === "number" ? { start: edit.start } : {}),
      ...(typeof edit.count === "number" ? { count: edit.count } : {}),
    }];
  });
  return { kind: "edit", edits };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown): string | undefined;
function requiredString(value: unknown, field: string): string;
function requiredString(value: unknown, field?: string): string | undefined {
  if (field !== undefined) {
    if (typeof value !== "string" || value.length === 0) {
      throw new Error(`invalid ${field}`);
    }
    return value;
  }
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function recordOfStrings(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    ),
  );
}

function assignOptionalString(
  agent: SagentTeamAgent,
  property: "parent" | "above" | "watching" | "conversationId" | "sessionPath" | "model" | "provider" | "account",
  value: unknown,
): void {
  const parsed = optionalString(value);
  if (parsed !== undefined) {
    agent[property] = parsed;
  }
}

function assignOptionalNumber(
  agent: SagentTeamAgent,
  property: "spendUsd" | "maxSpendUsd",
  value: unknown,
): void {
  if (typeof value === "number") {
    agent[property] = value;
  }
}

function sessionItemType(value: unknown): SagentSessionItemType | undefined {
  switch (value) {
    case "user":
    case "agent":
    case "assistant":
    case "thinking":
    case "tool":
      return value;
    default:
      return undefined;
  }
}

function toolStatus(value: unknown): SagentToolStatus | undefined {
  switch (value) {
    case "running":
    case "completed":
    case "failed":
    case "detached":
    case "cancelled":
      return value;
    default:
      return undefined;
  }
}

function addStringProperty(
  item: SagentSessionItem,
  object: Record<string, unknown>,
  property:
    | "model"
    | "text"
    | "callId"
    | "name"
    | "summary"
    | "path"
    | "sender"
    | "recipient"
    | "timestamp",
): void {
  const value = object[property];
  if (typeof value === "string") {
    item[property] = value;
  }
}
