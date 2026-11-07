import type { SagentEvent } from "../sagentProtocol";
import type {
  SagentTeamAgent,
  SagentToolStatus,
} from "../sagentProtocol";
import type { ChangeReviewPresentationTarget } from "../changeReviewModel";

/**
 * Structured runtime activity as emitted by the current Sagent VS Code server.
 *
 * This union is deliberately narrower than Sagent's complete RuntimeEvent
 * vocabulary. It contains only facts the current server already projects for
 * presentation. It does not become a second runtime or task manager.
 */
export type RuntimeActivityEvent =
  | {
      readonly type: "tool.started";
      readonly callId: string;
      readonly name: string;
      readonly summary: string;
      readonly arguments?: Record<string, unknown>;
      readonly turnId?: string;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "tool.completed";
      readonly callId: string;
      readonly resultCallId?: string;
      readonly status: SagentToolStatus;
      readonly text: string;
      readonly isError: boolean;
      readonly path?: string;
      readonly turnId?: string;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "retry.waiting";
      readonly attempt: number;
      readonly until: number;
      readonly message: string;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "compaction.started";
      readonly agentLabel?: string;
    }
  | {
      readonly type: "compaction.completed";
      readonly tokenBefore: number;
      readonly tokenAfter: number;
      readonly payloadEntries: number;
      readonly fallbackReason: string;
      readonly preservedTailCount: number;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "compaction.failed";
      readonly message: string;
      readonly tapeLength: number;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "agent.completed";
      readonly label: string;
      readonly elapsed?: number;
      readonly tokens?: number;
      readonly cost?: number;
    }
  | {
      readonly type: "status.changed";
      readonly status: RuntimeStatus;
      readonly text: string;
      /** A structured Sagent runtime phase; never inferred from status text. */
      readonly phase?: RuntimeActivityPhase;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "turn.started";
      readonly turnId?: string;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "turn.completed" | "turn.cancelled";
      readonly turnId?: string;
      readonly agentLabel?: string;
    }
  | {
      readonly type: "turn.failed";
      readonly turnId?: string;
      readonly message?: string;
      readonly agentLabel?: string;
    };

export type RuntimeStatus = "working" | "tool" | "ready" | "cancelled" | "notice";

/**
 * The current native runtime phase, projected for presentation.
 *
 * Sagent's runtime events are authoritative for transitions. This type only
 * gives the VS Code surface stable labels and does not model a second agent
 * lifecycle.
 */
export type RuntimeActivityPhase =
  | "idle"
  | "thinking"
  | "tools"
  | "compacting"
  | "retrying"
  | "working";

export type ActiveRuntimeTurn = {
  readonly turnId?: string;
  /** Local presentation timestamp when Sagent's turn.started arrived. */
  readonly startedAtMs: number;
};

export type RuntimeToolActivity = {
  readonly callId: string;
  readonly resultCallId?: string;
  readonly name: string;
  readonly summary: string;
  readonly status: SagentToolStatus;
  readonly text: string;
  readonly isError: boolean;
  readonly path?: string;
  readonly arguments?: Record<string, unknown>;
  readonly turnId?: string;
  readonly agentLabel?: string;
};

export type RuntimeRetryState =
  | { readonly status: "idle" }
  | {
      readonly status: "waiting";
      readonly attempt: number;
      readonly until: number;
      readonly message: string;
      readonly agentLabel?: string;
    };

export type RuntimeCompactionState =
  | { readonly status: "idle" }
  | { readonly status: "running"; readonly agentLabel?: string }
  | {
      readonly status: "completed";
      readonly tokenBefore: number;
      readonly tokenAfter: number;
      readonly payloadEntries: number;
      readonly fallbackReason: string;
      readonly preservedTailCount: number;
      readonly agentLabel?: string;
    }
  | {
      readonly status: "failed";
      readonly message: string;
      readonly tapeLength: number;
      readonly agentLabel?: string;
    };

/**
 * Diagnostics Sagent has already made available through Team and runtime
 * records. Missing values remain missing: the UI must not estimate cost or
 * tokens from text, elapsed time, or tool output.
 */
export type RuntimeAgentDiagnostics = {
  readonly label: string;
  readonly model?: string;
  readonly provider?: string;
  readonly account?: string;
  readonly tokens?: number;
  readonly costUsd?: number;
  readonly spendUsd?: number;
  readonly maxSpendUsd?: number;
  readonly elapsedSeconds?: number;
};

export type RuntimeDiagnostics = {
  readonly agents: readonly RuntimeAgentDiagnostics[];
};

export function isBackgroundToolActivity(
  activity: RuntimeToolActivity,
): boolean {
  return activity.status === "detached";
}

export type RuntimeActivityState = {
  readonly status: RuntimeStatus | undefined;
  readonly statusText: string | undefined;
  readonly phase: RuntimeActivityPhase;
  readonly activeTurn?: ActiveRuntimeTurn;
  readonly tools: readonly RuntimeToolActivity[];
  readonly retry: RuntimeRetryState;
  readonly compaction: RuntimeCompactionState;
  readonly diagnostics: RuntimeDiagnostics;
  /**
   * These are inputs for the existing ChangeReviewModel. VS Code still owns
   * diff lookup and editor actions; this module does not calculate a diff.
   */
  readonly changeReviewTargets: readonly ChangeReviewPresentationTarget[];
};

export function createInitialRuntimeActivityState(): RuntimeActivityState {
  return {
    status: undefined,
    statusText: undefined,
    phase: "idle",
    tools: [],
    retry: { status: "idle" },
    compaction: { status: "idle" },
    diagnostics: { agents: [] },
    changeReviewTargets: [],
  };
}

/**
 * Apply one structured notification to a presentation-only read model.
 *
 * Sagent's Runtime remains authoritative for cancellation, retries,
 * compaction, jobs, and accounting. This function only retains facts already
 * published by Sagent so a future view can render them without inspecting
 * human-readable text or reconstructing runtime state.
 */
export function applyRuntimeActivityEvent(
  state: RuntimeActivityState,
  event: RuntimeActivityEvent,
  nowMs = Date.now(),
): RuntimeActivityState {
  switch (event.type) {
    case "tool.started":
      return {
        ...state,
        status: "tool",
        statusText: event.summary,
        phase: "tools",
        retry: { status: "idle" },
        tools: upsertTool(state.tools, {
          callId: event.callId,
          name: event.name,
          summary: event.summary,
          status: "running",
          text: "",
          isError: false,
          ...(event.arguments ? { arguments: event.arguments } : {}),
          ...(event.turnId ? { turnId: event.turnId } : {}),
          ...(event.agentLabel ? { agentLabel: event.agentLabel } : {}),
        }),
      };
    case "tool.completed": {
      const existing = state.tools.find((tool) => tool.callId === event.callId);
      return {
        ...state,
        tools: upsertTool(state.tools, {
          callId: event.callId,
          ...(event.resultCallId ? { resultCallId: event.resultCallId } : {}),
          name: existing?.name ?? "Tool",
          summary: existing?.summary ?? "Completed tool activity",
          status: event.status,
          text: event.text,
          isError: event.isError,
          ...(existing?.arguments ? { arguments: existing.arguments } : {}),
          ...(event.path ? { path: event.path } : {}),
          ...(event.turnId ?? existing?.turnId
            ? { turnId: event.turnId ?? existing?.turnId }
            : {}),
          ...(event.agentLabel ?? existing?.agentLabel
            ? { agentLabel: event.agentLabel ?? existing?.agentLabel }
            : {}),
        }),
      };
    }
    case "retry.waiting":
      return {
        ...state,
        status: "notice",
        statusText: event.message,
        phase: "retrying",
        retry: {
          status: "waiting",
          attempt: event.attempt,
          until: event.until,
          message: event.message,
          ...(event.agentLabel ? { agentLabel: event.agentLabel } : {}),
        },
      };
    case "compaction.started":
      return {
        ...state,
        status: "notice",
        statusText: "Compacting session…",
        phase: "compacting",
        compaction: {
          status: "running",
          ...(event.agentLabel ? { agentLabel: event.agentLabel } : {}),
        },
      };
    case "compaction.completed":
      return {
        ...state,
        status: "notice",
        statusText: "Session compacted",
        phase: "working",
        compaction: {
          status: "completed",
          tokenBefore: event.tokenBefore,
          tokenAfter: event.tokenAfter,
          payloadEntries: event.payloadEntries,
          fallbackReason: event.fallbackReason,
          preservedTailCount: event.preservedTailCount,
          ...(event.agentLabel ? { agentLabel: event.agentLabel } : {}),
        },
      };
    case "compaction.failed":
      return {
        ...state,
        status: "notice",
        statusText: event.message,
        phase: "working",
        compaction: {
          status: "failed",
          message: event.message,
          tapeLength: event.tapeLength,
          ...(event.agentLabel ? { agentLabel: event.agentLabel } : {}),
        },
      };
    case "agent.completed":
      return {
        ...state,
        diagnostics: upsertDiagnostics(state.diagnostics, {
          label: event.label,
          ...(event.elapsed !== undefined ? { elapsedSeconds: event.elapsed } : {}),
          ...(event.tokens !== undefined ? { tokens: event.tokens } : {}),
          ...(event.cost !== undefined ? { costUsd: event.cost } : {}),
        }),
      };
    case "status.changed":
      return {
        ...state,
        status: event.status,
        statusText: event.text,
        phase: phaseForStatus(event.status, event.phase, state.phase),
        ...(event.status === "working" || event.status === "ready"
          ? { retry: { status: "idle" } as const }
          : {}),
      };
    case "turn.started":
      return {
        ...state,
        status: "working",
        statusText: "Working…",
        phase: "thinking",
        activeTurn: {
          ...(event.turnId ? { turnId: event.turnId } : {}),
          startedAtMs: nowMs,
        },
        retry: { status: "idle" },
      };
    case "turn.completed":
      return {
        ...state,
        status: "ready",
        statusText: "Ready",
        phase: "idle",
        activeTurn: undefined,
        retry: { status: "idle" },
      };
    case "turn.cancelled":
      return {
        ...state,
        status: "cancelled",
        statusText: "Cancelled",
        phase: "idle",
        activeTurn: undefined,
        retry: { status: "idle" },
      };
    case "turn.failed":
      return {
        ...state,
        status: "notice",
        statusText: event.message ?? "Turn failed",
        phase: "idle",
        activeTurn: undefined,
      };
  }
}

/**
 * Preserve native team metadata for a diagnostics surface. The projection is
 * intentionally a direct field mapping; it does not calculate spend or
 * discover models independently of Sagent's Team response.
 */
export function diagnosticsFromTeamAgent(
  agent: SagentTeamAgent,
): RuntimeAgentDiagnostics {
  return {
    label: agent.label,
    ...(agent.model ? { model: agent.model } : {}),
    ...(agent.provider ? { provider: agent.provider } : {}),
    ...(agent.account ? { account: agent.account } : {}),
    ...(agent.spendUsd !== undefined ? { spendUsd: agent.spendUsd } : {}),
    ...(agent.maxSpendUsd !== undefined ? { maxSpendUsd: agent.maxSpendUsd } : {}),
  };
}

export function setTeamDiagnostics(
  state: RuntimeActivityState,
  agents: readonly SagentTeamAgent[],
): RuntimeActivityState {
  return {
    ...state,
    diagnostics: {
      agents: agents.map(diagnosticsFromTeamAgent),
    },
  };
}

/**
 * Retain the existing typed change-review grouping inputs without importing
 * VS Code's host API into this pure read model. The extension host continues
 * to create these targets through its existing native change-review adapter.
 */
export function setChangeReviewTargets(
  state: RuntimeActivityState,
  targets: readonly ChangeReviewPresentationTarget[],
): RuntimeActivityState {
  return {
    ...state,
    changeReviewTargets: [...targets],
  };
}

/**
 * Validate the structured fields needed by this presentation slice. This is
 * protocol-shape validation, not parsing model/tool output or status text.
 */
export function toRuntimeActivityEvent(
  event: SagentEvent,
): RuntimeActivityEvent | undefined {
  const agentLabel = optionalString(event.agentLabel);
  switch (event.type) {
    case "tool.started":
      return requiredStrings(event, ["callId", "name", "summary"])
        ? {
            type: "tool.started",
            callId: event.callId as string,
            name: event.name as string,
            summary: event.summary as string,
            ...(recordValue(event.arguments) ? { arguments: event.arguments as Record<string, unknown> } : {}),
            ...(optionalString(event.turnId) ? { turnId: event.turnId as string } : {}),
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    case "tool.completed": {
      const status = toolStatus(event.status);
      return requiredStrings(event, ["callId", "text"]) && status
        ? {
            type: "tool.completed",
            callId: event.callId as string,
            ...(optionalString(event.resultCallId) ? { resultCallId: event.resultCallId as string } : {}),
            status,
            text: event.text as string,
            isError: event.isError === true,
            ...(optionalString(event.path) ? { path: event.path as string } : {}),
            ...(optionalString(event.turnId) ? { turnId: event.turnId as string } : {}),
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    }
    case "retry.waiting":
      return typeof event.attempt === "number"
        && typeof event.until === "number"
        && typeof event.message === "string"
        ? {
            type: "retry.waiting",
            attempt: event.attempt,
            until: event.until,
            message: event.message,
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    case "compaction.started":
      return { type: "compaction.started", ...(agentLabel ? { agentLabel } : {}) };
    case "compaction.completed":
      return typeof event.tokenBefore === "number"
        && typeof event.tokenAfter === "number"
        && typeof event.payloadEntries === "number"
        && typeof event.fallbackReason === "string"
        && typeof event.preservedTailCount === "number"
        ? {
            type: "compaction.completed",
            tokenBefore: event.tokenBefore,
            tokenAfter: event.tokenAfter,
            payloadEntries: event.payloadEntries,
            fallbackReason: event.fallbackReason,
            preservedTailCount: event.preservedTailCount,
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    case "compaction.failed":
      return requiredStrings(event, ["message"])
        && typeof event.tapeLength === "number"
        ? {
            type: "compaction.failed",
            message: event.message as string,
            tapeLength: event.tapeLength,
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    case "agent.completed":
      return requiredStrings(event, ["label"])
        ? {
            type: "agent.completed",
            label: event.label as string,
            ...(numberValue(event.elapsed) !== undefined ? { elapsed: event.elapsed as number } : {}),
            ...(numberValue(event.tokens) !== undefined ? { tokens: event.tokens as number } : {}),
            ...(numberValue(event.cost) !== undefined ? { cost: event.cost as number } : {}),
          }
        : undefined;
    case "status.changed": {
      const status = runtimeStatus(event.status);
      const phase = runtimePhase(event.phase);
      return status && typeof event.text === "string"
        ? {
            type: "status.changed",
            status,
            text: event.text,
            ...(phase ? { phase } : {}),
            ...(agentLabel ? { agentLabel } : {}),
          }
        : undefined;
    }
    case "turn.started":
      return {
        type: "turn.started",
        ...(optionalString(event.turnId) ? { turnId: event.turnId as string } : {}),
        ...(agentLabel ? { agentLabel } : {}),
      };
    case "turn.completed":
    case "turn.cancelled":
      return {
        type: event.type,
        ...(optionalString(event.turnId) ? { turnId: event.turnId as string } : {}),
        ...(agentLabel ? { agentLabel } : {}),
      };
    case "turn.failed":
      return {
        type: event.type,
        ...(optionalString(event.turnId) ? { turnId: event.turnId as string } : {}),
        ...(optionalString(event.message) ? { message: event.message as string } : {}),
        ...(agentLabel ? { agentLabel } : {}),
      };
    default:
      return undefined;
  }
}

function upsertTool(
  tools: readonly RuntimeToolActivity[],
  update: RuntimeToolActivity,
): RuntimeToolActivity[] {
  const index = tools.findIndex((tool) => tool.callId === update.callId);
  if (index < 0) {
    return [...tools, update];
  }
  return tools.map((tool, toolIndex) => (toolIndex === index ? update : tool));
}

function upsertDiagnostics(
  diagnostics: RuntimeDiagnostics,
  update: RuntimeAgentDiagnostics,
): RuntimeDiagnostics {
  const index = diagnostics.agents.findIndex((agent) => agent.label === update.label);
  if (index < 0) {
    return { agents: [...diagnostics.agents, update] };
  }
  return {
    agents: diagnostics.agents.map((agent, agentIndex) =>
      agentIndex === index ? { ...agent, ...update } : agent,
    ),
  };
}

function requiredStrings(
  event: SagentEvent,
  names: readonly string[],
): boolean {
  return names.every(
    (name) => typeof event[name] === "string" && (event[name] as string).length > 0,
  );
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function recordValue(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toolStatus(value: unknown): SagentToolStatus | undefined {
  return value === "running"
    || value === "completed"
    || value === "failed"
    || value === "detached"
    || value === "cancelled"
    ? value
    : undefined;
}

function runtimeStatus(value: unknown): RuntimeStatus | undefined {
  return value === "working"
    || value === "tool"
    || value === "ready"
    || value === "cancelled"
    || value === "notice"
    ? value
    : undefined;
}

function runtimePhase(value: unknown): RuntimeActivityPhase | undefined {
  return value === "idle"
    || value === "thinking"
    || value === "tools"
    || value === "compacting"
    || value === "retrying"
    || value === "working"
    ? value
    : undefined;
}

function phaseForStatus(
  status: RuntimeStatus,
  explicitPhase: RuntimeActivityPhase | undefined,
  currentPhase: RuntimeActivityPhase,
): RuntimeActivityPhase {
  if (explicitPhase !== undefined) {
    return explicitPhase;
  }
  switch (status) {
    case "ready":
    case "cancelled":
      return "idle";
    case "tool":
      return "tools";
    case "working":
      return "working";
    default:
      return currentPhase;
  }
}

/** Match Sagent TUI's elapsed display: seconds are always visible. */
export function formatRuntimeElapsed(seconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const remainingSeconds = totalSeconds % 60;
  const parts = [
    days > 0 ? `${days}d` : "",
    days > 0 || hours > 0 ? `${hours}h` : "",
    days > 0 || hours > 0 || minutes > 0 ? `${minutes}m` : "",
    `${remainingSeconds}s`,
  ];
  return parts.filter(Boolean).join(" ");
}
