import {
  createInitialConversationState,
  type AgentActivity,
  type AgentActivityRecord,
  type AgentMessage,
  type CommandOutputRecord,
  type ConversationModelIdentity,
  type ConversationState,
  type ConversationTurnItem,
  type ConversationTimelineItem,
  type ConversationTurn,
  type ToolActivity,
  type TurnState,
} from "../shared/conversationTypes";
import type {
  SagentSessionItem,
  SagentFileChange,
  SagentSessionList,
  SagentSessionSnapshot,
  SagentTeamState,
} from "../shared/sagentProtocol";
import type { SagentCommandDescriptor } from "../shared/sagentProtocol";
import type { ModelSettingsSnapshot } from "../shared/modelSettings/modelSettingsContract";
import type { RuntimeActivityState } from "../shared/runtimeActivity/runtimeActivityModel";
import {
  sessionNavigationCreationState,
  sessionNavigationErrorState,
  sessionNavigationLoadingState,
  sessionNavigationSelectionState,
  sessionNavigationStateFromList,
} from "../shared/sessionNavigation";
import {
  createChangeReviewModel,
  type ChangeReviewPresentationTarget,
} from "../shared/changeReviewModel";

export type {
  ConnectionState,
  ConversationState,
  ConversationTurn,
  AgentActivity,
  AgentMessage,
  ConversationTurnItem,
  ToolActivity,
  ToolActivityState,
  TurnState,
} from "../shared/conversationTypes";

export type ConversationAction =
  | { type: "connection.started" }
  | {
      type: "session.opened";
      path?: string;
      snapshot?: SagentSessionSnapshot;
      agentLabel?: string;
    }
  | { type: "sessions.loading" }
  | { type: "sessions.updated"; listing: SagentSessionList }
  | { type: "sessions.failed"; message: string }
  | { type: "session.selection.started"; sessionPath: string }
  | { type: "session.creation.started" }
  | { type: "agent.session.opened"; label: string; path?: string; snapshot: SagentSessionSnapshot }
  | { type: "team.updated"; team: SagentTeamState }
  | { type: "model.settings.updated"; snapshot: ModelSettingsSnapshot }
  | { type: "commands.updated"; descriptors: SagentCommandDescriptor[] }
  | { type: "command.output"; output: CommandOutputRecord }
  | { type: "command.output.removed"; id: string }
  | { type: "runtime.activity.updated"; state: RuntimeActivityState }
  | { type: "changes.updated"; targets: readonly ChangeReviewPresentationTarget[] }
  | { type: "user.submitted"; text: string }
  | {
      type: "turn.started";
      text: string;
      turnId?: string;
      agentLabel?: string;
      modelIdentity?: ConversationModelIdentity;
    }
  | { type: "turn.cancelling" }
  | { type: "assistant.delta"; text: string; agentLabel?: string }
  | { type: "assistant.completed"; text: string; record?: string; agentLabel?: string }
  | { type: "assistant.thinking"; text: string; agentLabel?: string }
  | {
      type: "message.created";
      id: string;
      kind: "user" | "agent";
      text: string;
      turnId?: string;
      sender?: string;
      recipient?: string;
      agentLabel?: string;
    }
  | {
      type: "tool.started";
      callId: string;
      name: string;
      summary: string;
      arguments?: Record<string, unknown>;
      agentLabel?: string;
    }
  | {
      type: "tool.completed";
      callId: string;
      text: string;
      isError: boolean;
      state: ToolActivity["state"];
      path?: string;
      fileChange?: SagentFileChange;
      agentLabel?: string;
    }
  | { type: "status.changed"; text: string; agentLabel?: string }
  | {
      type: "agent.event";
      label: string;
      event: string;
      text?: string;
      records?: AgentActivityRecord[];
    }
  | {
      type: "agent.completed";
      label: string;
      elapsed?: number;
      tokens?: number;
      cost?: number;
    }
  | { type: "turn.completed"; text: string; agentLabel?: string }
  | { type: "turn.cancelled"; agentLabel?: string }
  | { type: "turn.failed"; message: string; agentLabel?: string }
  | { type: "connection.error"; message: string }
  | { type: "connection.disconnected" }
  | { type: "request.failed"; message: string };

let nextTurnNumber = 1;

export function reduceConversation(
  state: ConversationState,
  action: ConversationAction,
): ConversationState {
  switch (action.type) {
    case "connection.started":
      return { ...state, connection: "starting", statusText: "Starting Sagent…", lastError: undefined };
    case "session.opened": {
      const turns = action.snapshot ? turnsFromSnapshot(action.snapshot) : [];
      // The snapshot comes from Sagent's journal. Hydrating it here keeps the
      // webview presentation replaceable without creating a second session
      // store in the extension or browser runtime.
      return {
        ...state,
        connection: "ready",
        statusText: "Ready",
        sessionPath: action.path,
        sessionNavigation: action.path
          ? {
              ...state.sessionNavigation,
              activePath: action.path,
              pendingAction: undefined,
            }
          : state.sessionNavigation,
        lastError: undefined,
        agents: [],
        commandDescriptors: [],
        commandOutputs: [],
        selectedAgentLabel: action.agentLabel ?? state.selectedAgentLabel,
        turns,
        timeline: timelineFromTurns(turns),
        activeTurnId: undefined,
      };
    }
    case "sessions.loading":
      return {
        ...state,
        sessionNavigation: sessionNavigationLoadingState(state.sessionNavigation),
      };
    case "sessions.updated":
      return {
        ...state,
        sessionNavigation: sessionNavigationStateFromList(action.listing),
      };
    case "sessions.failed":
      return {
        ...state,
        sessionNavigation: sessionNavigationErrorState(
          state.sessionNavigation,
          action.message,
        ),
      };
    case "session.selection.started":
      return {
        ...state,
        sessionNavigation: sessionNavigationSelectionState(
          state.sessionNavigation,
          action.sessionPath,
        ),
      };
    case "session.creation.started":
      return {
        ...state,
        sessionNavigation: sessionNavigationCreationState(state.sessionNavigation),
      };
    case "agent.session.opened": {
      const turns = turnsFromSnapshot(action.snapshot);
      return {
        ...state,
        connection: "ready",
        statusText: "Ready",
        sessionPath: action.path,
        sessionNavigation: action.path
          ? {
              ...state.sessionNavigation,
              activePath: action.path,
              pendingAction: undefined,
            }
          : state.sessionNavigation,
        lastError: undefined,
        selectedAgentLabel: action.label,
        commandOutputs: [],
        turns,
        timeline: timelineFromTurns(turns),
        activeTurnId: undefined,
      };
    }
    case "team.updated":
      return {
        ...state,
        teamOpen: action.team.open,
        teamAgents: action.team.agents,
        selectedAgentLabel: selectedAgentForTeam(state.selectedAgentLabel, action.team.agents),
      };
    case "model.settings.updated":
      return { ...state, modelSettings: action.snapshot };
    case "commands.updated":
      return { ...state, commandDescriptors: action.descriptors };
    case "command.output": {
      const commandOutputs = [...state.commandOutputs.slice(-49), action.output];
      return {
        ...state,
        commandOutputs,
        timeline: pruneTimeline(
          [...state.timeline, { type: "command", id: action.output.id }],
          commandOutputs,
        ),
      };
    }
    case "command.output.removed":
      return {
        ...state,
        commandOutputs: state.commandOutputs.filter((output) => output.id !== action.id),
        timeline: state.timeline.filter(
          (item) => item.type !== "command" || item.id !== action.id,
        ),
      };
    case "runtime.activity.updated":
      return { ...state, runtimeActivity: action.state };
    case "changes.updated":
      return {
        ...state,
        changeReview: createChangeReviewModel(action.targets),
      };
    case "user.submitted": {
      const turn = createTurn(action.text, "pending");
      return {
        ...state,
        connection: "working",
        statusText: "Working…",
        activeTurnId: turn.id,
        turns: [...state.turns, turn],
        timeline: [...state.timeline, { type: "turn", id: turn.id }],
      };
    }
    case "turn.started": {
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      const active = activeTurn(state);
      if (active && active.state === "pending" && active.userText === action.text) {
        const turnId = action.turnId ?? active.id;
        return {
          ...replaceTimelineTurn(
            updateTurn(state, active.id, (turn) => ({
              ...turn,
              id: turnId,
              state: "running",
              startedAtMs: Date.now(),
              ...(action.modelIdentity ? { modelIdentity: action.modelIdentity } : {}),
            })),
            active.id,
            turnId,
          ),
          activeTurnId: turnId,
        };
      }
      const turn = createTurn(action.text, "running", action.turnId, action.modelIdentity);
      return {
        ...state,
        connection: "working",
        statusText: "Working…",
        activeTurnId: turn.id,
        turns: [...state.turns, turn],
        timeline: [...state.timeline, { type: "turn", id: turn.id }],
      };
    }
    case "turn.cancelling":
      return { ...state, connection: "cancelling", statusText: "Stopping…" };
    case "assistant.delta":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(state, (turn) => ({
        ...turn,
        state: "running",
        assistantText: turn.assistantText + action.text,
        items: appendAssistantText(turn.items, action.text),
      }));
    case "assistant.completed":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(state, (turn) => ({
        ...turn,
        state: "running",
        assistantText: turn.assistantText || action.text,
        items: completeAssistantText(turn.items, action.text),
      }));
    case "assistant.thinking":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(state, (turn) => ({
        ...turn,
        thinkingText: turn.thinkingText + action.text,
      }));
    case "message.created":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      if (action.kind === "user") {
        return state;
      }
      return appendAgentMessage(state, action);
    case "tool.started":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(state, (turn) => ({
        ...turn,
        tools: upsertTool(turn.tools, {
          callId: action.callId,
          name: action.name,
          summary: action.summary,
          state: "running",
          text: "",
          ...(action.arguments ? { arguments: action.arguments } : {}),
        }),
        items: appendTool(turn.items, {
          callId: action.callId,
          name: action.name,
          summary: action.summary,
          state: "running",
          text: "",
          ...(action.arguments ? { arguments: action.arguments } : {}),
        }),
      }));
    case "tool.completed": {
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      const target = findTurnWithTool(state, action.callId) ?? activeTurn(state);
      if (!target) {
        return state;
      }
      return updateTurn(state, target.id, (turn) => {
        const existing = turn.tools.find((tool) => tool.callId === action.callId);
        const tool: ToolActivity = {
          callId: action.callId,
          name: existing?.name ?? "Tool",
          summary: existing?.summary ?? "Completed tool activity",
          state: action.state,
          text: action.text,
          ...(existing?.arguments ? { arguments: existing.arguments } : {}),
          ...(action.path ? { path: action.path } : {}),
        };
        const updatedTools = upsertTool(turn.tools, tool);
        return {
          ...turn,
          tools: updatedTools,
          items: updateToolItem(turn.items, tool),
          changedFiles: action.path && !turn.changedFiles.includes(action.path)
            ? [...turn.changedFiles, action.path]
            : turn.changedFiles,
        };
      });
    }
    case "status.changed":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return { ...state, statusText: action.text };
    case "agent.event":
      return {
        ...state,
        statusText: `${action.label}: ${action.event}`,
        agents: upsertAgent(state.agents, {
          label: action.label,
          state: "working",
          lastEvent: action.event,
          ...(action.text ? { text: action.text } : {}),
          ...(action.records ? { records: action.records } : {}),
        }),
      };
    case "agent.completed":
      return {
        ...state,
        agents: upsertAgent(state.agents, {
          label: action.label,
          state: "completed",
          lastEvent: "completed",
          ...(action.elapsed !== undefined ? { elapsed: action.elapsed } : {}),
          ...(action.tokens !== undefined ? { tokens: action.tokens } : {}),
          ...(action.cost !== undefined ? { cost: action.cost } : {}),
        }),
      };
    case "turn.completed":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(
        { ...state, connection: "ready", statusText: "Ready" },
        (turn) => ({
          ...turn,
          state: "completed",
          elapsedSeconds: elapsedSeconds(turn.startedAtMs),
          // Sagent's completed run result is authoritative. Streaming deltas
          // are useful for responsiveness, but they are not a second source
          // of truth that should be concatenated with the final message.
          assistantText: action.text,
        }),
      );
    case "turn.cancelled":
      if (!eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(
        { ...state, connection: "ready", statusText: "Cancelled" },
        (turn) => ({
          ...turn,
          state: "cancelled",
          elapsedSeconds: elapsedSeconds(turn.startedAtMs),
        }),
      );
    case "turn.failed":
    case "request.failed":
      if (action.type === "turn.failed" && !eventBelongsToSelectedAgent(state, action.agentLabel)) {
        return state;
      }
      return updateActiveTurn(
        {
          ...state,
          connection: "error",
          statusText: "Error",
          lastError: action.type === "turn.failed" ? action.message : action.message,
        },
        (turn) => ({
          ...turn,
          state: "failed",
          elapsedSeconds: elapsedSeconds(turn.startedAtMs),
          error: action.message,
        }),
      );
    case "connection.error":
      return {
        ...state,
        connection: "error",
        statusText: "Connection error",
        lastError: action.message,
      };
    case "connection.disconnected":
      return { ...state, connection: "disconnected", statusText: "Disconnected" };
  }
}

function timelineFromTurns(turns: readonly ConversationTurn[]): ConversationTimelineItem[] {
  return turns.map((turn) => ({ type: "turn", id: turn.id }));
}

function replaceTimelineTurn(
  state: ConversationState,
  previousId: string,
  nextId: string,
): ConversationState {
  if (previousId === nextId) {
    return state;
  }
  return {
    ...state,
    timeline: state.timeline.map((item) =>
      item.type === "turn" && item.id === previousId
        ? { type: "turn", id: nextId }
        : item,
    ),
  };
}

function pruneTimeline(
  timeline: readonly ConversationTimelineItem[],
  commandOutputs: readonly CommandOutputRecord[],
): ConversationTimelineItem[] {
  const commandIds = new Set(commandOutputs.map((output) => output.id));
  return timeline.filter(
    (item) => item.type === "turn" || commandIds.has(item.id),
  );
}

export class ConversationStore {
  private state: ConversationState;
  private readonly listeners = new Set<(state: ConversationState) => void>();

  public constructor(initialState = createInitialConversationState()) {
    this.state = initialState;
  }

  public getState(): ConversationState {
    return this.state;
  }

  public dispatch(action: ConversationAction): void {
    this.state = reduceConversation(this.state, action);
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }

  public subscribe(listener: (state: ConversationState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

function createTurn(
  userText: string,
  state: TurnState,
  id?: string,
  modelIdentity?: ConversationModelIdentity,
): ConversationTurn {
  return {
    id: id ?? `turn-${nextTurnNumber++}`,
    userText,
    assistantText: "",
    thinkingText: "",
    agentMessages: [],
    items: [],
    tools: [],
    changedFiles: [],
    state,
    ...(state === "running" ? { startedAtMs: Date.now() } : {}),
    ...(modelIdentity ? { modelIdentity } : {}),
  };
}

function elapsedSeconds(startedAtMs: number | undefined): number | undefined {
  return startedAtMs === undefined
    ? undefined
    : Math.max(0, (Date.now() - startedAtMs) / 1_000);
}

function turnsFromSnapshot(snapshot: SagentSessionSnapshot): ConversationTurn[] {
  const turns = new Map<string, ConversationTurn>();
  for (const item of snapshot.items) {
    const turn = turns.get(item.turnId) ?? createTurn("", "completed", item.turnId);
    const updatedTurn = applySessionItem(turn, item);
    turns.set(item.turnId, updatedTurn);
  }
  return [...turns.values()];
}

function applySessionItem(
  turn: ConversationTurn,
  item: SagentSessionItem,
): ConversationTurn {
  const modelIdentity = item.model
    ? { modelIdentity: { model: item.model } }
    : {};
  switch (item.type) {
    case "user":
      return {
        ...turn,
        ...modelIdentity,
        userText: item.text ?? "",
      };
    case "agent":
      return {
        ...turn,
        ...modelIdentity,
        agentMessages: [
          ...turn.agentMessages,
          {
            id: item.id,
            text: item.text ?? "",
            ...(item.sender ? { sender: item.sender } : {}),
            ...(item.recipient ? { recipient: item.recipient } : {}),
          },
        ],
      };
    case "assistant":
      return {
        ...turn,
        assistantText: turn.assistantText + (item.text ?? ""),
        items: appendAssistantItem(turn.items, item.id, item.text ?? ""),
      };
    case "thinking":
      return { ...turn, thinkingText: turn.thinkingText + (item.text ?? "") };
    case "tool": {
      if (!item.callId) {
        return turn;
      }
      const tool: ToolActivity = {
        callId: item.callId,
        name: item.name ?? "Tool",
        summary: item.summary ?? item.name ?? "Tool activity",
        state: item.status ?? (item.isError ? "failed" : "running"),
        text: item.text ?? "",
        ...(item.arguments ? { arguments: item.arguments } : {}),
        ...(item.path ? { path: item.path } : {}),
      };
      return {
        ...turn,
        tools: upsertTool(turn.tools, tool),
        items: updateToolItem(turn.items, tool),
        changedFiles: item.path && !turn.changedFiles.includes(item.path)
          ? [...turn.changedFiles, item.path]
          : turn.changedFiles,
      };
    }
  }
}

function appendAssistantText(
  items: ConversationTurnItem[],
  text: string,
): ConversationTurnItem[] {
  const last = items[items.length - 1];
  if (last?.type === "assistant") {
    return [...items.slice(0, -1), { ...last, text: last.text + text }];
  }
  return appendAssistantItem(items, `assistant-${items.length}`, text);
}

function appendAssistantItem(
  items: ConversationTurnItem[],
  id: string,
  text: string,
): ConversationTurnItem[] {
  return [...items, { type: "assistant", id, text }];
}

function completeAssistantText(
  items: ConversationTurnItem[],
  text: string,
): ConversationTurnItem[] {
  const last = items[items.length - 1];
  if (last?.type === "assistant") {
    return [...items.slice(0, -1), { ...last, text }];
  }
  return appendAssistantItem(items, `assistant-${items.length}`, text);
}

function appendTool(items: ConversationTurnItem[], activity: ToolActivity): ConversationTurnItem[] {
  const index = items.findIndex(
    (item) => item.type === "tool" && item.activity.callId === activity.callId,
  );
  if (index < 0) {
    return [...items, { type: "tool", activity }];
  }
  return updateToolItem(items, activity);
}

function updateToolItem(
  items: ConversationTurnItem[],
  activity: ToolActivity,
): ConversationTurnItem[] {
  const index = items.findIndex(
    (item) => item.type === "tool" && item.activity.callId === activity.callId,
  );
  if (index < 0) {
    return [...items, { type: "tool", activity }];
  }
  return items.map((item, itemIndex) =>
    itemIndex === index && item.type === "tool"
      ? { ...item, activity }
      : item,
  );
}

function upsertTool(tools: ToolActivity[], activity: ToolActivity): ToolActivity[] {
  const index = tools.findIndex((tool) => tool.callId === activity.callId);
  if (index < 0) {
    return [...tools, activity];
  }
  return tools.map((tool, toolIndex) => (toolIndex === index ? activity : tool));
}

function appendAgentMessage(
  state: ConversationState,
  action: Extract<ConversationAction, { type: "message.created" }>,
): ConversationState {
  const targetId = action.turnId ?? state.activeTurnId;
  if (targetId && state.turns.some((turn) => turn.id === targetId)) {
    return updateTurn(state, targetId, (turn) => addAgentMessage(turn, action));
  }
  const target = activeTurn(state);
  return target ? updateTurn(state, target.id, (turn) => addAgentMessage(turn, action)) : state;
}

function addAgentMessage(
  turn: ConversationTurn,
  action: Extract<ConversationAction, { type: "message.created" }>,
): ConversationTurn {
  const message: AgentMessage = {
    id: action.id,
    text: action.text,
    ...(action.sender ? { sender: action.sender } : {}),
    ...(action.recipient ? { recipient: action.recipient } : {}),
  };
  return { ...turn, agentMessages: [...turn.agentMessages, message] };
}

function upsertAgent(
  agents: AgentActivity[],
  update: AgentActivity,
): AgentActivity[] {
  const existing = agents.find((agent) => agent.label === update.label);
  if (!existing) {
    return [...agents, update];
  }
  return agents.map((agent) =>
    agent.label === update.label ? { ...agent, ...update } : agent,
  );
}

function activeTurn(state: ConversationState): ConversationTurn | undefined {
  return state.turns.find((turn) => turn.id === state.activeTurnId);
}

function findTurnWithTool(
  state: ConversationState,
  callId: string,
): ConversationTurn | undefined {
  return [...state.turns].reverse().find((turn) => turn.tools.some((tool) => tool.callId === callId));
}

function updateActiveTurn(
  state: ConversationState,
  update: (turn: ConversationTurn) => ConversationTurn,
  activeId = state.activeTurnId,
): ConversationState {
  if (!activeId) {
    return state;
  }
  return updateTurn(state, activeId, update);
}

function updateTurn(
  state: ConversationState,
  turnId: string,
  update: (turn: ConversationTurn) => ConversationTurn,
): ConversationState {
  return {
    ...state,
    turns: state.turns.map((turn) => (turn.id === turnId ? update(turn) : turn)),
  };
}

function eventBelongsToSelectedAgent(
  state: ConversationState,
  agentLabel: string | undefined,
): boolean {
  return agentLabel === undefined || agentLabel === state.selectedAgentLabel;
}

function selectedAgentForTeam(
  currentLabel: string,
  agents: ConversationState["teamAgents"],
): string {
  if (agents.some((agent) => agent.label === currentLabel)) {
    return currentLabel;
  }
  return agents.find((agent) => agent.parent === undefined)?.label
    ?? agents[0]?.label
    ?? currentLabel;
}
