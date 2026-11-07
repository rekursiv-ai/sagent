import type { ChangeReviewModel } from "./changeReviewModel";
import {
  createInitialRuntimeActivityState,
  type RuntimeActivityState,
} from "./runtimeActivity/runtimeActivityModel";
import type { ModelSettingsSnapshot } from "./modelSettings/modelSettingsContract";
import type { SessionNavigationState } from "./sessionNavigation";
import type { SagentCommandDescriptor } from "./sagentProtocol";

export type ConnectionState =
  | "starting"
  | "ready"
  | "working"
  | "cancelling"
  | "disconnected"
  | "error";

export type TurnState = "pending" | "running" | "completed" | "cancelled" | "failed";

export type ToolActivityState =
  | "running"
  | "completed"
  | "failed"
  | "detached"
  | "cancelled";

export type AgentMessage = {
  id: string;
  text: string;
  sender?: string;
  recipient?: string;
};

export type AgentActivityState = "working" | "completed" | "failed";

export type AgentActivityRecord = {
  kind: "assistant" | "tool";
  callId?: string;
  name?: string;
  text?: string;
};

export type AgentActivity = {
  label: string;
  state: AgentActivityState;
  lastEvent: string;
  text?: string;
  elapsed?: number;
  tokens?: number;
  cost?: number;
  records?: AgentActivityRecord[];
};

export type ToolActivity = {
  callId: string;
  name: string;
  summary: string;
  state: ToolActivityState;
  text: string;
  path?: string;
  arguments?: Record<string, unknown>;
};

export type ConversationTurnItem =
  | { type: "assistant"; id: string; text: string }
  | { type: "tool"; activity: ToolActivity };

export type ConversationModelIdentity = {
  provider?: string;
  model: string;
};

export type ConversationTurn = {
  id: string;
  userText: string;
  assistantText: string;
  thinkingText: string;
  agentMessages: AgentMessage[];
  /** Ordered native Sagent output and tool calls for chronological rendering. */
  items: ConversationTurnItem[];
  tools: ToolActivity[];
  changedFiles: string[];
  state: TurnState;
  /** Local presentation clock for the current turn; not agent runtime state. */
  startedAtMs?: number;
  elapsedSeconds?: number;
  /** The native Sagent model captured when this turn started. */
  modelIdentity?: ConversationModelIdentity;
  error?: string;
};

export type CommandOutputRecord = {
  id: string;
  agentLabel: string;
  text: string;
  tone: "line" | "note" | "error";
  command?: string;
};

/**
 * The presentation order shared by normal turns and slash-command output.
 *
 * Turns and command output remain separate records because they have different
 * update lifecycles, but the webview must render them from one ordered stream.
 */
export type ConversationTimelineItem =
  | { type: "turn"; id: string }
  | { type: "command"; id: string };

export type ConversationState = {
  connection: ConnectionState;
  statusText: string;
  sessionPath?: string;
  sessionNavigation: SessionNavigationState;
  teamOpen: boolean;
  teamAgents: SagentTeamAgent[];
  selectedAgentLabel: string;
  /** Live settings are optional for compatibility with persisted webview state. */
  modelSettings?: ModelSettingsSnapshot;
  commandDescriptors: SagentCommandDescriptor[];
  commandOutputs: CommandOutputRecord[];
  timeline: ConversationTimelineItem[];
  runtimeActivity: RuntimeActivityState;
  changeReview: ChangeReviewModel;
  lastError?: string;
  agents: AgentActivity[];
  turns: ConversationTurn[];
  activeTurnId?: string;
};

export function createInitialConversationState(): ConversationState {
  return {
    connection: "disconnected",
    statusText: "Not connected",
    sessionNavigation: {
      status: "loading",
      sessions: [],
    },
    teamOpen: false,
    teamAgents: [],
    selectedAgentLabel: "Agent",
    runtimeActivity: createInitialRuntimeActivityState(),
    changeReview: {
      status: "empty",
      message: "No file changes are available to review.",
    },
    agents: [],
    turns: [],
    commandDescriptors: [],
    commandOutputs: [],
    timeline: [],
  };
}
import type { SagentTeamAgent } from "./sagentProtocol";

export type { SagentTeamAgent } from "./sagentProtocol";
