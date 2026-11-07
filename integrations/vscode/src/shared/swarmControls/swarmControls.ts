import type { SagentTeamAgent, SagentTeamState } from "../sagentProtocol";

/**
 * User intents exposed by the swarm surface.
 *
 * These are UI intents, not Sagent operations. In particular, send and spawn
 * intentionally carry no copied AgentSend or AgentSpawn argument schema. The
 * eventual host seam must call those native Sagent objects and own their
 * validation.
 */
export type SwarmControlAction =
  | { type: "focus-agent"; agentLabel: string }
  | { type: "open-agent-details"; agentLabel: string }
  | { type: "open-send-composer"; agentLabel: string }
  | { type: "halt-agent"; agentLabel: string }
  | { type: "kill-agent"; agentLabel: string }
  | { type: "open-spawn-composer"; parentLabel: string };

export type SwarmControlType = SwarmControlAction["type"];

export type SwarmControlAffordance = {
  type: SwarmControlType;
  label: string;
  enabled: boolean;
  disabledReason?: string;
};

/**
 * Capabilities supplied by the integration host. The current host supports
 * focus through the existing native command and details from getTeam. The
 * remaining flags stay false until the server exposes the corresponding
 * native Sagent operation.
 */
export type SwarmControlCapabilities = Readonly<{
  focus: boolean;
  details: boolean;
  send: boolean;
  halt: boolean;
  kill: boolean;
  spawn: boolean;
}>;

export const currentSwarmControlCapabilities: SwarmControlCapabilities = {
  focus: true,
  details: true,
  send: false,
  halt: false,
  kill: false,
  spawn: false,
};

export type SwarmAgentDetails = {
  /** The original native projection; no agent metadata is copied or normalized. */
  agent: SagentTeamAgent;
  /** Resolved from the native parent label, if the current projection contains it. */
  parent?: SagentTeamAgent;
  /** Resolved only from the native children label list, preserving its order. */
  children: readonly SagentTeamAgent[];
};

export type SwarmAgentControlModel = {
  details: SwarmAgentDetails;
  isFocused: boolean;
  affordances: readonly SwarmControlAffordance[];
};

export type SwarmControlsModel = {
  teamOpen: boolean;
  agents: readonly SwarmAgentControlModel[];
};

/**
 * The host can pass this contract to a replaceable React or native command
 * surface without giving that surface access to the Sagent client.
 */
export type SwarmControlsIntegrationContract = {
  model: SwarmControlsModel;
  onAction: (action: SwarmControlAction) => void;
};

export function createSwarmControlsModel(
  team: SagentTeamState,
  selectedAgentLabel: string | undefined,
  capabilities: SwarmControlCapabilities = currentSwarmControlCapabilities,
): SwarmControlsModel {
  const agentsByLabel = new Map(team.agents.map((agent) => [agent.label, agent]));

  return {
    teamOpen: team.open,
    agents: team.agents.map((agent) => ({
      details: createAgentDetails(agent, agentsByLabel),
      isFocused: agent.label === selectedAgentLabel,
      affordances: createAffordances(capabilities),
    })),
  };
}

export function createAgentDetails(
  agent: SagentTeamAgent,
  agentsByLabel: ReadonlyMap<string, SagentTeamAgent>,
): SwarmAgentDetails {
  const children = agent.children.flatMap((label) => {
    const child = agentsByLabel.get(label);
    return child ? [child] : [];
  });
  const parent = agent.parent ? agentsByLabel.get(agent.parent) : undefined;

  return {
    agent,
    ...(parent ? { parent } : {}),
    children,
  };
}

function createAffordances(
  capabilities: SwarmControlCapabilities,
): readonly SwarmControlAffordance[] {
  return [
    affordance(
      "focus-agent",
      "Focus",
      capabilities.focus,
      "Agent focus is not available from the current Sagent host.",
    ),
    affordance(
      "open-agent-details",
      "Details",
      capabilities.details,
      "Agent details are not available from the current Sagent host.",
    ),
    affordance(
      "open-send-composer",
      "Send",
      capabilities.send,
      "Sending to another agent requires a native AgentSend server operation.",
    ),
    affordance(
      "halt-agent",
      "Halt",
      capabilities.halt,
      "Halting a child requires its native Agent runtime to be addressed by the server.",
    ),
    affordance(
      "kill-agent",
      "Kill",
      capabilities.kill,
      "Killing an agent requires the native Team.stop operation on the server.",
    ),
    affordance(
      "open-spawn-composer",
      "Spawn",
      capabilities.spawn,
      "Spawning requires the native AgentSpawn tool with a real ToolContext.",
    ),
  ];
}

function affordance(
  type: SwarmControlType,
  label: string,
  enabled: boolean,
  disabledReason: string,
): SwarmControlAffordance {
  return {
    type,
    label,
    enabled,
    ...(enabled ? {} : { disabledReason }),
  };
}
