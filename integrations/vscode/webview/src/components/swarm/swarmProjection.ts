import type { AgentActivity } from "../../../../src/shared/conversationTypes";
import type { SagentTeamAgent } from "../../../../src/shared/sagentProtocol";

export type SwarmLoadState = "loading" | "ready" | "error";

export type SwarmViewState = {
  loadState: SwarmLoadState;
  teamOpen: boolean;
  agents: readonly SagentTeamAgent[];
  selectedAgentLabel?: string;
  activities: readonly AgentActivity[];
  errorMessage?: string;
};

/**
 * Actions are deliberately about intent, not implementation. The parent owns
 * the Sagent client and maps select-agent to getAgentSession(label), so this
 * component never creates a second session model or reaches across the webview
 * boundary itself.
 */
export type SwarmViewAction =
  | { type: "select-agent"; agentLabel: string }
  | { type: "retry" };

export type SwarmTreeNode = {
  agent: SagentTeamAgent;
  children: readonly SwarmTreeNode[];
  depth: number;
};

/**
 * Build the display tree from Team's structured relationships. The server's
 * children order is preserved; labels and human-readable status are never
 * inspected to infer hierarchy.
 *
 * The final pass keeps a malformed or partially refreshed projection visible
 * instead of silently dropping an agent. This is only a presentation fallback;
 * it does not repair or mutate Sagent's Team state.
 */
export function buildSwarmTree(
  agents: readonly SagentTeamAgent[],
): readonly SwarmTreeNode[] {
  const agentsByLabel = new Map(agents.map((agent) => [agent.label, agent]));
  const visitedLabels = new Set<string>();

  const createNode = (agent: SagentTeamAgent, depth: number): SwarmTreeNode => {
    visitedLabels.add(agent.label);
    const children = agent.children.flatMap((childLabel) => {
      const child = agentsByLabel.get(childLabel);
      if (!child || visitedLabels.has(child.label)) {
        return [];
      }
      return [createNode(child, depth + 1)];
    });
    return { agent, children, depth };
  };

  const roots = agents.filter(
    (agent) => agent.parent === undefined || !agentsByLabel.has(agent.parent),
  );
  const tree: SwarmTreeNode[] = [];
  for (const agent of roots) {
    if (!visitedLabels.has(agent.label)) {
      tree.push(createNode(agent, 0));
    }
  }

  // Team normally makes parent/children mutually consistent. Keeping any
  // unvisited entry as a root makes transient snapshots observable without
  // guessing a relationship the native projection did not provide.
  for (const agent of agents) {
    if (!visitedLabels.has(agent.label)) {
      tree.push(createNode(agent, 0));
    }
  }

  return tree;
}

export function activityForAgent(
  activities: readonly AgentActivity[],
  agentLabel: string,
): AgentActivity | undefined {
  return activities.find((activity) => activity.label === agentLabel);
}
