import { useEffect, useMemo, useState, type ReactElement } from "react";

import type { AgentActivity } from "../../../../src/shared/conversationTypes";
import type { SagentTeamAgent } from "../../../../src/shared/sagentProtocol";
import { joinClasses, sagentUiTokens } from "../../ui";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";
import {
  activityForAgent,
  buildSwarmTree,
  type SwarmTreeNode,
  type SwarmViewAction,
  type SwarmViewState,
} from "./swarmProjection";

export type SwarmViewProps = {
  state: SwarmViewState;
  onAction: (action: SwarmViewAction) => void;
};

/**
 * Rich swarm presentation for a parent webview to compose when it wants the
 * live Team beside conversation content. Native VS Code's TreeDataProvider
 * remains useful for compact navigation; this component owns only the richer
 * React presentation and shares the same Sagent projection.
 */
export function SwarmView({ state, onAction }: SwarmViewProps): ReactElement {
  const nodes = useMemo(() => buildSwarmTree(state.agents), [state.agents]);

  if (state.loadState === "loading") {
    return <SwarmMessageState title="Loading swarm" detail="Reading the live Sagent Team…" />;
  }

  if (state.loadState === "error") {
    return (
      <SwarmMessageState
        action={
          <Button onClick={() => onAction({ type: "retry" })}>
            <Icon name="refresh" />
            Retry
          </Button>
        }
        detail={state.errorMessage ?? "The Sagent Team could not be loaded."}
        title="Swarm unavailable"
      />
    );
  }

  if (!state.teamOpen || state.agents.length === 0) {
    return (
      <SwarmMessageState
        detail="Start a Sagent turn or spawn an agent to see the live hierarchy here."
        title="No active agents"
      />
    );
  }

  return (
    <section aria-label="Sagent swarm" className="flex min-h-0 flex-col gap-2">
      <SwarmHeader
        activeCount={state.agents.filter((agent) => agent.running).length}
        agentCount={state.agents.length}
      />
      <SwarmTree
        activities={state.activities}
        nodes={nodes}
        onAction={onAction}
        selectedAgentLabel={state.selectedAgentLabel}
      />
    </section>
  );
}

function SwarmHeader({ activeCount, agentCount }: { activeCount: number; agentCount: number }): ReactElement {
  return (
    <div className={joinClasses("flex items-center justify-between gap-2 px-1", sagentUiTokens.mutedForeground)}>
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide">
        <Icon name="organization" />
        <span>Swarm</span>
      </div>
      <span className="text-xs" role="status" aria-live="polite">
        {activeCount > 0 ? `${activeCount} working · ${agentCount} total` : `${agentCount} agents`}
      </span>
    </div>
  );
}

function SwarmTree({
  activities,
  nodes,
  onAction,
  selectedAgentLabel,
}: {
  activities: readonly AgentActivity[];
  nodes: readonly SwarmTreeNode[];
  onAction: (action: SwarmViewAction) => void;
  selectedAgentLabel?: string;
}): ReactElement {
  const [expandedLabels, setExpandedLabels] = useState<Set<string>>(
    () => new Set(nodes.filter((node) => node.children.length > 0).map((node) => node.agent.label)),
  );

  useEffect(() => {
    setExpandedLabels((current) => {
      const availableLabels = collectNodeLabels(nodes);
      const next = new Set(
        [...current].filter((label) => availableLabels.has(label)),
      );
      for (const node of nodes) {
        if (node.children.length > 0 && !current.has(node.agent.label)) {
          next.add(node.agent.label);
        }
      }
      return next;
    });
  }, [nodes]);

  const toggleExpanded = (label: string): void => {
    setExpandedLabels((current) => {
      const next = new Set(current);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  return (
    <ul aria-label="Agent hierarchy" className="m-0 flex min-h-0 flex-col gap-0.5 overflow-auto p-0" role="tree">
      {nodes.map((node) => (
        <SwarmTreeNodeView
          activities={activities}
          expandedLabels={expandedLabels}
          key={node.agent.label}
          node={node}
          onAction={onAction}
          onToggleExpanded={toggleExpanded}
          selectedAgentLabel={selectedAgentLabel}
        />
      ))}
    </ul>
  );
}

function collectNodeLabels(nodes: readonly SwarmTreeNode[]): Set<string> {
  const labels = new Set<string>();
  const visit = (node: SwarmTreeNode): void => {
    labels.add(node.agent.label);
    node.children.forEach(visit);
  };
  nodes.forEach(visit);
  return labels;
}

function SwarmTreeNodeView({
  activities,
  expandedLabels,
  node,
  onAction,
  onToggleExpanded,
  selectedAgentLabel,
}: {
  activities: readonly AgentActivity[];
  expandedLabels: ReadonlySet<string>;
  node: SwarmTreeNode;
  onAction: (action: SwarmViewAction) => void;
  onToggleExpanded: (label: string) => void;
  selectedAgentLabel?: string;
}): ReactElement {
  const { agent, children } = node;
  const isExpanded = expandedLabels.has(agent.label);
  const activity = activityForAgent(activities, agent.label);
  const status = getAgentStatus(agent, activity);
  const isSelected = agent.label === selectedAgentLabel;

  return (
    <li className="m-0 list-none" role="none">
      <div
        aria-level={node.depth + 1}
        aria-selected={isSelected}
        className={joinClasses(
          "flex min-w-0 items-start gap-1 rounded px-1 py-1",
          isSelected ? "bg-sagent-hover" : "hover:bg-sagent-hover",
        )}
        role="treeitem"
      >
        {children.length > 0 ? (
          <button
            aria-label={`${isExpanded ? "Collapse" : "Expand"} ${agent.label}`}
            aria-expanded={isExpanded}
            className={joinClasses(
              "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded",
              sagentUiTokens.focus,
              "text-sagent-muted-foreground hover:bg-sagent-hover",
            )}
            onClick={() => onToggleExpanded(agent.label)}
            type="button"
          >
            <span aria-hidden="true" className={joinClasses("codicon", isExpanded ? "codicon-chevron-down" : "codicon-chevron-right")} />
          </button>
        ) : (
          <span aria-hidden="true" className="h-5 w-5 shrink-0" />
        )}
        <button
          aria-label={`Select agent ${agent.label}`}
          className={joinClasses("min-w-0 flex-1 rounded text-left", sagentUiTokens.focus)}
          onClick={() => onAction({ type: "select-agent", agentLabel: agent.label })}
          type="button"
        >
          <AgentSummary
            agent={agent}
            activity={activity}
            isRoot={agent.parent === undefined}
            status={status}
          />
        </button>
      </div>
      {isExpanded && children.length > 0 && (
        <ul className="m-0 ml-4 border-l border-sagent-border pl-1" role="group">
          {children.map((child) => (
            <SwarmTreeNodeView
              activities={activities}
              expandedLabels={expandedLabels}
              key={child.agent.label}
              node={child}
              onAction={onAction}
              onToggleExpanded={onToggleExpanded}
              selectedAgentLabel={selectedAgentLabel}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function AgentSummary({
  agent,
  activity,
  isRoot,
  status,
}: {
  agent: SagentTeamAgent;
  activity: AgentActivity | undefined;
  isRoot: boolean;
  status: AgentStatus;
}): ReactElement {
  const model = agent.provider && agent.model ? `${agent.provider}/${agent.model}` : agent.model ?? agent.provider;
  const relationship = agent.watching ? `watching ${agent.watching}` : agent.parent ? `child of ${agent.parent}` : "root agent";
  const detail = activity?.lastEvent ?? relationship;
  const owedReplyCount = Object.keys(agent.owedReplies).length;

  return (
    <span className="flex min-w-0 flex-col gap-0.5">
      <span className="flex min-w-0 items-center gap-1.5">
        <span aria-hidden="true" className={joinClasses("h-1.5 w-1.5 shrink-0 rounded-full", statusDotClass(status))} />
        <span className="min-w-0 truncate text-sm font-medium text-sagent-foreground">{agent.label}</span>
        {isRoot && <span className="text-xs text-sagent-muted-foreground">root</span>}
        {agent.persistent && <span className="text-xs text-sagent-muted-foreground">persistent</span>}
      </span>
      <span className="ml-3 flex min-w-0 items-center gap-1 text-xs text-sagent-muted-foreground">
        <span className="min-w-0 truncate">{detail}</span>
        {model && <span className="shrink-0 truncate">· {model}</span>}
      </span>
      {owedReplyCount > 0 && (
        <span className="ml-3 text-xs text-sagent-muted-foreground">
          {owedReplyCount} {owedReplyCount === 1 ? "reply" : "replies"} owed
        </span>
      )}
    </span>
  );
}

function SwarmMessageState({
  action,
  detail,
  title,
}: {
  action?: ReactElement;
  detail: string;
  title: string;
}): ReactElement {
  return (
    <section aria-live="polite" className="flex flex-col items-center gap-2 px-3 py-6 text-center">
      <Icon name="organization" className="text-lg text-sagent-muted-foreground" />
      <h2 className="m-0 text-sm font-medium text-sagent-foreground">{title}</h2>
      <p className="m-0 max-w-sm text-xs text-sagent-muted-foreground">{detail}</p>
      {action}
    </section>
  );
}

type AgentStatus = "working" | "failed" | "idle";

function getAgentStatus(agent: SagentTeamAgent, activity: AgentActivity | undefined): AgentStatus {
  if (activity?.state === "failed") {
    return "failed";
  }
  if (agent.running || activity?.state === "working") {
    return "working";
  }
  return "idle";
}

function statusDotClass(status: AgentStatus): string {
  switch (status) {
    case "working":
      return "bg-sagent-focus";
    case "failed":
      return "bg-[var(--vscode-errorForeground)]";
    case "idle":
      return "bg-[var(--vscode-testing-iconPassed)]";
  }
}
