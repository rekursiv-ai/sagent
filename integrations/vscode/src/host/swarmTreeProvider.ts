import * as vscode from "vscode";

import type { SagentTeamAgent } from "../shared/sagentProtocol";

/**
 * Renders Sagent's native Team projection with VS Code's built-in tree chrome.
 * Parent/child relationships come from Team; labels are never parsed to infer
 * hierarchy in the extension.
 */
export class SagentSwarmTreeDataProvider
  implements vscode.TreeDataProvider<SagentSwarmItem>, vscode.Disposable
{
  private readonly changeEmitter = new vscode.EventEmitter<
    SagentSwarmItem | undefined | null | void
  >();
  private agents = new Map<string, SagentTeamAgent>();
  private selectedLabel: string | undefined;
  private startRequested = false;

  public constructor(private readonly ensureStarted: () => Promise<void>) {}

  public readonly onDidChangeTreeData = this.changeEmitter.event;

  public getTreeItem(item: SagentSwarmItem): vscode.TreeItem {
    return item;
  }

  public async getChildren(element?: SagentSwarmItem): Promise<SagentSwarmItem[]> {
    if (!this.startRequested) {
      this.startRequested = true;
      await this.ensureStarted();
    }
    const agents = element
      ? element.agent.children.flatMap((label) => this.agentFor(label))
      : [...this.agents.values()].filter((agent) => agent.parent === undefined);
    return agents.map((agent) => this.createItem(agent));
  }

  public update(agents: SagentTeamAgent[], selectedLabel: string): void {
    this.agents = new Map(agents.map((agent) => [agent.label, agent]));
    this.selectedLabel = selectedLabel;
    this.changeEmitter.fire();
  }

  public dispose(): void {
    this.changeEmitter.dispose();
  }

  private agentFor(label: string): SagentTeamAgent[] {
    const agent = this.agents.get(label);
    return agent ? [agent] : [];
  }

  private createItem(agent: SagentTeamAgent): SagentSwarmItem {
    return new SagentSwarmItem(
      agent,
      agent.label === this.selectedLabel,
    );
  }
}

class SagentSwarmItem extends vscode.TreeItem {
  public constructor(
    public readonly agent: SagentTeamAgent,
    isSelected: boolean,
  ) {
    super(
      agent.label,
      agent.children.length > 0
        ? vscode.TreeItemCollapsibleState.Collapsed
        : vscode.TreeItemCollapsibleState.None,
    );
    // Team labels are the native stable identity of live agents. Providing an
    // id lets VS Code preserve tree selection and expansion across refreshes.
    this.id = agent.label;
    this.contextValue = "sagentAgent";
    const model = agent.provider && agent.model
      ? `${agent.provider}/${agent.model}`
      : agent.model;
    this.description = [
      agent.running ? "working" : "idle",
      isSelected ? "focused" : undefined,
      model,
    ].filter(Boolean).join(" · ");
    this.tooltip = createAgentTooltip(agent);
    this.iconPath = new vscode.ThemeIcon(
      isSelected ? "check" : agent.running ? "sync~spin" : "organization",
    );
    this.command = {
      command: "sagent.selectAgent",
      title: "Focus Sagent Agent",
      arguments: [agent.label],
    };
  }
}

function createAgentTooltip(agent: SagentTeamAgent): vscode.MarkdownString {
  const tooltip = new vscode.MarkdownString();
  tooltip.appendText(agent.label);
  tooltip.appendMarkdown("\n\n");
  tooltip.appendText(agent.running ? "Working" : "Idle");
  if (agent.parent) {
    tooltip.appendMarkdown("\n");
    tooltip.appendText("Spawned by: ");
    tooltip.appendText(agent.parent);
  } else if (agent.watching) {
    tooltip.appendMarkdown("\n");
    tooltip.appendText("Watching: ");
    tooltip.appendText(agent.watching);
  }
  if (agent.provider || agent.model) {
    tooltip.appendMarkdown("\n");
    tooltip.appendText("Model: ");
    tooltip.appendText(
      agent.provider && agent.model
        ? `${agent.provider}/${agent.model}`
        : agent.provider ?? agent.model ?? "",
    );
  }
  if (agent.persistent) {
    tooltip.appendMarkdown("\n");
    tooltip.appendText("Persistent child");
  }
  if (agent.tools.length > 0) {
    tooltip.appendMarkdown("\n");
    tooltip.appendText(`Tools: ${agent.tools.length}`);
  }
  return tooltip;
}

export type { SagentSwarmItem };
