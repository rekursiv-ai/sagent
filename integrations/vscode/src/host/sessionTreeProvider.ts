import * as vscode from "vscode";

import type { SagentSessionInfo, SagentSessionList } from "../shared/sagentProtocol";
import { formatSessionAge } from "../shared/sessionNavigation/sessionNavigationFormatting";

const SESSION_TITLE_ALIASES_KEY = "sagent.sessionTitleAliases";

/**
 * A native VS Code tree for Sagent journals. The provider owns no session
 * records; it only holds the latest listing returned by Sagent for rendering.
 */
export class SagentSessionTreeDataProvider
  implements vscode.TreeDataProvider<SagentSessionItem>, vscode.Disposable
{
  private readonly changeEmitter = new vscode.EventEmitter<
    SagentSessionItem | undefined | null | void
  >();
  private sessions: SagentSessionInfo[] = [];
  private activePath: string | undefined;
  private isStarted = false;
  private startPromise: Promise<void> | undefined;
  private refreshRequest = 0;
  private readonly titleAliases: Record<string, string>;

  public constructor(
    private readonly loadSessions: () => Promise<SagentSessionList>,
    private readonly ensureStarted: () => Promise<void>,
    private readonly workspaceState: vscode.Memento,
  ) {
    this.titleAliases = workspaceState.get<Record<string, string>>(
      SESSION_TITLE_ALIASES_KEY,
      {},
    );
  }

  public readonly onDidChangeTreeData = this.changeEmitter.event;

  public getTreeItem(item: SagentSessionItem): vscode.TreeItem {
    return item;
  }

  public async getChildren(element?: SagentSessionItem): Promise<SagentSessionItem[]> {
    if (!this.isStarted) {
      // Tree views can ask for children more than once while the view is being
      // laid out. Share the startup request so those calls do not race to open
      // separate Sagent processes or sessions. A rejected request is cleared,
      // which leaves the tree able to retry after the user reconnects.
      this.startPromise ??= this.ensureStarted().then(() => {
        this.isStarted = true;
      });
      try {
        await this.startPromise;
      } finally {
        this.startPromise = undefined;
      }
      await this.refresh();
    }
    return element ? [] : this.sessions.map((session) => this.createItem(session));
  }

  public async refresh(): Promise<void> {
    const request = ++this.refreshRequest;
    const listing = await this.loadSessions();
    // A refresh can be triggered by both session selection and message
    // completion. Keep the newest Sagent response authoritative if requests
    // complete out of order.
    if (request !== this.refreshRequest) {
      return;
    }
    this.sessions = listing.sessions;
    this.activePath = listing.activePath;
    this.changeEmitter.fire();
  }

  public setActivePath(path: string | undefined): void {
    if (this.activePath === path) {
      return;
    }
    this.activePath = path;
    this.changeEmitter.fire();
  }

  public async renameSession(item: SagentSessionItem): Promise<void> {
    const currentTitle = this.titleFor(item.session);
    const title = await vscode.window.showInputBox({
      prompt: "Rename Sagent session",
      value: currentTitle,
      validateInput: (value) => value.trim() ? undefined : "Enter a session name.",
      ignoreFocusOut: true,
    });
    if (title === undefined) {
      return;
    }
    const trimmed = title.trim();
    if (trimmed === currentTitle) {
      return;
    }
    this.titleAliases[item.session.path] = trimmed;
    await this.workspaceState.update(SESSION_TITLE_ALIASES_KEY, this.titleAliases);
    this.changeEmitter.fire(item);
  }

  public withDisplayTitles(listing: SagentSessionList): SagentSessionList {
    return {
      ...listing,
      sessions: listing.sessions.map((session) => ({
        ...session,
        title: this.titleFor(session),
      })),
    };
  }

  public dispose(): void {
    this.changeEmitter.dispose();
  }

  private createItem(session: SagentSessionInfo): SagentSessionItem {
    const isActive = session.path === this.activePath;
    return new SagentSessionItem(session, isActive, this.titleFor(session));
  }

  private titleFor(session: SagentSessionInfo): string {
    return this.titleAliases[session.path] ?? session.title ?? "";
  }
}

export class SagentSessionItem extends vscode.TreeItem {
  public constructor(
    public readonly session: SagentSessionInfo,
    isActive: boolean,
    displayTitle?: string,
  ) {
    super(displayTitle || session.title || "Untitled session", vscode.TreeItemCollapsibleState.None);
    // Sagent's journal path is stable for the session and lets VS Code retain
    // selection state when the native tree is refreshed.
    this.id = session.path;
    this.contextValue = "sagentSession";
    this.description = [
      formatSessionAge(session.mtime),
      `${session.records} records`,
      session.model,
    ]
      .filter(Boolean)
      .join(" · ");
    this.tooltip = `${displayTitle || session.title || "Untitled session"}\n${session.path}`;
    this.iconPath = new vscode.ThemeIcon(isActive ? "check" : "comment-discussion");
    this.command = {
      command: "sagent.selectSession",
      title: "Open Sagent Session",
      arguments: [session.path],
    };
  }
}
