import * as vscode from "vscode";

import { SagentViewProvider } from "./host/sagentViewProvider";
import { SagentSessionItem } from "./host/sessionTreeProvider";

export function activate(context: vscode.ExtensionContext): void {
  const provider = new SagentViewProvider(context.extensionUri, context.workspaceState);
  context.subscriptions.push(
    provider,
    registerSagentView("sagent.chat", provider),
    registerSagentView("sagent.chatSecondary", provider),
    vscode.window.registerTreeDataProvider(
      "sagent.sessions",
      provider.getSessionTreeDataProvider(),
    ),
    vscode.window.registerTreeDataProvider(
      "sagent.sessionsSecondary",
      provider.getSessionTreeDataProvider(),
    ),
    vscode.window.registerTreeDataProvider(
      "sagent.swarm",
      provider.getSwarmTreeDataProvider(),
    ),
    vscode.window.registerTreeDataProvider(
      "sagent.swarmSecondary",
      provider.getSwarmTreeDataProvider(),
    ),
    vscode.commands.registerCommand("sagent.open", () => provider.open()),
    vscode.commands.registerCommand(
      "sagent.selectSession",
      (sessionPath: unknown) =>
        typeof sessionPath === "string" ? provider.selectSession(sessionPath) : undefined,
    ),
    vscode.commands.registerCommand(
      "sagent.newSession",
      () => provider.newSession(),
    ),
    vscode.commands.registerCommand(
      "sagent.renameSession",
      (item: unknown) =>
        item instanceof SagentSessionItem ? provider.renameSession(item) : undefined,
    ),
    vscode.commands.registerCommand(
      "sagent.selectAgent",
      (label: unknown) =>
        typeof label === "string" ? provider.selectAgentFromTree(label) : undefined,
    ),
  );
}

export function deactivate(): void {
  // The provider is disposed by VS Code through the extension context.
}

function registerSagentView(
  viewId: string,
  provider: SagentViewProvider,
): vscode.Disposable {
  return vscode.window.registerWebviewViewProvider(viewId, provider, {
    webviewOptions: { retainContextWhenHidden: true },
  });
}
