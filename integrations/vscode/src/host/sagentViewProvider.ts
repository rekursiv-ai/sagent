import * as vscode from "vscode";
import * as net from "node:net";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import {
  ConversationStore,
  type ConversationAction,
} from "./conversationStore";
import type { AgentActivityRecord } from "../shared/conversationTypes";
import { SagentClient, SagentExecutableMissingError } from "./sagentClient";
import {
  parseSagentFileChange,
  type SagentEvent,
  type SagentToolStatus,
} from "../shared/sagentProtocol";
import type {
  ModelSettingsAction,
  ModelSettingsScope,
} from "../shared/modelSettings/modelSettingsContract";
import type {
  SagentRekursivContext,
  SagentRekursivSkillCandidate,
} from "../shared/sagentProtocol";
import {
  applyRuntimeActivityEvent,
  setChangeReviewTargets,
  setTeamDiagnostics,
  toRuntimeActivityEvent,
} from "../shared/runtimeActivity/runtimeActivityModel";
import {
  SagentChangeReviewService,
  toChangeReviewTarget,
  type SagentChangeReviewTarget,
} from "./changeReviewService";
import { toChangeReviewPresentationTarget } from "../shared/changeReviewModel";
import { createSagentWebviewHtml } from "./webviewHtml";
import { SagentSessionTreeDataProvider, type SagentSessionItem } from "./sessionTreeProvider";
import { SagentSwarmTreeDataProvider } from "./swarmTreeProvider";
import type { ExtensionToWebviewMessage } from "../shared/webviewProtocol";

const DEFAULT_TRACKINIZER_URL = "http://127.0.0.1:8765";

async function isUrlReachable(targetUrl: string, timeoutMs = 250): Promise<boolean> {
  try {
    const parsed = new URL(targetUrl);
    const host = parsed.hostname;
    if (!host) {
      return false;
    }
    const port = parsed.port ? parseInt(parsed.port, 10) : (parsed.protocol === "https:" ? 443 : 80);
    return await new Promise<boolean>((resolve) => {
      const socket = net.createConnection({ host, port }, () => {
        socket.destroy();
        resolve(true);
      });
      socket.setTimeout(timeoutMs);
      socket.on("timeout", () => {
        socket.destroy();
        resolve(false);
      });
      socket.on("error", () => {
        socket.destroy();
        resolve(false);
      });
    });
  } catch {
    return false;
  }
}

let nextCommandOutputId = 1;

export class SagentViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
  private readonly views = new Set<vscode.WebviewView>();
  private readonly client: SagentClient;
  private readonly changeReviewService = new SagentChangeReviewService();
  private readonly store = new ConversationStore();
  private readonly eventSubscription: vscode.Disposable;
  private readonly storeUnsubscribe: () => void;
  private readonly sessionTreeProvider: SagentSessionTreeDataProvider;
  private readonly swarmTreeProvider: SagentSwarmTreeDataProvider;
  private readonly viewSubscriptions = new Map<vscode.WebviewView, vscode.Disposable>();
  private readonly pendingComposerInsertions: ExtensionToWebviewMessage[] = [];
  private changeReviewTargets = new Map<string, SagentChangeReviewTarget>();
  private startPromise: Promise<void> | undefined;
  private teamRefreshPromise: Promise<void> | undefined;
  private agentSelectionRequest = 0;
  private sessionSelectionRequest = 0;

  public constructor(
    private readonly extensionUri: vscode.Uri,
    workspaceState: vscode.Memento,
  ) {
    this.client = new SagentClient(extensionUri);
    this.eventSubscription = this.client.onEvent((event) => this.handleSagentEvent(event));
    this.sessionTreeProvider = new SagentSessionTreeDataProvider(
      () => this.loadSessionsForTree(),
      () => this.start(),
      workspaceState,
    );
    this.swarmTreeProvider = new SagentSwarmTreeDataProvider(() => this.start());
    this.storeUnsubscribe = this.store.subscribe((state) => {
      this.sessionTreeProvider.setActivePath(state.sessionPath);
      this.swarmTreeProvider.update(state.teamAgents, state.selectedAgentLabel);
      this.post({ type: "state", state });
    });
  }

  public getSessionTreeDataProvider(): SagentSessionTreeDataProvider {
    return this.sessionTreeProvider;
  }

  public getSwarmTreeDataProvider(): SagentSwarmTreeDataProvider {
    return this.swarmTreeProvider;
  }

  public resolveWebviewView(view: vscode.WebviewView): void {
    this.viewSubscriptions.get(view)?.dispose();
    this.views.add(view);
    view.webview.options = { enableScripts: true };
    view.webview.html = createSagentWebviewHtml(view.webview, this.extensionUri);
    const messageSubscription = view.webview.onDidReceiveMessage((message: unknown) => {
      void this.handleMessage(message);
    });
    const viewDisposeSubscription = view.onDidDispose(() => {
      this.views.delete(view);
      this.viewSubscriptions.delete(view);
      messageSubscription.dispose();
    });
    this.viewSubscriptions.set(
      view,
      new vscode.Disposable(() => {
        messageSubscription.dispose();
        viewDisposeSubscription.dispose();
      }),
    );
    this.post({ type: "state", state: this.store.getState() });
  }

  public async open(): Promise<void> {
    for (const view of this.views) {
      view.show(false);
    }
    try {
      await vscode.commands.executeCommand("sagent.chatSecondary.focus");
    } catch {
      try {
        await vscode.commands.executeCommand("workbench.view.extension.sagentSecondary");
      } catch {
        try {
          await vscode.commands.executeCommand("workbench.view.extension.sagent");
        } catch {
          // View cannot be revealed directly
        }
      }
    }
  }

  public dispose(): void {
    for (const subscription of this.viewSubscriptions.values()) {
      subscription.dispose();
    }
    this.viewSubscriptions.clear();
    this.views.clear();
    this.eventSubscription.dispose();
    this.storeUnsubscribe();
    this.sessionTreeProvider.dispose();
    this.swarmTreeProvider.dispose();
    this.client.dispose();
  }

  private async handleMessage(message: unknown): Promise<void> {
    if (!isMessage(message)) {
      return;
    }
    if (message.type === "ready") {
      this.post({ type: "state", state: this.store.getState() });
      while (this.pendingComposerInsertions.length > 0) {
        const insertion = this.pendingComposerInsertions.shift();
        if (insertion) {
          this.post(insertion);
        }
      }
      await this.start();
    } else if (message.type === "send" && typeof message.text === "string") {
      await this.send(message.text);
    } else if (message.type === "command" && typeof message.line === "string") {
      await this.executeCommand(message.line);
    } else if (message.type === "remove-command-output" && typeof message.id === "string") {
      this.store.dispatch({ type: "command.output.removed", id: message.id });
    } else if (message.type === "cancel") {
      await this.cancel();
    } else if (message.type === "retry") {
      this.client.dispose();
      await this.start(true);
    } else if (message.type === "select-session" && typeof message.path === "string") {
      await this.selectSession(message.path);
    } else if (message.type === "new-session") {
      await this.newSession();
    } else if (message.type === "refresh-sessions") {
      await this.refreshSessions();
    } else if (message.type === "select-agent" && typeof message.label === "string") {
      await this.selectAgent(message.label);
    } else if (message.type === "refresh-team") {
      await this.refreshTeamForView();
    } else if (message.type === "refresh-model-settings") {
      await this.refreshModelSettings();
    } else if (message.type === "rekursiv.setup") {
      if (Array.isArray(message.skills) && message.skills.every((skill) => typeof skill === "string")) {
        await this.setupRekursiv(message.skills);
      }
    } else if (message.type === "rekursiv.disable") {
      await this.disableRekursiv();
    } else if (
      message.type === "model-settings"
      && isModelSettingsScope(message.scope)
      && isModelSettingsAction(message.action)
    ) {
      await this.applyModelSettings(message.scope, message.action);
    } else if (message.type === "open-change" && typeof message.changeId === "string") {
      await this.openChange(message.changeId);
    } else if (message.type === "preview-change" && typeof message.changeId === "string") {
      await this.previewChange(message.changeId);
    } else if (message.type === "open-file" && typeof message.path === "string") {
      await this.openFile(message.path);
    } else if (message.type === "review" && typeof message.path === "string") {
      await this.openDiff(message.path);
    }
  }

  private async start(force = false, sessionPath?: string): Promise<void> {
    if (this.startPromise && !force) {
      await this.startPromise;
      return;
    }
    if (!force && this.store.getState().connection !== "disconnected") {
      return;
    }
    this.startPromise = this.startProcess(sessionPath);
    try {
      await this.startPromise;
    } finally {
      this.startPromise = undefined;
    }
  }

  private async startProcess(sessionPath?: string): Promise<void> {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) {
      this.store.dispatch({ type: "connection.error", message: "Open a workspace folder before starting Sagent." });
      return;
    }
    this.store.dispatch({ type: "connection.started" });
    try {
      this.client.start(folder.uri.fsPath);
      const initResult = await this.client.initialize();
      const capabilities = isRecord(initResult.capabilities) ? initResult.capabilities : undefined;
      const skills = capabilities?.skills === true
        ? await this.client.getSkills()
        : [];

      const configuration = vscode.workspace.getConfiguration("sagent");
      const mode = configuration.get<string>("mode", "standard") === "rekursiv" ? "rekursiv" : "standard";
      const configuredSkillDirectories = configuration.get<string[]>("rekursivSkillDirectories", []);
      const trackinizerUrl = configuration.get<string>("trackinizerUrl", "") || DEFAULT_TRACKINIZER_URL;
      const isTrackinizerReachable = await isUrlReachable(trackinizerUrl);

      const trackinizerChecks = [
        {
          name: "Trackinizer server URL",
          status: isTrackinizerReachable ? "available" : "unavailable",
          detail: isTrackinizerReachable
            ? `Server is listening at ${trackinizerUrl}`
            : `Could not connect to ${trackinizerUrl}`,
        },
      ];

      const rekursivContext: SagentRekursivContext = {
        mode,
        orientation: mode === "rekursiv",
        skillDirectories: configuredSkillDirectories,
        skills,
        environment: {
          trackinizer: {
            status: isTrackinizerReachable ? "available" : "unavailable",
            url: trackinizerUrl,
            checks: trackinizerChecks,
          },
          wesearch: {
            status: "available",
            checks: [
              {
                name: "Wesearch native tools",
                status: "available",
                detail: "Wesearch is provided by Sagent's native research tools",
              },
            ],
          },
        },
      };

      this.store.dispatch({ type: "rekursiv.context.updated", context: rekursivContext });
      this.store.dispatch({
        type: "rekursiv.candidates.updated",
        candidates: discoverRekursivSkillCandidates(folder.uri.fsPath, this.extensionUri.fsPath),
      });
      await this.openSession(sessionPath);
    } catch (error) {
      if (error instanceof SagentExecutableMissingError && this.client.canInstallNativeSagent()) {
        const action = await vscode.window.showErrorMessage(
          error.message,
          "Install Sagent",
        );
        if (action === "Install Sagent") {
          try {
            await vscode.window.withProgress(
              {
                location: vscode.ProgressLocation.Notification,
                title: "Installing Sagent on this VS Code host…",
                cancellable: false,
              },
              () => this.client.installNativeSagent(folder.uri.fsPath),
            );
            void vscode.window.showInformationMessage(
              "Sagent was installed. Starting the native Sagent server…",
            );
            await this.startProcess(sessionPath);
          } catch (installError) {
            const message = errorMessage(installError);
            this.store.dispatch({ type: "connection.error", message });
            this.store.dispatch({ type: "sessions.failed", message });
          }
        } else {
          this.store.dispatch({ type: "connection.error", message: error.message });
          this.store.dispatch({ type: "sessions.failed", message: error.message });
        }
        return;
      }
      const message = errorMessage(error);
      this.store.dispatch({ type: "connection.error", message });
      this.store.dispatch({ type: "sessions.failed", message });
    }
  }

  private async setupRekursiv(skills: string[]): Promise<void> {
    if (["starting", "working", "cancelling"].includes(this.store.getState().connection)) {
      void vscode.window.showWarningMessage(
        "Wait for the current Sagent operation to finish before changing Rekursiv mode.",
      );
      return;
    }
    const candidates = discoverRekursivSkillCandidates(
      vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "",
      this.extensionUri.fsPath,
    );
    const selected = new Set(skills);
    if (skills.length === 0 || skills.some((id) => !candidates.some((candidate) => candidate.id === id))) {
      void vscode.window.showWarningMessage(
        "No valid native Rekursiv skill bundles were selected.",
      );
      return;
    }
    const configuration = vscode.workspace.getConfiguration("sagent");
    const trackinizerUrl = configuration.get<string>("trackinizerUrl", "") || DEFAULT_TRACKINIZER_URL;
    await configuration.update(
      "mode",
      "rekursiv",
      vscode.ConfigurationTarget.Workspace,
    );
    await configuration.update(
      "rekursivSkills",
      [...selected],
      vscode.ConfigurationTarget.Workspace,
    );
    await configuration.update(
      "trackinizerUrl",
      trackinizerUrl.trim() || DEFAULT_TRACKINIZER_URL,
      vscode.ConfigurationTarget.Workspace,
    );
    await this.restartSagent();
  }

  private async disableRekursiv(): Promise<void> {
    if (["starting", "working", "cancelling"].includes(this.store.getState().connection)) {
      void vscode.window.showWarningMessage(
        "Wait for the current Sagent operation to finish before changing Rekursiv mode.",
      );
      return;
    }
    await vscode.workspace.getConfiguration("sagent").update(
      "mode",
      "standard",
      vscode.ConfigurationTarget.Workspace,
    );
    await this.restartSagent();
  }

  private async restartSagent(): Promise<void> {
    const sessionPath = this.store.getState().sessionPath;
    this.client.dispose();
    await this.start(true, sessionPath);
  }

  public async selectSession(sessionPath?: string): Promise<void> {
    if (this.store.getState().connection === "working") {
      return;
    }
    this.store.dispatch(
      sessionPath
        ? { type: "session.selection.started", sessionPath }
        : { type: "session.creation.started" },
    );
    const request = ++this.sessionSelectionRequest;
    try {
      if (!this.client.isRunning()) {
        await this.start(true, sessionPath);
        return;
      }
      await this.openSession(sessionPath, request);
    } catch (error) {
      if (request === this.sessionSelectionRequest) {
        this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
        this.store.dispatch({ type: "sessions.failed", message: errorMessage(error) });
      }
    }
  }

  public async selectAgentFromTree(label: string): Promise<void> {
    await this.selectAgent(label);
  }

  public async newSession(): Promise<void> {
    await this.selectSession();
  }

  public async renameSession(item: SagentSessionItem): Promise<void> {
    await this.sessionTreeProvider.renameSession(item);
    await this.sessionTreeProvider.refresh();
  }

  /** Add the current public VS Code editor selection to the active composer. */
  public async addEditorSelectionToChat(uri?: vscode.Uri): Promise<void> {
    let editor = vscode.window.activeTextEditor;
    if (uri && (!editor || editor.document.uri.toString() !== uri.toString())) {
      editor = vscode.window.visibleTextEditors.find(
        (candidate) => candidate.document.uri.toString() === uri.toString(),
      );
    }
    if (!editor || editor.selection.isEmpty) {
      return;
    }
    const text = editor.document.getText(editor.selection);
    if (!text.trim()) {
      return;
    }

    await this.open();
    await this.start();
    const startLine = editor.selection.start.line + 1;
    const endLine = editor.selection.end.line + 1;
    const message: ExtensionToWebviewMessage = {
      type: "composer.insert",
      text,
      source: "editor",
      label: vscode.workspace.asRelativePath(editor.document.uri, false),
      languageId: editor.document.languageId,
      filePath: vscode.workspace.asRelativePath(editor.document.uri, false),
      startLine,
      endLine,
    };
    if (this.views.size === 0) {
      this.pendingComposerInsertions.push(message);
    } else {
      this.post(message);
    }
  }

  private async openSession(
    sessionPath?: string,
    request?: number,
  ): Promise<void> {
    const session = await this.client.openSession(sessionPath);
    const team = await this.client.getTeam();
    if (request !== undefined && request !== this.sessionSelectionRequest) {
      return;
    }
    const rootAgent = team.agents.find((agent) => agent.parent === undefined);
    this.store.dispatch({
      type: "session.opened",
      path: session.path,
      snapshot: session.snapshot,
      ...(rootAgent ? { agentLabel: rootAgent.label } : {}),
    });
    this.store.dispatch({ type: "team.updated", team });
    this.store.dispatch({
      type: "runtime.activity.updated",
      state: setTeamDiagnostics(this.store.getState().runtimeActivity, team.agents),
    });
    this.updateChangeReview(session.snapshot, rootAgent?.label ?? this.store.getState().selectedAgentLabel);
    await this.refreshCommands(rootAgent?.label ?? this.store.getState().selectedAgentLabel);
    await this.refreshModelSettings().catch((error) => {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    });
    await this.loadSessionsForTree().catch(() => undefined);
    await this.sessionTreeProvider.refresh();
  }

  private async send(text: string): Promise<void> {
    if (!text.trim() || this.store.getState().connection !== "ready") {
      return;
    }
    this.store.dispatch({ type: "user.submitted", text });
    try {
      await this.client.sendMessage(text, this.store.getState().selectedAgentLabel);
      await this.refreshTeam();
      await this.sessionTreeProvider.refresh().catch(() => undefined);
    } catch (error) {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    }
  }

  public async refreshSessions(): Promise<void> {
    if (!this.client.isRunning()) {
      await this.start(true);
      return;
    }
    await this.loadSessionsForTree().catch(() => undefined);
  }

  private async selectAgent(label: string): Promise<void> {
    const state = this.store.getState();
    if (
      label === state.selectedAgentLabel
      || !state.teamAgents.some((agent) => agent.label === label)
    ) {
      return;
    }
    const request = ++this.agentSelectionRequest;
    try {
      const session = await this.client.getAgentSession(label);
      if (request !== this.agentSelectionRequest) {
        return;
      }
      this.store.dispatch({
        type: "agent.session.opened",
        label: session.label,
        path: session.path,
        snapshot: session.snapshot,
      });
      this.updateChangeReview(session.snapshot, session.label);
      await this.refreshCommands(session.label);
      await this.refreshModelSettings(session.label);
    } catch (error) {
      if (request === this.agentSelectionRequest) {
        this.store.dispatch({
          type: "request.failed",
          message: "Could not load agent " + label + ": " + errorMessage(error),
        });
      }
    }
  }

  private async refreshTeam(): Promise<void> {
    if (this.teamRefreshPromise) {
      await this.teamRefreshPromise;
      return;
    }
    this.teamRefreshPromise = this.client
      .getTeam()
      .then((team) => {
        this.store.dispatch({ type: "team.updated", team });
        this.store.dispatch({
          type: "runtime.activity.updated",
          state: setTeamDiagnostics(
            this.store.getState().runtimeActivity,
            team.agents,
          ),
        });
      })
      .catch(() => undefined)
      .finally(() => {
        this.teamRefreshPromise = undefined;
      });
    await this.teamRefreshPromise;
  }

  private async refreshCommands(agentLabel = this.store.getState().selectedAgentLabel): Promise<void> {
    if (!this.client.isRunning()) {
      return;
    }
    try {
      const commandList = await this.client.getCommands(agentLabel);
      this.store.dispatch({ type: "commands.updated", descriptors: commandList.commands });
    } catch (error) {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    }
  }

  private async executeCommand(line: string): Promise<void> {
    if (!line.trim() || this.store.getState().connection === "disconnected") {
      return;
    }
    try {
      await this.client.executeCommand(this.store.getState().selectedAgentLabel, line);
      await this.refreshTeam();
      await this.refreshCommands();
    } catch (error) {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    }
  }

  public async refreshTeamForView(): Promise<void> {
    if (!this.client.isRunning()) {
      await this.start(true);
      return;
    }
    await this.refreshTeam();
  }

  private async cancel(): Promise<void> {
    const connection = this.store.getState().connection;
    if (connection !== "working" && connection !== "cancelling") {
      return;
    }
    this.store.dispatch({ type: "turn.cancelling" });
    try {
      await this.client.cancelTurn();
    } catch (error) {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    }
  }

  private handleSagentEvent(event: SagentEvent): void {
    const action = toConversationAction(event);
    if (action) {
      this.store.dispatch(action);
      if (action.type === "tool.completed" && action.path) {
        this.registerLiveChangeTarget(action);
      }
    }
    const runtimeEvent = toRuntimeActivityEvent(event);
    if (runtimeEvent) {
      this.store.dispatch({
        type: "runtime.activity.updated",
        state: applyRuntimeActivityEvent(
          this.store.getState().runtimeActivity,
          runtimeEvent,
        ),
      });
    }
    if (event.type === "agent.event" || event.type === "agent.completed") {
      void this.refreshTeam();
    }
  }

  private post(message: Record<string, unknown>): void {
    for (const view of this.views) {
      void view.webview.postMessage(message);
    }
  }

  private async openDiff(filePath: string): Promise<void> {
    await this.changeReviewService.openFile(filePath);
  }

  private async openFile(filePath: string): Promise<void> {
    await vscode.commands.executeCommand("vscode.open", vscode.Uri.file(filePath));
  }

  private async openChange(changeId: string): Promise<void> {
    const target = this.changeReviewTargets.get(changeId);
    if (target) {
      await this.changeReviewService.open(target);
    }
  }

  private async previewChange(changeId: string): Promise<void> {
    const target = this.changeReviewTargets.get(changeId);
    if (!target) {
      return;
    }
    const preview = await this.changeReviewService.createPreview(target);
    this.post({ type: "change-preview", preview });
  }

  private async loadSessionsForTree(): Promise<import("../shared/sagentProtocol").SagentSessionList> {
    try {
      const listing = this.sessionTreeProvider.withDisplayTitles(
        await this.client.listSessions(),
      );
      this.store.dispatch({ type: "sessions.updated", listing });
      return listing;
    } catch (error) {
      this.store.dispatch({ type: "sessions.failed", message: errorMessage(error) });
      throw error;
    }
  }

  private updateChangeReview(
    snapshot: import("../shared/sagentProtocol").SagentSessionSnapshot,
    agentLabel: string,
  ): void {
    this.changeReviewTargets = new Map();
    const targets = snapshot.items.flatMap((item) => {
      const target = toChangeReviewTarget(item);
      if (!target) {
        return [];
      }
      this.changeReviewTargets.set(target.recordId, target);
      return [toChangeReviewPresentationTarget(target, agentLabel)];
    });
    this.store.dispatch({ type: "changes.updated", targets });
    this.store.dispatch({
      type: "runtime.activity.updated",
      state: setChangeReviewTargets(this.store.getState().runtimeActivity, targets),
    });
  }

  /**
   * Live tool events arrive before the next session snapshot is requested.
   * Register the same stable tool-call identity used by snapshot projection so
   * the chat change list can request a preview immediately after a write.
   */
  private registerLiveChangeTarget(
    action: Extract<ConversationAction, { type: "tool.completed" }>,
  ): void {
    const filePath = action.path;
    if (!filePath || !action.fileChange) {
      return;
    }
    const turn = this.store.getState().turns.find((candidate) =>
      candidate.tools.some((tool) => tool.callId === action.callId),
    );
    const tool = turn?.tools.find((candidate) => candidate.callId === action.callId);
    if (!turn || !tool || (tool.name !== "Edit" && tool.name !== "Write")) {
      return;
    }
    const recordId = `tool-${action.callId}`;
    const target: SagentChangeReviewTarget = {
      filePath,
      uri: vscode.Uri.file(filePath),
      toolName: tool.name,
      turnId: turn.id,
      recordId,
      fileChange: action.fileChange,
    };
    this.changeReviewTargets.set(recordId, target);
    const targets = [...this.changeReviewTargets.values()].map((candidate) =>
      toChangeReviewPresentationTarget(
        candidate,
        this.store.getState().selectedAgentLabel,
      ),
    );
    this.store.dispatch({ type: "changes.updated", targets });
  }

  private async refreshModelSettings(
    agentLabel = this.store.getState().selectedAgentLabel,
  ): Promise<void> {
    if (!this.client.isRunning()) {
      return;
    }
    const snapshot = await this.client.getModelSettings(agentLabel);
    this.store.dispatch({ type: "model.settings.updated", snapshot });
  }

  private async applyModelSettings(
    scope: ModelSettingsScope,
    action: ModelSettingsAction,
  ): Promise<void> {
    try {
      const snapshot = await this.client.setModelSettings(
        this.store.getState().selectedAgentLabel,
        scope,
        action,
      );
      this.store.dispatch({ type: "model.settings.updated", snapshot });
      await this.refreshTeam();
    } catch (error) {
      this.store.dispatch({ type: "request.failed", message: errorMessage(error) });
    }
  }
}

function toConversationAction(event: SagentEvent): ConversationAction | undefined {
  switch (event.type) {
    case "turn.started":
      return {
        type: "turn.started",
        text: stringValue(event.text),
        ...(typeof event.turnId === "string" ? { turnId: event.turnId } : {}),
        ...(typeof event.timestamp === "string" ? { timestamp: event.timestamp } : {}),
        ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}),
        ...(modelIdentity(event) ? { modelIdentity: modelIdentity(event) } : {}),
      };
    case "command.output":
      return {
        type: "command.output",
        output: {
          id: `command-${nextCommandOutputId++}`,
          agentLabel: agentLabel(event) ?? "Agent",
          text: stringValue(event.text),
          tone: commandOutputTone(event.tone),
          ...(typeof event.command === "string" && event.command
            ? { command: event.command }
            : {}),
        },
      };
    case "message.created": {
      const kind = event.kind;
      if (kind !== "user" && kind !== "agent") {
        return undefined;
      }
      return {
        type: "message.created",
        id: stringValue(event.id),
        kind,
        text: stringValue(event.text),
        ...(typeof event.turnId === "string" ? { turnId: event.turnId } : {}),
        ...(typeof event.sender === "string" ? { sender: event.sender } : {}),
        ...(typeof event.recipient === "string" ? { recipient: event.recipient } : {}),
        ...(typeof event.timestamp === "string" ? { timestamp: event.timestamp } : {}),
        ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}),
      };
    }
    case "assistant.delta":
      return { type: "assistant.delta", text: stringValue(event.text), ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "assistant.completed":
      return {
        type: "assistant.completed",
        text: stringValue(event.text),
        ...(typeof event.record === "string" ? { record: event.record } : {}),
        ...(typeof event.timestamp === "string" ? { timestamp: event.timestamp } : {}),
        ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}),
      };
    case "assistant.thinking":
      return { type: "assistant.thinking", text: stringValue(event.text), ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "tool.started":
      return {
        type: "tool.started",
        callId: stringValue(event.callId),
        name: stringValue(event.name),
        summary: stringValue(event.summary),
        ...(isRecord(event.arguments) ? { arguments: event.arguments } : {}),
        ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}),
      };
    case "tool.completed": {
      const fileChange = parseSagentFileChange(event.fileChange);
      return {
        type: "tool.completed",
        callId: stringValue(event.callId),
        text: stringValue(event.text),
        isError: event.isError === true,
        state: toolActivityState(event.status, event.isError === true),
        ...(typeof event.path === "string" ? { path: event.path } : {}),
        ...(fileChange ? { fileChange } : {}),
        ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}),
      };
    }
    case "status.changed":
      return { type: "status.changed", text: stringValue(event.text), ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "agent.event": {
      const records = agentActivityRecords(event.records);
      return {
        type: "agent.event",
        label: stringValue(event.label),
        event: stringValue(event.event),
        ...(typeof event.text === "string" ? { text: event.text } : {}),
        ...(records ? { records } : {}),
      };
    }
    case "agent.completed":
      return {
        type: "agent.completed",
        label: stringValue(event.label),
        ...(numberValue(event.elapsed) !== undefined
          ? { elapsed: numberValue(event.elapsed) }
          : {}),
        ...(numberValue(event.tokens) !== undefined
          ? { tokens: numberValue(event.tokens) }
          : {}),
        ...(numberValue(event.cost) !== undefined
          ? { cost: numberValue(event.cost) }
          : {}),
      };
    case "retry.waiting":
      return { type: "status.changed", text: "Waiting to retry…" };
    case "compaction.started":
      return { type: "status.changed", text: "Compacting session…" };
    case "compaction.completed":
      return { type: "status.changed", text: "Session compacted" };
    case "compaction.failed":
      return { type: "status.changed", text: "Session compaction failed" };
    case "turn.completed":
      return { type: "turn.completed", text: stringValue(event.text), ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "turn.cancelled":
      return { type: "turn.cancelled", ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "turn.failed":
      return { type: "turn.failed", message: stringValue(event.message), ...(agentLabel(event) ? { agentLabel: agentLabel(event) } : {}) };
    case "connection.error":
      return { type: "connection.error", message: stringValue(event.message) };
    default:
      return undefined;
  }
}

function agentActivityRecords(value: unknown): AgentActivityRecord[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const records = value.flatMap((entry) => {
    if (!isRecord(entry) || (entry.kind !== "assistant" && entry.kind !== "tool")) {
      return [];
    }
    const record: AgentActivityRecord = { kind: entry.kind };
    if (typeof entry.callId === "string") {
      record.callId = entry.callId;
    }
    if (typeof entry.name === "string") {
      record.name = entry.name;
    }
    if (typeof entry.text === "string") {
      record.text = entry.text;
    }
    return [record];
  });
  return records.length > 0 ? records : undefined;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function commandOutputTone(value: unknown): "line" | "note" | "error" {
  return value === "line" || value === "error" ? value : "note";
}

function agentLabel(event: SagentEvent): string | undefined {
  return typeof event.agentLabel === "string" ? event.agentLabel : undefined;
}

function modelIdentity(event: SagentEvent): { provider: string; model: string } | undefined {
  return typeof event.modelProvider === "string" && typeof event.modelId === "string"
    ? { provider: event.modelProvider, model: event.modelId }
    : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toolActivityState(
  value: unknown,
  isError: boolean,
): SagentToolStatus {
  switch (value) {
    case "running":
    case "completed":
    case "failed":
    case "detached":
    case "cancelled":
      return value;
    default:
      return isError ? "failed" : "completed";
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function discoverRekursivSkillCandidates(
  cwd: string,
  extensionPath?: string,
): SagentRekursivSkillCandidate[] {
  const roots = new Set<string>();
  const seenIds = new Set<string>();
  const discoveredCandidates: SagentRekursivSkillCandidate[] = [];
  const candidateDefinitions = [
    {
      id: "trackinizer",
      label: "Trackinizer / Trax",
      description: "Durable work, evidence, sessions, issues, and experiments.",
      relativePaths: [
        ["trackinizer", "trackinizer", "trax", "docs", "skills"],
        ["trackinizer", "trax", "docs", "skills"],
      ],
    },
    {
      id: "priml",
      label: "Priml",
      description: "Reusable experiments and evaluations.",
      relativePaths: [["priml", "docs"]],
    },
    {
      id: "configgle",
      label: "Configgle",
      description: "Reproducible experiment configuration.",
      relativePaths: [["configgle", "docs"]],
    },
  ];
  const searchRoots: string[] = [];
  if (process.env.REKURSIV_HOME) {
    searchRoots.push(resolve(process.env.REKURSIV_HOME));
  }
  if (cwd) {
    let current = resolve(cwd);
    while (true) {
      searchRoots.push(current);
      const parent = dirname(current);
      if (parent === current) {
        break;
      }
      current = parent;
    }
  }
  for (const root of searchRoots) {
    for (const definition of candidateDefinitions) {
      if (seenIds.has(definition.id)) {
        continue;
      }
      for (const relativePath of definition.relativePaths) {
        const path = join(root, ...relativePath);
        if (!existsSync(path)) {
          continue;
        }
        const normalized = resolve(path);
        if (roots.has(normalized)) {
          continue;
        }
        roots.add(normalized);
        seenIds.add(definition.id);
        discoveredCandidates.push({
          id: definition.id,
          label: definition.label,
          description: definition.description,
          path: normalized,
        });
        break;
      }
    }
  }
  if (extensionPath) {
    const bundledRoot = join(extensionPath, "resources", "rekursiv-skills");
    for (const definition of candidateDefinitions) {
      if (seenIds.has(definition.id)) {
        continue;
      }
      const path = resolve(join(bundledRoot, definition.id));
      if (!existsSync(path)) {
        continue;
      }
      roots.add(path);
      seenIds.add(definition.id);
      discoveredCandidates.push({
        id: definition.id,
        label: definition.label,
        description: definition.description,
        path,
      });
    }
  }
  if (!seenIds.has("wesearch")) {
    discoveredCandidates.push({
      id: "wesearch",
      label: "Wesearch",
      description: "Web and paper research through Sagent's native tools.",
      path: "",
    });
  }
  return discoveredCandidates;
}

function isMessage(value: unknown): value is {
  type: string;
  text?: unknown;
  line?: unknown;
  id?: unknown;
  path?: unknown;
  skills?: unknown;
  label?: unknown;
  changeId?: unknown;
  scope?: unknown;
  action?: unknown;
} {
  return typeof value === "object" && value !== null && "type" in value && typeof value.type === "string";
}

function isModelSettingsScope(value: unknown): value is ModelSettingsScope {
  return value === "current-agent" || value === "new-session-default";
}

function isModelSettingsAction(value: unknown): value is ModelSettingsAction {
  if (!isRecord(value) || typeof value.type !== "string") {
    return false;
  }
  switch (value.type) {
    case "refresh":
      return true;
    case "select-scope":
      return isModelSettingsScope(value.scope);
    case "select-provider":
    case "select-account":
    case "select-model":
      return isModelSettingsScope(value.scope) && typeof value[value.type === "select-provider"
        ? "provider"
        : value.type === "select-account" ? "account" : "model"] === "string";
    case "set-setting":
      return isModelSettingsScope(value.scope)
        && typeof value.name === "string"
        && [
          "context",
          "thinking.effort",
          "thinking.budget",
          "thinking.output",
          "service_tier",
          "cache_ttl_sec",
          "manage_context_server_side",
        ].includes(value.name)
        && (typeof value.value === "string"
          || typeof value.value === "number"
          || typeof value.value === "boolean");
    default:
      return false;
  }
}
