import { useEffect, useRef, useState, type ReactElement } from "react";

import {
  createInitialConversationState,
  type ConversationState,
} from "../../src/shared/conversationTypes";
import { createChangeReviewModel } from "../../src/shared/changeReviewModel";
import type { ChangePreview } from "../../src/shared/changePreviewModel";
import { createInitialSessionNavigationState } from "../../src/shared/sessionNavigation";
import type {
  ExtensionToWebviewMessage,
  WebviewToExtensionMessage,
} from "../../src/shared/webviewProtocol";
import { Composer } from "./components/Composer";
import { ConversationView } from "./components/ConversationView";
import type { ChangeReviewAction } from "./components/changeReview/changeReviewContract";
import type { ModelSettingsAction } from "../../src/shared/modelSettings/modelSettingsContract";
import { ModelSettingsDialog } from "./components/modelSettings/ModelSettingsDialog";
import { RekursivDialog } from "./components/rekursiv/RekursivDialog";
import { Button } from "./components/ui/Button";
import { Icon } from "./components/ui/Icon";
import sagentLogoUrl from "./assets/sagent-logo.png";

type VsCodeApi = {
  getState(): ConversationState | undefined;
  setState(state: ConversationState): void;
  postMessage(message: WebviewToExtensionMessage): void;
};

declare function acquireVsCodeApi<T>(): T;

const vscode = acquireVsCodeApi<VsCodeApi>();

export function App(): ReactElement {
  const [state, setState] = useState<ConversationState>(
    () => normalizeConversationState(vscode.getState()),
  );
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [modelSettingsAnchor, setModelSettingsAnchor] = useState<HTMLButtonElement | null>(null);
  const [rekursivSettingsOpen, setRekursivSettingsOpen] = useState(false);
  const [rekursivSettingsAnchor, setRekursivSettingsAnchor] = useState<HTMLButtonElement | null>(null);
  const [changePreviews, setChangePreviews] = useState<Record<string, ChangePreview>>({});
  const conversationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }, [state.turns, state.timeline]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent<ExtensionToWebviewMessage>) => {
      if (event.data.type === "state") {
        setState(event.data.state);
      } else if (event.data.type === "change-preview") {
        const preview = event.data.preview;
        setChangePreviews((previews) => ({
          ...previews,
          [preview.changeId]: preview,
        }));
      }
    };

    window.addEventListener("message", handleMessage);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  useEffect(() => {
    vscode.setState(state);
  }, [state]);

  const isBusy = ["starting", "working", "cancelling"].includes(state.connection);
  const selectedAgent = state.teamAgents.find(
    (agent) => agent.label === state.selectedAgentLabel,
  );
  const activeSession = state.sessionNavigation.sessions.find(
    (session) => session.path === state.sessionPath,
  );
  const canSendToSelectedAgent = selectedAgent !== undefined;
  const inputDisabled =
    ["starting", "disconnected", "error"].includes(state.connection)
    || !canSendToSelectedAgent;
  return (
    <div className="sagent-app">
      {(state.connection === "error" || state.connection === "disconnected") && (
        <header className="app-header">
          <Button onClick={() => vscode.postMessage({ type: "retry" })}>
            <Icon name="refresh" />
            Reconnect
          </Button>
        </header>
      )}

      <div className="workspace-content">
        {/*
         * Deferred embedded navigation surfaces:
         *
         * SessionNavigation and SwarmView remain implemented and tested for a
         * future embedded-workspace design, but they are intentionally unused
         * here. The native VS Code Sessions and Swarm TreeDataProvider views
         * are the current canonical navigation surfaces, so mounting these
         * React copies would duplicate them in Chat.
         *
         * <SessionNavigation
         *   state={state.sessionNavigation}
         *   onCreateSession={() => vscode.postMessage({ type: "new-session" })}
         *   onRetry={() => vscode.postMessage({ type: "refresh-sessions" })}
         *   onSelectSession={(path) =>
         *     vscode.postMessage({ type: "select-session", path })
         *   }
         * />
         * <SwarmView
         *   state={createSwarmViewState(state)}
         *   onAction={(action) => handleSwarmAction(action, vscode)}
         * />
        */}
        {/*
         * Intentionally unused frontend surfaces:
         *
         * RuntimeActivityPanel and ChangeReviewPanel remain implemented for a
         * future workspace layout, but they are not mounted here because the
         * conversation already presents tool calls and file changes. Mounting
         * them would duplicate information in the chat. Their state and host
         * wiring remain intact so the surfaces can be restored later without
         * recreating Sagent data or runtime behavior.
         */}

        <main className="conversation" ref={conversationRef} aria-live="polite">
          <div className="conversation-content">
            {state.timeline.length === 0 && state.connection !== "error" ? (
              <Welcome />
            ) : (
              <ConversationView
                agents={state.agents}
                changeReview={state.changeReview}
                commandOutputs={state.commandOutputs}
                onDeleteCommandOutput={(id) =>
                  vscode.postMessage({ type: "remove-command-output", id })
                }
                runtimeActivity={state.runtimeActivity}
                onOpenFile={(path) => vscode.postMessage({ type: "open-file", path })}
                onRequestPreview={(changeId) =>
                  vscode.postMessage({ type: "preview-change", changeId })
                }
                onReview={(path) => vscode.postMessage({ type: "review", path })}
                previews={changePreviews}
                timeline={state.timeline}
                turns={state.turns}
              />
            )}
            {state.lastError && <div className="error-message">{state.lastError}</div>}
          </div>
        </main>
      </div>

      <Composer
        agentLabel={state.selectedAgentLabel}
        busy={isBusy}
        cancelling={state.connection === "cancelling"}
        disabled={inputDisabled}
        error={state.connection === "error" ? state.lastError : undefined}
        sessionKey={state.sessionPath}
        sessionLabel={activeSession?.title ?? "Current session"}
        onCancel={() => vscode.postMessage({ type: "cancel" })}
        onOpenModelSettings={(anchor) => {
          setModelSettingsAnchor(anchor);
          setModelSettingsOpen(true);
        }}
        modelLabel={modelButtonLabel(state)}
        modelDiagnostics={modelDiagnostics(state)}
        modelSettingsAvailable={state.modelSettings !== undefined}
        onOpenRekursivSettings={(anchor) => {
          setRekursivSettingsAnchor(anchor);
          setRekursivSettingsOpen(true);
        }}
        rekursivLabel={state.rekursivContext?.mode === "rekursiv" ? "Rekursiv" : "Standard"}
        commandDescriptors={state.commandDescriptors}
        onSubmit={(text) => {
          if (text.trimStart().startsWith("/")) {
            vscode.postMessage({ type: "command", line: text });
          } else {
            vscode.postMessage({ type: "send", text });
          }
        }}
      />

      {state.modelSettings && (
        <ModelSettingsDialog
          open={modelSettingsOpen}
          anchor={modelSettingsAnchor}
          snapshot={state.modelSettings}
          onAction={(action) => handleModelSettingsAction(action, vscode)}
          onClose={() => setModelSettingsOpen(false)}
        />
      )}
      <RekursivDialog
        anchor={rekursivSettingsAnchor}
        candidates={state.rekursivCandidates}
        context={state.rekursivContext}
        open={rekursivSettingsOpen}
        onClose={() => setRekursivSettingsOpen(false)}
        onDisable={() => {
          setRekursivSettingsOpen(false);
          vscode.postMessage({ type: "rekursiv.disable" });
        }}
        onSetup={(skills) => {
          setRekursivSettingsOpen(false);
          vscode.postMessage({ type: "rekursiv.setup", skills });
        }}
      />
    </div>
  );
}

function Welcome(): ReactElement {
  return (
    <div className="welcome">
      <img className="welcome-logo" src={sagentLogoUrl} alt="Sagent" />
      <h1>A harness for agent swarms.</h1>
      <p>Describe the problem. Let agents investigate, implement, and review the solution across your workspace.</p>
    </div>
  );
}

function normalizeConversationState(
  persistedState: ConversationState | undefined,
): ConversationState {
  if (!persistedState) {
    return createInitialConversationState();
  }
  return {
    ...persistedState,
    sessionNavigation:
      persistedState.sessionNavigation ?? createInitialSessionNavigationState(),
    teamOpen: persistedState.teamOpen ?? persistedState.teamAgents.length > 0,
    teamAgents: persistedState.teamAgents ?? [],
    selectedAgentLabel: persistedState.selectedAgentLabel ?? "Agent",
    runtimeActivity: persistedState.runtimeActivity ?? createInitialConversationState().runtimeActivity,
    commandDescriptors: persistedState.commandDescriptors ?? [],
    commandOutputs: persistedState.commandOutputs ?? [],
    rekursivCandidates: persistedState.rekursivCandidates ?? [],
    timeline: persistedState.timeline ?? legacyTimeline(persistedState),
    changeReview:
      persistedState.changeReview ?? createChangeReviewModel([]),
    agents: persistedState.agents ?? [],
    turns: persistedState.turns.map((turn) => ({
      ...turn,
      agentMessages: turn.agentMessages ?? [],
      items: turn.items ?? [],
    })),
  };
}

function legacyTimeline(state: ConversationState): ConversationState["timeline"] {
  return [
    ...state.turns.map((turn) => ({ type: "turn" as const, id: turn.id })),
    ...state.commandOutputs.map((output) => ({ type: "command" as const, id: output.id })),
  ];
}

function handleModelSettingsAction(
  action: ModelSettingsAction,
  api: VsCodeApi,
): void {
  if (action.type === "refresh") {
    api.postMessage({ type: "refresh-model-settings" });
    return;
  }
  api.postMessage({
    type: "model-settings",
    scope: action.scope,
    action,
  });
}

function modelButtonLabel(state: ConversationState): string {
  const snapshot = state.modelSettings;
  if (!snapshot) {
    return "Model";
  }
  return `${snapshot.provider.currentProvider}/${snapshot.model.currentModel} (${snapshot.settings.thinkingEffort})`;
}

function modelDiagnostics(state: ConversationState): string | undefined {
  const diagnostics = state.modelSettings?.diagnostics;
  if (!diagnostics) {
    return undefined;
  }
  return [
    `Model: ${diagnostics.model}`,
    `Context: ${diagnostics.context}`,
    `Reasoning: ${diagnostics.thinking.effort}`,
    `Budget: ${diagnostics.thinking.budget}`,
    `Output: ${diagnostics.thinking.output}`,
    `Service tier: ${diagnostics.serviceTier}`,
    `Cache: ${diagnostics.cacheTtlSeconds}s`,
    `Limits: ${diagnostics.limits.maxRequestTokens} request / ${diagnostics.limits.maxResponseTokens} response tokens`,
  ].join("\n");
}

function handleChangeReviewAction(
  action: ChangeReviewAction,
  api: VsCodeApi,
): void {
  if (action.type === "open-diff") {
    api.postMessage({ type: "open-change", changeId: action.changeId });
  } else {
    api.postMessage({ type: "retry" });
  }
}
