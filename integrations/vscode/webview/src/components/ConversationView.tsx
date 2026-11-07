import { useEffect, useState, type ReactElement } from "react";

import type {
  AgentActivity,
  CommandOutputRecord,
  ConversationTurn,
  ConversationTurnItem,
  ConversationTimelineItem,
  ToolActivity,
} from "../../../src/shared/conversationTypes";
import type { ChangePreview } from "../../../src/shared/changePreviewModel";
import type { ChangeReviewModel } from "../../../src/shared/changeReviewModel";
import {
  formatRuntimeElapsed,
  type RuntimeActivityState,
} from "../../../src/shared/runtimeActivity/runtimeActivityModel";
import { AgentActivityRecordList } from "./conversation/AgentActivityRecordList";
import { CommandOutputEntry } from "./command/CommandOutputList";
import { MarkdownMessage } from "./conversation/MarkdownMessage";
import { DiffHoverPreview } from "./changeReview/DiffHoverPreview";
import { Button } from "./ui/Button";
import { Icon } from "./ui/Icon";

export type ConversationViewProps = {
  agents: AgentActivity[];
  changeReview: ChangeReviewModel;
  commandOutputs: readonly CommandOutputRecord[];
  runtimeActivity: RuntimeActivityState;
  onDeleteCommandOutput: (id: string) => void;
  onOpenFile: (path: string) => void;
  onReview: (path: string) => void;
  onRequestPreview: (changeId: string) => void;
  previews: Readonly<Record<string, ChangePreview>>;
  timeline: readonly ConversationTimelineItem[];
  turns: ConversationTurn[];
};

/**
 * Renders Sagent's presentation read model. Runtime ownership stays in the
 * extension host and Sagent; this component only composes visible activity.
 */
export function ConversationView({
  agents,
  changeReview,
  commandOutputs,
  onDeleteCommandOutput,
  runtimeActivity,
  onOpenFile,
  onReview,
  onRequestPreview,
  previews,
  timeline,
  turns,
}: ConversationViewProps): ReactElement {
  const turnsById = new Map(turns.map((turn) => [turn.id, turn]));
  const commandOutputsById = new Map(commandOutputs.map((output) => [output.id, output]));
  const hasRunningTurn = turns.some(
    (turn) => turn.state === "pending" || turn.state === "running",
  );
  const latestAssistantTurnId = [...turns]
    .reverse()
    .find((turn) => turn.state === "completed" && turn.assistantText)?.id;

  return (
    <>
      {agents.length > 0 && <AgentActivityPanel agents={agents} />}
      {timeline.map((item) => {
        if (item.type === "turn") {
          const turn = turnsById.get(item.id);
          if (!turn) {
            return null;
          }
          return (
            <TurnView
              changeReview={changeReview}
              runtimeActivity={runtimeActivity}
              key={`${item.type}-${item.id}`}
              onOpenFile={onOpenFile}
              onRequestPreview={onRequestPreview}
              onReview={onReview}
              previews={previews}
              persistentAssistantCopy={
                !hasRunningTurn && latestAssistantTurnId === turn.id
              }
              turn={turn}
            />
          );
        }
        const output = commandOutputsById.get(item.id);
        if (!output) {
          return null;
        }
        return (
          <section
            aria-label="Sagent command output"
            className="command-output-list"
            key={`${item.type}-${item.id}`}
          >
            <CommandOutputEntry
              onDelete={() => onDeleteCommandOutput(output.id)}
              output={output}
            />
          </section>
        );
      })}
    </>
  );
}

function TurnView({
  changeReview,
  runtimeActivity,
  onOpenFile,
  onRequestPreview,
  onReview,
  persistentAssistantCopy,
  previews,
  turn,
}: {
  changeReview: ChangeReviewModel;
  runtimeActivity: RuntimeActivityState;
  onOpenFile: (path: string) => void;
  onRequestPreview: (changeId: string) => void;
  onReview: (path: string) => void;
  persistentAssistantCopy: boolean;
  previews: Readonly<Record<string, ChangePreview>>;
  turn: ConversationTurn;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  const [copiedUser, setCopiedUser] = useState(false);

  async function copyUserInput(): Promise<void> {
    try {
      await copyText(turn.userText);
      setCopiedUser(true);
      window.setTimeout(() => setCopiedUser(false), 1_500);
    } catch {
      setCopiedUser(false);
    }
  }

  async function copyAssistantOutput(): Promise<void> {
    if (!turn.assistantText) {
      return;
    }
    try {
      await copyText(turn.assistantText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <article className="turn" data-turn-id={turn.id}>
      <div className="user-message-group copy-hover-target">
        <div className="user-message">
          <div className="message-text">{turn.userText}</div>
        </div>
        <div className="user-message-actions" aria-label="User input actions">
          <div className="message-actions">
            <Button
              aria-label={copiedUser ? "User input copied" : "Copy user input"}
              className="message-action-button copy-hover-action"
              onClick={() => void copyUserInput()}
              title={copiedUser ? "Copied" : "Copy user input"}
              variant="ghost"
            >
              <Icon name="copy" />
            </Button>
          </div>
        </div>
      </div>

      <div className="assistant-message" aria-label="Sagent response">
        <div className="assistant-message-header">
          <div
            className="message-label assistant-message-label"
            title={turn.modelIdentity?.provider}
          >
            {turn.modelIdentity?.model ?? "Assistant"}
          </div>
          <TurnRuntimeSummary runtimeActivity={runtimeActivity} turn={turn} />
        </div>
        <TurnActivityItems
          assistantText={turn.assistantText}
          assistantCopyButton={turn.state === "completed" && turn.assistantText ? (
            <div className="message-actions" aria-label="Agent output actions">
              <Button
                aria-label={copied ? "Agent output copied" : "Copy agent output"}
                className={`message-action-button copy-hover-action${persistentAssistantCopy ? " copy-action-persistent" : ""}`}
                onClick={() => void copyAssistantOutput()}
                title={copied ? "Copied" : "Copy agent output"}
                variant="ghost"
              >
                <Icon name="copy" />
              </Button>
            </div>
          ) : undefined}
          items={turn.items}
          onOpenFile={onOpenFile}
          onReview={onReview}
        />
      </div>

      {turn.thinkingText && (
        <details className="activity-panel">
          <summary>Reasoning</summary>
          <div className="activity-content">{turn.thinkingText}</div>
        </details>
      )}

      {turn.agentMessages.length > 0 && (
        <details className="activity-panel">
          <summary>{turn.agentMessages.length} agent messages</summary>
          <div className="activity-content">
            {turn.agentMessages.map((message) => (
              <div className="agent-message" key={message.id}>
                <div className="message-label">
                  {message.sender ?? "Agent"}
                  {message.recipient ? ` → ${message.recipient}` : ""}
                </div>
                <MarkdownMessage content={message.text} />
              </div>
            ))}
          </div>
        </details>
      )}

      {turn.tools.some(isCompletedFileChange) && (
        <details className="activity-panel">
          <summary>
            Changes ({turn.tools.filter(isCompletedFileChange).length})
          </summary>
          <div className="activity-content">
            <ul className="change-file-list">
              {turn.tools.filter(isCompletedFileChange).map((tool) => {
                const path = tool.path!;
                const changeId = findChangeId(changeReview, tool.callId);
                return (
                  <li key={tool.callId}>
                    <DiffHoverPreview
                      onRequest={() => {
                        if (changeId) {
                          onRequestPreview(changeId);
                        }
                      }}
                      preview={changeId ? previews[changeId] : undefined}
                    >
                      <Button
                        aria-label={`Review changes for ${path}`}
                        className="review-link"
                        onClick={() => onReview(path)}
                        variant="ghost"
                      >
                        <Icon name="diff" />
                        <span>{path}</span>
                      </Button>
                    </DiffHoverPreview>
                  </li>
                );
              })}
            </ul>
          </div>
        </details>
      )}

      {turn.error && <div className="error-message">{turn.error}</div>}
    </article>
  );
}

function TurnActivityItems({
  assistantCopyButton,
  assistantText,
  items,
  onOpenFile,
  onReview,
}: {
  assistantCopyButton?: ReactElement;
  assistantText: string;
  items: ConversationTurnItem[];
  onOpenFile: (path: string) => void;
  onReview: (path: string) => void;
}): ReactElement {
  if (items.length === 0) {
    return (
      <div className="assistant-output-copy-surface copy-hover-target">
        <MarkdownMessage content={assistantText} />
        {assistantCopyButton}
      </div>
    );
  }

  const rendered: ReactElement[] = [];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    if (item.type === "assistant") {
      const hasLaterAssistant = items.slice(index + 1).some(
        (laterItem) => laterItem.type === "assistant",
      );
      rendered.push(
        <div className="assistant-output-copy-surface copy-hover-target" key={item.id}>
          <MarkdownMessage content={item.text} />
          {!hasLaterAssistant && assistantCopyButton}
        </div>,
      );
      continue;
    }

    const tools: ToolActivity[] = [item.activity];
    while (index + 1 < items.length && items[index + 1]?.type === "tool") {
      index += 1;
      const next = items[index];
      if (next.type === "tool") {
        tools.push(next.activity);
      }
    }
    const running = tools.some((tool) => tool.state === "running");
    rendered.push(
      <details className="activity-panel tool-group" key={`tools-${tools[0]!.callId}`} open={running}>
        <summary>
          {running ? "Working with tools…" : `${tools.length} tool action${tools.length === 1 ? "" : "s"}`}
        </summary>
        <div className="activity-content">
          {tools.map((tool) => (
            <ToolActivityView
              key={tool.callId}
              activity={tool}
              onOpenFile={onOpenFile}
              onReview={onReview}
            />
          ))}
        </div>
      </details>,
    );
  }

  return <>{rendered}</>;
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Could not copy agent output");
  }
}

function TurnRuntimeSummary({
  runtimeActivity,
  turn,
}: {
  readonly runtimeActivity: RuntimeActivityState;
  readonly turn: ConversationTurn;
}): ReactElement | null {
  const activeTurn = runtimeActivity.activeTurn;
  const isActive =
    turn.state === "running"
    && activeTurn !== undefined
    && (activeTurn.turnId === undefined || activeTurn.turnId === turn.id);
  const startedAtMs = turn.startedAtMs ?? (isActive ? activeTurn?.startedAtMs : undefined);
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const timer = window.setInterval(() => setNowMs(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [isActive]);

  if (startedAtMs === undefined) {
    return null;
  }
  const seconds = isActive
    ? Math.max(0, (nowMs - startedAtMs) / 1_000)
    : turn.elapsedSeconds;
  if (seconds === undefined) {
    return null;
  }
  return (
    <div className="turn-runtime-summary" aria-live={isActive ? "polite" : undefined}>
      {isActive && <span className="turn-runtime-spinner" aria-hidden="true">◌</span>}
      <span>Worked for {formatRuntimeElapsed(seconds)}</span>
    </div>
  );
}

function findChangeId(model: ChangeReviewModel, callId: string): string | undefined {
  if (model.status !== "review") {
    return undefined;
  }
  for (const group of model.groups) {
    const file = group.files.find((candidate) => candidate.id === `tool-${callId}`);
    if (file) {
      return file.id;
    }
  }
  return undefined;
}

function isCompletedFileChange(tool: ConversationTurn["tools"][number]): boolean {
  return (
    tool.state === "completed"
    && (tool.name === "Edit" || tool.name === "Write")
    && tool.path !== undefined
  );
}

function AgentActivityPanel({ agents }: { agents: AgentActivity[] }): ReactElement {
  return (
    <section className="agents-panel" aria-label="Sagent agents">
      <div className="agents-heading">Agents</div>
      <div className="agents-list">
        {agents.map((agent) => (
          <div className="agent-entry" key={agent.label}>
            <div className="agent-row">
              <span className="agent-indicator" data-state={agent.state} aria-hidden="true" />
              <span className="agent-label">{agent.label}</span>
              <span className="agent-event">{agent.lastEvent}</span>
            </div>
            {agent.text && <div className="agent-detail">{agent.text}</div>}
            {agent.records && <AgentActivityRecordList records={agent.records} />}
          </div>
        ))}
      </div>
    </section>
  );
}

function ToolActivityView({
  activity,
  onOpenFile,
  onReview,
}: {
  activity: ToolActivity;
  onOpenFile: (path: string) => void;
  onReview: (path: string) => void;
}): ReactElement {
  const stateLabel = activity.state === "running" ? "Running" : activity.state;
  const invocation = toolInvocationLabel(activity);
  const [copied, setCopied] = useState(false);

  async function copyToolOutput(): Promise<void> {
    try {
      await copyText(activity.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <details className="tool-activity" open={activity.state === "running"}>
      <summary className="tool-summary-row">
        <span className="tool-invocation" title={invocation}>{invocation}</span>
        <span className="tool-state">{stateLabel}</span>
      </summary>
      <div className="tool-details">
        {activity.text !== undefined && (
          <div className="tool-result-shell copy-hover-target">
            <Button
              aria-label={copied ? "Tool output copied" : "Copy tool output"}
              className="tool-output-copy-button copy-hover-action"
              onClick={() => void copyToolOutput()}
              title={copied ? "Copied" : "Copy tool output"}
              variant="ghost"
            >
              <Icon name="copy" />
            </Button>
            <div className="tool-result">{activity.text}</div>
          </div>
        )}
        {activity.path && (
          <div className="tool-file-actions">
            <Button
              aria-label={`Open ${activity.path}`}
              className="review-link"
              onClick={() => onOpenFile(activity.path!)}
              title={`Open ${activity.path}`}
              variant="ghost"
            >
              <Icon name="file" />
              <span>{activity.path}</span>
            </Button>
            <Button
              aria-label={`Review changes for ${activity.path}`}
              className="review-link"
              onClick={() => onReview(activity.path!)}
              variant="ghost"
            >
              <Icon name="diff" />
              Review diff
            </Button>
          </div>
        )}
      </div>
    </details>
  );
}

function toolInvocationLabel(activity: ToolActivity): string {
  const argumentsValue = activity.arguments ?? {};
  if (activity.name === "Bash" && typeof argumentsValue.command === "string") {
    return argumentsValue.command;
  }
  if (activity.path) {
    return activity.path;
  }
  if (typeof argumentsValue.file_path === "string") {
    return argumentsValue.file_path;
  }
  if (activity.name === "WebSearch" && typeof argumentsValue.query === "string") {
    return argumentsValue.query;
  }
  if (activity.name === "WebFetch" && typeof argumentsValue.url === "string") {
    return argumentsValue.url;
  }
  return activity.summary || activity.name;
}
