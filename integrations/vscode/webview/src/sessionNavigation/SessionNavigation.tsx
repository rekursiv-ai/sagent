import type { ReactElement, ReactNode } from "react";

import type {
  SessionNavigationIntegrationContract,
  SessionNavigationPendingAction,
} from "../../../src/shared/sessionNavigation";
import type { SagentSessionInfo } from "../../../src/shared/sagentProtocol";
import { formatSessionAge } from "../../../src/shared/sessionNavigation";
import { Button } from "../components/ui/Button";
import { Icon } from "../components/ui/Icon";
import { joinClasses, sagentUiTokens } from "../ui";

export type SessionNavigationProps = SessionNavigationIntegrationContract;

/**
 * Reusable presentation for Sagent's native session index.
 *
 * The component receives already-typed Sagent records and emits parent intent.
 * It does not read journals, derive titles from text, or call VS Code APIs;
 * those responsibilities stay behind the existing extension/server boundary.
 */
export function SessionNavigation({
  state,
  onSelectSession,
  onCreateSession,
  onRetry,
}: SessionNavigationProps): ReactElement {
  const isBusy = state.pendingAction !== undefined;

  return (
    <section
      aria-label="Sagent sessions"
      aria-busy={state.status === "loading" || isBusy}
      className={joinClasses(
        "flex min-h-0 flex-col border-b",
        sagentUiTokens.border,
        sagentUiTokens.surface,
      )}
    >
      <header className="flex items-center gap-2 px-3 py-2">
        <div className="min-w-0 flex-1">
          <h2 className="m-0 truncate text-xs font-semibold uppercase tracking-wide text-sagent-muted-foreground">
            Sessions
          </h2>
          <p className="m-0 truncate text-xs text-sagent-muted-foreground">
            Past Sagent conversations
          </p>
        </div>
        <Button
          aria-label="Start a new Sagent session"
          disabled={isBusy}
          onClick={onCreateSession}
          title="New session"
          variant="ghost"
        >
          <Icon name="add" />
        </Button>
      </header>

      {state.status === "loading" && state.sessions.length === 0 && (
        <SessionNavigationMessage>
          <Icon className="animate-spin" name="loading" />
          Loading sessions…
        </SessionNavigationMessage>
      )}

      {state.status === "error" && (
        <div
          className="m-2 rounded border border-[var(--vscode-inputValidation-errorBorder)] bg-[var(--vscode-inputValidation-errorBackground)] p-2 text-xs"
          role="alert"
        >
          <p className="m-0 text-[var(--vscode-errorForeground)]">
            {state.errorMessage ?? "Could not load Sagent sessions."}
          </p>
          <Button className="mt-2" onClick={onRetry}>
            <Icon name="refresh" />
            Retry
          </Button>
        </div>
      )}

      {state.status !== "error" && state.sessions.length === 0 && (
        <SessionNavigationMessage>
          No past sessions in this workspace.
        </SessionNavigationMessage>
      )}

      {state.sessions.length > 0 && (
        <ul
          aria-label="Sagent session history"
          className="m-0 min-h-0 overflow-y-auto p-1"
          role="listbox"
        >
          {state.sessions.map((session) => (
            <SessionNavigationItem
              active={session.path === state.activePath}
              disabled={isBusy}
              key={session.path}
              onSelect={() => onSelectSession(session.path)}
              pending={isSelecting(state.pendingAction, session.path)}
              session={session}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function SessionNavigationMessage({
  children,
}: {
  children: ReactNode;
}): ReactElement {
  return (
    <p className="m-0 flex items-center gap-2 px-3 pb-3 text-xs text-sagent-muted-foreground">
      {children}
    </p>
  );
}

function SessionNavigationItem({
  active,
  disabled,
  onSelect,
  pending,
  session,
}: {
  active: boolean;
  disabled: boolean;
  onSelect: () => void;
  pending: boolean;
  session: SagentSessionInfo;
}): ReactElement {
  const title = session.title || "Untitled session";
  const metadata = [
    formatSessionAge(session.mtime),
    `${session.records} records`,
    session.model,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <li className="m-0 list-none">
      <button
        aria-label={`${title}${active ? " (current session)" : ""}`}
        aria-selected={active}
        className={joinClasses(
          "flex w-full min-w-0 items-start gap-2 rounded px-2 py-1.5 text-left text-sm",
          "text-sagent-foreground hover:bg-sagent-hover",
          sagentUiTokens.focus,
          active && "bg-sagent-hover",
        )}
        disabled={disabled}
        onClick={onSelect}
        role="option"
        title={`${title}\n${session.path}`}
        type="button"
      >
        <span className="mt-0.5 w-4 shrink-0 text-sagent-muted-foreground">
          {pending ? (
            <Icon className="animate-spin" name="loading" />
          ) : (
            <span aria-hidden="true">{active ? "●" : "○"}</span>
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium">{title}</span>
          <span className="block truncate text-xs text-sagent-muted-foreground">
            {metadata}
          </span>
        </span>
      </button>
    </li>
  );
}

function isSelecting(
  pendingAction: SessionNavigationPendingAction | undefined,
  sessionPath: string,
): boolean {
  return pendingAction?.type === "select" && pendingAction.sessionPath === sessionPath;
}
