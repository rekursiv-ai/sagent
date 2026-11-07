import type {
  SagentSessionInfo,
  SagentSessionList,
} from "../sagentProtocol";

/**
 * The state needed by the session navigator. SagentSessionInfo is intentionally
 * reused here so the UI cannot silently discard Sagent's native metadata.
 */
export type SessionNavigationState = {
  status: "loading" | "ready" | "error";
  sessions: readonly SagentSessionInfo[];
  activePath?: string;
  pendingAction?: SessionNavigationPendingAction;
  errorMessage?: string;
};

export type SessionNavigationPendingAction =
  | { type: "select"; sessionPath: string }
  | { type: "create" };

/**
 * The parent owns transport and application state. The navigator only emits
 * intent, allowing the existing webview message bridge to remain authoritative
 * for Sagent session selection and creation.
 */
export type SessionNavigationIntegrationContract = {
  state: SessionNavigationState;
  onSelectSession: (sessionPath: string) => void;
  onCreateSession: () => void;
  onRetry: () => void;
};

export function createInitialSessionNavigationState(): SessionNavigationState {
  return {
    status: "loading",
    sessions: [],
  };
}

export function sessionNavigationStateFromList(
  listing: SagentSessionList,
): SessionNavigationState {
  return {
    status: "ready",
    sessions: listing.sessions,
    activePath: listing.activePath,
  };
}

/**
 * Refreshing keeps the last successful list visible. This avoids a jarring
 * empty state while Sagent is answering a refresh request and never invents a
 * second cache of session records.
 */
export function sessionNavigationLoadingState(
  state: SessionNavigationState,
): SessionNavigationState {
  return {
    ...state,
    status: "loading",
    errorMessage: undefined,
  };
}

export function sessionNavigationErrorState(
  state: SessionNavigationState,
  errorMessage: string,
): SessionNavigationState {
  return {
    ...state,
    status: "error",
    pendingAction: undefined,
    errorMessage,
  };
}

export function sessionNavigationSelectionState(
  state: SessionNavigationState,
  sessionPath: string,
): SessionNavigationState {
  return {
    ...state,
    pendingAction: { type: "select", sessionPath },
    errorMessage: undefined,
  };
}

export function sessionNavigationCreationState(
  state: SessionNavigationState,
): SessionNavigationState {
  return {
    ...state,
    pendingAction: { type: "create" },
    errorMessage: undefined,
  };
}
