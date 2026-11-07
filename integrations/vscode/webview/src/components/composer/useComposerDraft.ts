import { useEffect, useState } from "react";

export type ComposerDraftOptions = {
  agentLabel: string;
  defaultDraft?: string;
  draft?: string;
  draftStorageKey?: string;
  sessionKey?: string;
};

export type ComposerDraftState = {
  draft: string;
  isControlled: boolean;
  setDraft: (draft: string) => void;
};

type UncontrolledDraft = {
  storageKey: string;
  value: string;
};

const storagePrefix = "sagent.composer.draft.v1";

/**
 * Keeps an in-progress prompt with the presentation layer rather than
 * sending it through Sagent. Session storage is scoped to the current
 * webview lifetime, which protects a prompt from view reloads without making
 * drafts part of Sagent's authoritative journal.
 */
export function useComposerDraft({
  agentLabel,
  defaultDraft = "",
  draft,
  draftStorageKey,
  sessionKey,
}: ComposerDraftOptions): ComposerDraftState {
  const isControlled = draft !== undefined;
  const storageKey = draftStorageKey ?? createStorageKey(sessionKey, agentLabel);
  const [uncontrolledDraft, setUncontrolledDraft] = useState<UncontrolledDraft>(() => ({
    storageKey,
    value: readDraft(storageKey) ?? defaultDraft,
  }));

  useEffect(() => {
    if (isControlled || uncontrolledDraft.storageKey === storageKey) {
      return;
    }

    // A changed session or agent gets its own draft. Do not carry text across
    // targets, and do not write the previous target's value into this key.
    setUncontrolledDraft({
      storageKey,
      value: readDraft(storageKey) ?? defaultDraft,
    });
  }, [defaultDraft, isControlled, storageKey, uncontrolledDraft.storageKey]);

  useEffect(() => {
    if (isControlled || uncontrolledDraft.storageKey !== storageKey) {
      return;
    }
    writeDraft(storageKey, uncontrolledDraft.value);
  }, [isControlled, storageKey, uncontrolledDraft]);

  const setDraft = (nextDraft: string): void => {
    if (!isControlled) {
      setUncontrolledDraft({ storageKey, value: nextDraft });
    }
  };

  return {
    draft: isControlled
      ? draft ?? ""
      : uncontrolledDraft.storageKey === storageKey
        ? uncontrolledDraft.value
        : "",
    isControlled,
    setDraft,
  };
}

function createStorageKey(sessionKey: string | undefined, agentLabel: string): string {
  return [storagePrefix, sessionKey ?? "current", agentLabel].join(":");
}

function readDraft(storageKey: string): string | undefined {
  try {
    return typeof window === "undefined"
      ? undefined
      : window.sessionStorage.getItem(storageKey) ?? undefined;
  } catch {
    // Storage can be unavailable in a restricted webview. In-memory drafts
    // still work; persistence is an enhancement, not a transport dependency.
    return undefined;
  }
}

function writeDraft(storageKey: string, draft: string): void {
  try {
    if (typeof window === "undefined") {
      return;
    }
    if (draft.length === 0) {
      window.sessionStorage.removeItem(storageKey);
    } else {
      window.sessionStorage.setItem(storageKey, draft);
    }
  } catch {
    // See readDraft: an unavailable storage implementation must not prevent
    // the user from composing or sending a message.
  }
}
