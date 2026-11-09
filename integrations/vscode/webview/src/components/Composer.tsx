import {
  useId,
  useEffect,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
  type ReactElement,
} from "react";

import { Button } from "./ui/Button";
import { Icon } from "./ui/Icon";
import { joinClasses, sagentUiTokens } from "../ui";
import { useComposerDraft } from "./composer/useComposerDraft";
import { useAutosizeTextarea } from "./composer/useAutosizeTextarea";
import {
  commandCompletionContext,
  commandCompletionItems,
  type SagentCommandDescriptor,
} from "../../../src/shared/commandSurface/commandSurface";

export type ComposerProps = {
  agentLabel: string;
  busy: boolean;
  cancelling?: boolean;
  defaultDraft?: string;
  disabled: boolean;
  draft?: string;
  draftStorageKey?: string;
  error?: string;
  sessionKey?: string;
  sessionLabel?: string;
  onCancel: () => void;
  onOpenModelSettings?: (anchor: HTMLButtonElement) => void;
  onOpenRekursivSettings?: (anchor: HTMLButtonElement) => void;
  rekursivLabel?: string;
  modelLabel?: string;
  modelDiagnostics?: string;
  modelSettingsAvailable?: boolean;
  onDraftChange?: (draft: string) => void;
  onInputKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (text: string) => void;
  placeholder?: string;
  commandDescriptors?: readonly SagentCommandDescriptor[];
};

/**
 * Owns prompt-entry behavior only. The extension host still decides what a
 * submitted message means and routes it through Sagent's native Agent runtime.
 */
export function Composer({
  agentLabel,
  busy,
  cancelling = false,
  defaultDraft,
  disabled,
  draft: controlledDraft,
  draftStorageKey,
  error,
  sessionKey,
  sessionLabel = "Current session",
  onCancel,
  onOpenModelSettings,
  onOpenRekursivSettings,
  rekursivLabel = "Rekursiv",
  modelLabel = "Model",
  modelDiagnostics,
  modelSettingsAvailable = false,
  onDraftChange,
  onInputKeyDown,
  onSubmit,
  placeholder,
  commandDescriptors = [],
}: ComposerProps): ReactElement {
  const {
    draft,
    setDraft: setComposerDraft,
  } = useComposerDraft({
    agentLabel,
    defaultDraft,
    draft: controlledDraft,
    draftStorageKey,
    sessionKey,
  });
  const textareaRef = useAutosizeTextarea(draft, 220);
  const errorId = useId();
  const statusId = useId();
  const [activeCommandIndex, setActiveCommandIndex] = useState(0);

  const targetAgentLabel = agentLabel || "Sagent";
  const targetSessionLabel = sessionLabel || "Current session";
  const canSend = !disabled && !busy && draft.trim().length > 0;
  const commandContext = commandCompletionContext(draft);
  const commandItems = commandContext
    ? commandCompletionItems(commandDescriptors, commandContext)
    : [];
  const showCommandMenu = commandContext !== undefined && commandItems.length > 0;

  useEffect(() => setActiveCommandIndex(0), [draft]);
  const statusText = disabled
    ? placeholder ?? "Sagent is not ready"
    : cancelling
      ? "Stopping the current turn…"
      : busy
        ? `Sagent is working with ${targetAgentLabel}`
        : `Message ${targetAgentLabel}`;

  const submit = (): void => {
    const text = draft.trim();
    if (!text || !canSend) {
      return;
    }
    onSubmit(text);
    setComposerDraft("");
    onDraftChange?.("");
    // Keep the normal prompt workflow fast: after sending, the user can
    // immediately begin another prompt without reaching for the mouse.
    textareaRef.current?.focus({ preventScroll: true });
  };

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>): void => {
    const nextDraft = event.target.value;
    setComposerDraft(nextDraft);
    onDraftChange?.(nextDraft);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    onInputKeyDown?.(event);
    if (event.defaultPrevented) {
      return;
    }
    if (showCommandMenu && event.key === "ArrowDown") {
      event.preventDefault();
      setActiveCommandIndex((index) => Math.min(index + 1, commandItems.length - 1));
      return;
    }
    if (showCommandMenu && event.key === "ArrowUp") {
      event.preventDefault();
      setActiveCommandIndex((index) => Math.max(index - 1, 0));
      return;
    }
    if (showCommandMenu && (event.key === "Tab" || (event.key === "Enter" && !event.ctrlKey && !event.metaKey))) {
      event.preventDefault();
      const item = commandItems[activeCommandIndex];
      if (item) {
        setComposerDraft(item.insertText + " ");
        onDraftChange?.(item.insertText + " ");
      }
      return;
    }
    // Enter remains a newline. Ctrl/Cmd+Enter submits, except while an IME is
    // composing text; intercepting that Enter would corrupt non-Latin input.
    if (
      event.key === "Enter"
      && (event.ctrlKey || event.metaKey)
      && !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <section
      aria-label="Sagent message composer"
      aria-busy={busy}
      className={joinClasses(
        "border-t p-3",
        sagentUiTokens.border,
        sagentUiTokens.surface,
      )}
    >
      <div
        aria-label={`Message ${targetAgentLabel} in ${targetSessionLabel}`}
        className="relative overflow-visible rounded-md border border-sagent-border bg-sagent-input shadow-sm focus-within:border-sagent-focus"
        role="group"
      >
        {showCommandMenu && (
          <div aria-label="Sagent commands" className="command-completion" role="listbox">
            {commandItems.map((item, index) => (
              <button
                aria-selected={index === activeCommandIndex}
                className={index === activeCommandIndex ? "command-completion-item active" : "command-completion-item"}
                key={item.descriptor.name}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  setComposerDraft(item.insertText + " ");
                  onDraftChange?.(item.insertText + " ");
                  textareaRef.current?.focus({ preventScroll: true });
                }}
                role="option"
                type="button"
              >
                <code>{item.label}</code>
                <span>{item.detail}</span>
              </button>
            ))}
          </div>
        )}
        <textarea
          aria-describedby={[statusId, error ? errorId : undefined].filter(Boolean).join(" ")}
          ref={textareaRef}
          aria-keyshortcuts="Control+Enter Meta+Enter"
          aria-label={`Message ${targetAgentLabel}`}
          aria-invalid={error ? true : undefined}
          className={joinClasses(
            "block max-h-[220px] min-h-[72px] w-full resize-none border-0 px-3 py-2.5 outline-none placeholder:text-sagent-muted-foreground disabled:cursor-not-allowed",
            sagentUiTokens.input,
          )}
          disabled={disabled}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={statusText}
          rows={1}
          value={draft}
        />
        <span className="sr-only" id={statusId} role="status" aria-live="polite">
          {statusText}
        </span>
        <div className="flex items-center justify-between gap-2 px-2 pb-2">
          <div className="min-w-0 flex items-center gap-1">
            <Button
              aria-haspopup="dialog"
              aria-label={`Open Rekursiv settings: ${rekursivLabel}`}
              className="max-w-40 truncate px-2 text-xs"
              onClick={(event) => onOpenRekursivSettings?.(event.currentTarget)}
              title="Rekursiv capabilities"
              variant="ghost"
            >
              {rekursivLabel}
            </Button>
          </div>
          <div className="flex min-w-0 items-center gap-1">
            <Button
              aria-haspopup="dialog"
              aria-label={`Open model settings: ${modelLabel}`}
              className="max-w-56 truncate px-2 text-xs"
              disabled={!modelSettingsAvailable}
              onClick={(event) => onOpenModelSettings?.(event.currentTarget)}
              title={modelSettingsAvailable ? modelDiagnostics ?? modelLabel : "Model settings are loading"}
              variant="ghost"
            >
              {modelLabel}
            </Button>
            {busy ? (
              <Button
                aria-label={cancelling ? "Stopping Sagent" : "Stop Sagent"}
                disabled={cancelling}
                onClick={onCancel}
                title={cancelling ? "Stopping Sagent" : "Stop Sagent"}
                variant="secondary"
              >
                <Icon name={cancelling ? "loading" : "debug-stop"} />
              </Button>
            ) : (
              <Button
                aria-label="Send message to Sagent"
                disabled={!canSend}
                onClick={submit}
                title="Send message"
                variant="primary"
              >
                <Icon name="send" />
              </Button>
            )}
          </div>
        </div>
      </div>
      {error && (
        <div
          className="mt-2 text-xs text-[var(--vscode-errorForeground)]"
          id={errorId}
          role="alert"
        >
          {error}
        </div>
      )}
    </section>
  );
}
