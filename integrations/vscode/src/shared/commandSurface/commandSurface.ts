/**
 * Presentation contracts for Sagent's slash-command surface.
 *
 * The webview never imports Sagent's command implementation. The extension
 * host asks the Sagent server for metadata projected from its live native
 * command engine, then sends complete command lines back for native parsing
 * and execution.
 */

export type SagentCommandKind = "action" | "setting";

/** The native Sagent owner that is useful context in a command picker. */
export type SagentCommandScope =
  | "session"
  | "agent"
  | "team"
  | "provider"
  | "runtime"
  | "tool"
  | "screen"
  | "mcp";

/**
 * The operations a later server adapter may expose to the UI.
 *
 * These are intent names, not implementations. A request carrying one of
 * these values must still be validated and executed by Sagent. In particular,
 * the extension must never turn `argumentsText` into a model prompt.
 */
export type SagentCommandOperation =
  | "show-help"
  | "run-action"
  | "read-setting"
  | "apply-setting";

/**
 * Serializable metadata corresponding to Sagent's native `Setting`/`Action`.
 *
 * `name` is the dotted name without the leading slash. `options` is a snapshot
 * supplied by Sagent when available; it is not a second source of truth and
 * must not be used to validate or apply a command in the extension.
 */
export type SagentCommandDescriptor = {
  name: string;
  kind: SagentCommandKind;
  scope: SagentCommandScope;
  helpShort: string;
  helpLong?: string;
  usage?: string;
  options?: readonly string[];
  readable: boolean;
  operation: SagentCommandOperation;
};

/** A command request, kept separate from a normal user/model message. */
export type SagentCommandRequest = {
  commandName: string;
  argumentsText: string;
  operation: SagentCommandOperation;
};

/** A command request plus the metadata needed to render its result. */
export type SagentCommandInvocation = {
  descriptor: SagentCommandDescriptor;
  request: SagentCommandRequest;
};

/** The portion of input that is safe to use for completion presentation. */
export type SagentCommandCompletionContext = {
  query: string;
};

/** A webview-friendly completion item; it carries no execution callback. */
export type SagentCommandCompletionItem = {
  label: string;
  insertText: string;
  detail: string;
  documentation?: string;
  descriptor: SagentCommandDescriptor;
};

/**
 * Read only the command-name portion of a prompt for autocomplete.
 *
 * This is deliberately not a slash-command parser. It accepts only the
 * incomplete command token (`/`, `/mo`, `/model.`); once whitespace appears,
 * the native Sagent command parser must own the arguments and their grammar.
 */
export function commandCompletionContext(
  input: string,
): SagentCommandCompletionContext | undefined {
  const match = /^\/([\w-]*(?:\.[\w-]*)*)$/u.exec(input);
  return match ? { query: match[1] ?? "" } : undefined;
}

/**
 * Return command completions using metadata supplied by Sagent.
 *
 * Prefix filtering is presentation-only. It intentionally does not recreate
 * Sagent's exact/suffix/fuzzy `match()` behavior; the native command layer
 * remains authoritative when the completed request is executed.
 */
export function commandCompletionItems(
  descriptors: readonly SagentCommandDescriptor[],
  context: SagentCommandCompletionContext,
): SagentCommandCompletionItem[] {
  const query = context.query.toLocaleLowerCase();
  return descriptors
    .filter((descriptor) => descriptor.name.toLocaleLowerCase().startsWith(query))
    .map((descriptor) => ({
      label: `/${descriptor.name}`,
      insertText: `/${descriptor.name}`,
      detail: descriptor.usage
        ? `${descriptor.helpShort} · ${descriptor.usage}`
        : descriptor.helpShort,
      ...(descriptor.helpLong ? { documentation: descriptor.helpLong } : {}),
      descriptor,
    }));
}

/**
 * Create a typed invocation without interpreting its argument string.
 *
 * The empty/non-empty distinction mirrors Sagent's `Setting` contract: a
 * readable setting with no arguments is queried, while an action or a setting
 * with arguments is sent to the native executor. Quoting, target selection,
 * flags, and validation remain Sagent concerns.
 */
export function createCommandInvocation(
  descriptor: SagentCommandDescriptor,
  argumentsText: string,
): SagentCommandInvocation {
  const trimmedArguments = argumentsText.trim();
  const operation =
    descriptor.kind === "setting" && descriptor.readable && !trimmedArguments
      ? "read-setting"
      : descriptor.operation;

  return {
    descriptor,
    request: {
      commandName: descriptor.name,
      argumentsText,
      operation,
    },
  };
}
