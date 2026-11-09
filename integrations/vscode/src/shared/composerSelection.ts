export type ComposerSelection = {
  text: string;
  source: "editor" | "chat";
  label?: string;
  languageId?: string;
  filePath?: string;
  startLine?: number;
  endLine?: number;
};

/**
 * Turn a selection into visible, editable composer text.
 *
 * The selection remains ordinary prompt text so the user can revise it before
 * sending. Source metadata is included without requiring a second context
 * transport or changing Sagent's native message protocol.
 */
export function formatComposerSelection(selection: ComposerSelection): string {
  const text = selection.text.trim();
  if (!text) {
    return "";
  }

  if (selection.source === "chat") {
    const quote = text
      .split(/\r?\n/)
      .map((line) => `> ${line}`)
      .join("\n");
    const label = selection.label ? ` from ${selection.label}` : "";
    return `\n\nSelected text${label}:\n${quote}\n`;
  }

  const location = selection.filePath
    ? `${selection.filePath}${formatLineRange(selection.startLine, selection.endLine)}`
    : selection.label ?? "Selected editor text";
  const language = selection.languageId && selection.languageId !== "plaintext"
    ? selection.languageId
    : "";
  return `\n\nSelected text from ${location}:\n\`\`\`${language}\n${text}\n\`\`\`\n`;
}

function formatLineRange(startLine: number | undefined, endLine: number | undefined): string {
  if (startLine === undefined) {
    return "";
  }
  return `:${startLine}${endLine !== undefined && endLine !== startLine ? `-${endLine}` : ""}`;
}
