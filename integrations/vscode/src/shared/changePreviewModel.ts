/**
 * Serializable line data for the optional change hover preview.
 *
 * The webview renders this presentation model, while the extension host
 * obtains the original/current resources through VS Code.  Keeping the file
 * system and diff lookup out of this module makes the preview easy to remove
 * without changing Sagent's session model or native click-to-review path.
 */
export type ChangePreviewLine = {
  readonly kind: "context" | "added" | "removed";
  readonly text: string;
  readonly oldLineNumber?: number;
  readonly newLineNumber?: number;
};

export type ChangePreview =
  | {
      readonly status: "ready";
      readonly changeId: string;
      readonly path: string;
      readonly lines: readonly ChangePreviewLine[];
      readonly addedLines: number;
      readonly removedLines: number;
    }
  | {
      readonly status: "unavailable";
      readonly changeId: string;
      readonly path: string;
      readonly message: string;
    };
