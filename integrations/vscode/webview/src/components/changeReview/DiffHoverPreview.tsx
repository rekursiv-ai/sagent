import { useEffect, useRef, useState, type ReactElement, type ReactNode } from "react";

import type { ChangePreview } from "../../../../src/shared/changePreviewModel";

export type DiffHoverPreviewProps = {
  readonly children: ReactNode;
  readonly onRequest: () => void;
  readonly preview?: ChangePreview;
};

/**
 * A deliberately small, removable presentation layer for diff previews.
 * It consumes structured lines from the host and leaves opening the full
 * native VS Code diff editor to the child action's existing click handler.
 */
export function DiffHoverPreview({
  children,
  onRequest,
  preview,
}: DiffHoverPreviewProps): ReactElement {
  const [visible, setVisible] = useState(false);
  const requested = useRef(false);
  const hoverTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => {
    if (hoverTimer.current !== undefined) {
      window.clearTimeout(hoverTimer.current);
    }
  }, []);

  const openPreview = (): void => {
    setVisible(true);
    if (!requested.current && !preview) {
      requested.current = true;
      onRequest();
    }
  };

  const schedulePreview = (): void => {
    if (hoverTimer.current !== undefined) {
      window.clearTimeout(hoverTimer.current);
    }
    hoverTimer.current = window.setTimeout(() => {
      hoverTimer.current = undefined;
      openPreview();
    }, 500);
  };

  const closePreview = (): void => {
    if (hoverTimer.current !== undefined) {
      window.clearTimeout(hoverTimer.current);
      hoverTimer.current = undefined;
    }
    setVisible(false);
  };

  return (
    <div
      className="diff-hover-preview-anchor"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          closePreview();
        }
      }}
      onFocus={openPreview}
      onMouseEnter={schedulePreview}
      onMouseLeave={closePreview}
    >
      {children}
      {visible && preview && <DiffPreviewPopover preview={preview} />}
    </div>
  );
}

function DiffPreviewPopover({ preview }: { readonly preview: ChangePreview }): ReactElement {
  if (preview.status === "unavailable") {
    return <div className="diff-preview-popover diff-preview-unavailable">{preview.message}</div>;
  }

  return (
    <div
      aria-label={`Diff preview for ${preview.path}`}
      className="diff-preview-popover"
      role="tooltip"
    >
      <div className="diff-preview-header">
        <span className="diff-preview-path">{preview.path}</span>
        <span className="diff-preview-counts">
          <span className="diff-preview-added">+{preview.addedLines}</span>
          <span className="diff-preview-removed">-{preview.removedLines}</span>
        </span>
      </div>
      <div className="diff-preview-lines">
        {preview.lines.map((line, index) => (
          <div className={`diff-preview-line diff-preview-line-${line.kind}`} key={`${index}-${line.kind}`}>
            <span className="diff-preview-line-number">{line.oldLineNumber ?? ""}</span>
            <span className="diff-preview-line-number">{line.newLineNumber ?? ""}</span>
            <span className="diff-preview-marker" aria-hidden="true">
              {line.kind === "added" ? "+" : line.kind === "removed" ? "−" : " "}
            </span>
            <code>{line.text || " "}</code>
          </div>
        ))}
      </div>
    </div>
  );
}
