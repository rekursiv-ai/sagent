import { useEffect, useLayoutEffect, useRef, useState, type ReactElement } from "react";
import { createPortal } from "react-dom";

import type { ComposerSelection } from "../../../../src/shared/composerSelection";
import { Button } from "../ui/Button";

type SelectionActionProps = {
  children: ReactElement;
  onAdd: (selection: ComposerSelection) => void;
};

type SelectionState = {
  text: string;
  anchorLeft: number;
  anchorTop: number;
  anchorBottom: number;
};

type SelectionPosition = {
  left: number;
  top: number;
};

/** Adds a Codex-style action for text selected inside the Sagent transcript. */
export function SelectionAction({ children, onAdd }: SelectionActionProps): ReactElement {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const actionRef = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<SelectionState>();
  const [position, setPosition] = useState<SelectionPosition>();

  useEffect(() => {
    const update = (): void => {
      if (
        actionRef.current?.matches(":hover")
        || actionRef.current?.contains(document.activeElement)
      ) {
        return;
      }
      const nativeSelection = window.getSelection();
      const surface = surfaceRef.current;
      if (!nativeSelection || nativeSelection.isCollapsed || !surface || nativeSelection.rangeCount === 0) {
        setSelection(undefined);
        return;
      }
      const range = nativeSelection.getRangeAt(0);
      const container = range.commonAncestorContainer instanceof Element
        ? range.commonAncestorContainer
        : range.commonAncestorContainer.parentElement;
      const message = container?.closest("[data-chat-selection]");
      if (!message || !surface.contains(message)) {
        setSelection(undefined);
        return;
      }
      const text = nativeSelection.toString().trim();
      const rect = range.getBoundingClientRect();
      if (!text || (rect.width === 0 && rect.height === 0)) {
        setSelection(undefined);
        return;
      }
      setSelection({
        text,
        anchorLeft: rect.left,
        anchorTop: rect.top,
        anchorBottom: rect.bottom,
      });
    };

    document.addEventListener("selectionchange", update);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      document.removeEventListener("selectionchange", update);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, []);

  useLayoutEffect(() => {
    if (!selection || !actionRef.current) {
      setPosition(undefined);
      return;
    }

    const actionRect = actionRef.current.getBoundingClientRect();
    const edgePadding = 8;
    const gap = 8;
    const maxLeft = Math.max(edgePadding, window.innerWidth - actionRect.width - edgePadding);
    const maxTop = Math.max(edgePadding, window.innerHeight - actionRect.height - edgePadding);
    const left = Math.max(edgePadding, Math.min(selection.anchorLeft, maxLeft));
    const preferredTop = selection.anchorTop - actionRect.height - gap;
    const fallbackTop = selection.anchorBottom + gap;
    const top = preferredTop >= edgePadding
      ? preferredTop
      : Math.min(fallbackTop, maxTop);

    setPosition({ left, top: Math.max(edgePadding, top) });
  }, [selection]);

  return (
    <div className="chat-selection-surface" ref={surfaceRef}>
      {children}
      {selection && createPortal(
        <div
          className="chat-selection-action"
          ref={actionRef}
          style={{
            left: position?.left,
            top: position?.top,
            visibility: position ? "visible" : "hidden",
          }}
        >
          <Button
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => {
              onAdd({ source: "chat", text: selection.text });
              setSelection(undefined);
            }}
            variant="primary"
          >
            Add to chat
          </Button>
        </div>,
        document.body,
      )}
    </div>
  );
}
