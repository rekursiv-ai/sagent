import { useLayoutEffect, useRef } from "react";

/**
 * Keeps a multiline composer sized to its content without taking layout
 * ownership away from the parent. The maximum is intentional: an unusually
 * large draft should continue in the textarea instead of pushing the
 * conversation out of the view.
 */
export function useAutosizeTextarea(value: string, maxHeight: number) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    // Reset before measuring so shrinking after deletion is as reliable as
    // growing after insertion. A layout effect avoids a visible one-frame
    // jump in the webview while the prompt is being edited.
    textarea.style.height = "0px";
    textarea.style.overflowY = "hidden";
    const contentHeight = textarea.scrollHeight;
    textarea.style.height = `${Math.min(contentHeight, maxHeight)}px`;
    textarea.style.overflowY = contentHeight > maxHeight ? "auto" : "hidden";
  }, [maxHeight, value]);

  return textareaRef;
}
