import { useCallback, useLayoutEffect, useRef, useState, type RefObject } from "react";

const BOTTOM_TOLERANCE_PX = 8;
const SHOW_BUTTON_THRESHOLD_PX = 96;

export type ConversationScrollController = {
  containerRef: RefObject<HTMLElement | null>;
  contentRef: RefObject<HTMLDivElement | null>;
  showScrollToLatest: boolean;
  scrollToLatest: () => void;
};

function distanceFromBottom(element: HTMLElement): number {
  return Math.max(0, element.scrollHeight - element.scrollTop - element.clientHeight);
}

/**
 * Keeps the reader in control of the transcript viewport.
 *
 * Content updates follow the bottom only when the reader was already there.
 * The content observer covers delayed layout changes such as markdown images
 * and expanded tool output; it never takes control after the reader scrolls up.
 */
export function useConversationScroll(contentVersion: unknown): ConversationScrollController {
  const containerRef = useRef<HTMLElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);
  const initializedRef = useRef(false);
  const frameRef = useRef<number | undefined>(undefined);
  const [showScrollToLatest, setShowScrollToLatest] = useState(false);

  const updatePinnedState = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const distance = distanceFromBottom(container);
    const nextPinned = distance <= BOTTOM_TOLERANCE_PX;
    const nextShowButton = distance >= SHOW_BUTTON_THRESHOLD_PX;
    pinnedRef.current = nextPinned;
    setShowScrollToLatest((current) => (
      current === nextShowButton ? current : nextShowButton
    ));
  }, []);

  const scheduleScrollToLatest = useCallback(() => {
    if (!pinnedRef.current || frameRef.current !== undefined) {
      return;
    }
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = undefined;
      const container = containerRef.current;
      if (container && pinnedRef.current) {
        container.scrollTop = container.scrollHeight;
      }
    });
  }, []);

  const scrollToLatest = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    pinnedRef.current = true;
    setShowScrollToLatest(false);
    container.scrollTop = container.scrollHeight;
  }, []);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const handleScroll = () => updatePinnedState();
    container.addEventListener("scroll", handleScroll, { passive: true });

    // A newly mounted transcript is not user-scrolled, even if its initial
    // content is taller than the viewport. Establish the latest position
    // before measuring so the first measurement cannot show the jump button.
    if (!initializedRef.current) {
      initializedRef.current = true;
      pinnedRef.current = true;
      container.scrollTop = container.scrollHeight;
    }
    updatePinnedState();

    const content = contentRef.current;
    const resizeObserver = typeof ResizeObserver === "undefined" || !content
      ? undefined
      : new ResizeObserver(() => scheduleScrollToLatest());
    if (resizeObserver && content) {
      resizeObserver.observe(content);
    }

    return () => {
      container.removeEventListener("scroll", handleScroll);
      resizeObserver?.disconnect();
      if (frameRef.current !== undefined) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = undefined;
      }
    };
  }, [scheduleScrollToLatest, updatePinnedState]);

  useLayoutEffect(() => {
    scheduleScrollToLatest();
  }, [contentVersion, scheduleScrollToLatest]);

  return {
    containerRef,
    contentRef,
    showScrollToLatest,
    scrollToLatest,
  };
}
