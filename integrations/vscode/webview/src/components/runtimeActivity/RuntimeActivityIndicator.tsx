import { useEffect, useState, type ReactElement } from "react";

import {
  formatRuntimeElapsed,
  type RuntimeActivityPhase,
} from "../../../../src/shared/runtimeActivity/runtimeActivityModel";
import { Icon } from "../ui/Icon";

const phaseLabels: Record<Exclude<RuntimeActivityPhase, "idle">, string> = {
  thinking: "Thinking",
  tools: "Running tools",
  compacting: "Compacting",
  retrying: "Retrying",
  working: "Working",
};

/**
 * Render the live run indicator without owning runtime state.
 *
 * Sagent events decide the phase and the extension host records the turn
 * boundary. This component only repaints the elapsed presentation clock.
 */
export function RuntimeActivityIndicator({
  phase,
  startedAtMs,
}: {
  readonly phase: Exclude<RuntimeActivityPhase, "idle">;
  readonly startedAtMs: number;
}): ReactElement {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const updateClock = (): void => setNowMs(Date.now());
    updateClock();
    const timer = window.setInterval(updateClock, 1_000);
    return () => window.clearInterval(timer);
  }, [startedAtMs]);

  const elapsedSeconds = Math.max(0, (nowMs - startedAtMs) / 1_000);
  const elapsedText = formatRuntimeElapsed(elapsedSeconds);
  return (
    <span
      aria-label={`${phaseLabels[phase]}, running for ${elapsedText}`}
      className="runtime-activity-indicator"
    >
      <Icon className="animate-spin" name="loading" />
      <span>{phaseLabels[phase]}</span>
      <span className="runtime-activity-elapsed">{elapsedText}</span>
    </span>
  );
}
