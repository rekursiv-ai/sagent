import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from "react";
import { createPortal } from "react-dom";

import type {
  SagentRekursivContext,
  SagentRekursivSkillCandidate,
} from "../../../../src/shared/sagentProtocol";
import { Button } from "../ui/Button";

export function RekursivContextPanel({
  context,
  candidates,
  popoverTarget,
  onSetup,
  onDisable,
}: {
  readonly context?: SagentRekursivContext;
  readonly candidates: SagentRekursivSkillCandidate[];
  readonly popoverTarget?: HTMLElement;
  readonly onSetup: (skills: string[]) => void;
  readonly onDisable: () => void;
}): ReactElement | null {
  const isRekursiv = context?.mode === "rekursiv";
  const [selected, setSelected] = useState<string[]>([]);
  const [visibleChecks, setVisibleChecks] = useState<PopoverState>();
  const [popoverPosition, setPopoverPosition] = useState<PopoverPosition>();
  const hoverTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const hideTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const popoverRef = useRef<HTMLDivElement>(null);
  useEffect(() => () => {
    if (hoverTimeout.current) {
      clearTimeout(hoverTimeout.current);
    }
    if (hideTimeout.current) {
      clearTimeout(hideTimeout.current);
    }
  }, []);
  useLayoutEffect(() => {
    if (!visibleChecks || !popoverRef.current) {
      setPopoverPosition(undefined);
      return;
    }
    const popover = popoverRef.current;
    const place = (): void => {
      const bounds = popover.getBoundingClientRect();
      const gap = 4;
      const edge = 8;
      const belowTop = visibleChecks.anchor.bottom + gap;
      const fitsBelow = belowTop + bounds.height <= window.innerHeight - edge;
      const top = fitsBelow
        ? belowTop
        : Math.max(edge, visibleChecks.anchor.top - bounds.height - gap);
      const left = Math.min(
        Math.max(edge, visibleChecks.anchor.left + 24),
        Math.max(edge, window.innerWidth - bounds.width - edge),
      );
      setPopoverPosition({ left, top });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [visibleChecks]);
  useEffect(() => {
    const configured = new Set(context?.skillDirectories ?? []);
    setSelected(
      candidates
        .filter((candidate) => {
          const available = candidateAvailable(candidate.id, context);
          return available && (isRekursiv
            ? configured.has(candidate.path) || candidate.id === "wesearch"
            : true);
        })
        .map((candidate) => candidate.id),
    );
  }, [candidates, context?.skillDirectories, isRekursiv]);

  const availableSkills = context?.skills.filter((skill) => skill.source !== "builtin") ?? [];
  const toggle = (id: string): void => {
    setSelected((current) => current.includes(id)
      ? current.filter((item) => item !== id)
      : [...current, id]);
  };
  const clearHoverTimeout = (): void => {
    if (hoverTimeout.current) {
      clearTimeout(hoverTimeout.current);
      hoverTimeout.current = undefined;
    }
  };
  const clearHideTimeout = (): void => {
    if (hideTimeout.current) {
      clearTimeout(hideTimeout.current);
      hideTimeout.current = undefined;
    }
  };
  const hidePopover = (): void => {
    clearHoverTimeout();
    clearHideTimeout();
    hideTimeout.current = setTimeout(() => setVisibleChecks(undefined), 100);
  };
  const showPopover = (id: string, anchor: HTMLElement): void => {
    clearHoverTimeout();
    clearHideTimeout();
    setVisibleChecks({ id, anchor: anchor.getBoundingClientRect() });
  };
  const schedulePopover = (id: string, anchor: HTMLElement): void => {
    clearHideTimeout();
    clearHoverTimeout();
    hoverTimeout.current = setTimeout(() => showPopover(id, anchor), 400);
  };
  return (
    <>
      <section className="rekursiv-context" aria-label="Rekursiv context">
      <div className="rekursiv-context-header">
        <span className="message-label">{isRekursiv ? "Toggle Rekursiv skills" : "Rekursiv stack"}</span>
        {isRekursiv && (
          <>
            <Button className="rekursiv-context-action" onClick={onDisable}>
              Disable
            </Button>
            <Button
              className="rekursiv-context-action"
              disabled={selected.length === 0}
              onClick={() => onSetup(selected)}
            >
              Apply skill selection
            </Button>
          </>
        )}
        {!isRekursiv && (
          <Button
            className="rekursiv-context-action"
            disabled={selected.length === 0}
            variant="primary"
            onClick={() => onSetup(selected)}
          >
            Enable Rekursiv
          </Button>
        )}
      </div>
      {!isRekursiv && (
        <p className="rekursiv-context-description">
          Native Rekursiv capabilities were detected and are selected by default.
        </p>
      )}
      <div className="rekursiv-context-candidates">
        {candidates.length === 0 ? (
          <p className="rekursiv-context-description">
            No native Rekursiv skill bundles were detected near this workspace.
          </p>
        ) : candidates.map((candidate) => {
          const available = candidateAvailable(candidate.id, context);
          const checks = candidateChecks(candidate.id, context);
          return (
            <div
              className="rekursiv-context-candidate-wrap"
              key={candidate.id}
              onBlur={hidePopover}
              onFocus={(event) => showPopover(candidate.id, event.currentTarget)}
              onMouseEnter={(event) => schedulePopover(candidate.id, event.currentTarget)}
              onMouseLeave={hidePopover}
            >
              <label className="rekursiv-context-candidate">
                <input
                  checked={selected.includes(candidate.id)}
                  disabled={candidate.id === "wesearch" || !available}
                  onChange={() => toggle(candidate.id)}
                  type="checkbox"
                />
                <span>
                  <strong>
                    {candidate.label}
                    {!available && <em className="rekursiv-context-unavailable"> (unavailable)</em>}
                  </strong>
                  <small>{candidate.description}</small>
                </span>
              </label>
            </div>
          );
        })}
      </div>
      {isRekursiv && availableSkills.length > 0 && (
        <details className="rekursiv-context-skills">
          <summary>
            <span>Available Rekursiv skills</span>
            <span className="rekursiv-context-count">
              {context?.skills.length ?? 0} skills discovered
            </span>
          </summary>
          <ul>
            {availableSkills.map((skill) => (
              <li key={`${skill.source}:${skill.name}`}>
                <details>
                  <summary>{skill.name}</summary>
                  {skill.description && <small>{skill.description}</small>}
                </details>
              </li>
            ))}
          </ul>
        </details>
      )}
      </section>
      {visibleChecks && popoverTarget && createPortal(
        <div
          className="rekursiv-context-popover"
          ref={popoverRef}
          role="tooltip"
          style={popoverStyle(popoverPosition, visibleChecks.anchor)}
          onMouseEnter={clearHideTimeout}
          onMouseLeave={hidePopover}
        >
          <strong>{candidates.find((candidate) => candidate.id === visibleChecks.id)?.label} availability</strong>
          <ul>
            {candidateChecks(visibleChecks.id, context).map((check) => (
              <li key={`${visibleChecks.id}:${check.name}`}>
                <span className={`rekursiv-status rekursiv-status-${check.status}`} />
                <span>
                  <b>{check.name}</b>
                  <small>{check.detail}</small>
                </span>
              </li>
            ))}
          </ul>
        </div>,
        popoverTarget,
      )}
    </>
  );
}

type PopoverState = {
  id: string;
  anchor: DOMRect;
};

type PopoverPosition = {
  left: number;
  top: number;
};

function popoverStyle(
  position: PopoverPosition | undefined,
  anchor: DOMRect,
): CSSProperties {
  return {
    left: position?.left ?? anchor.left + 24,
    top: position?.top ?? anchor.bottom + 4,
  };
}

function candidateChecks(
  id: string,
  context?: SagentRekursivContext,
): Array<{ name: string; status: string; detail: string }> {
  const environment = context?.environment[id];
  return [
    {
      name: "Bundled skill instructions",
      status: id === "wesearch" ? "available" : "available",
      detail: id === "wesearch"
        ? "Wesearch is provided by Sagent's native research tools"
        : "Native skill instructions are bundled in the VS Code extension",
    },
    ...(environment?.checks ?? []),
  ];
}

function candidateAvailable(id: string, context?: SagentRekursivContext): boolean {
  if (id === "wesearch" || id === "priml" || id === "configgle") {
    return true;
  }
  if (id === "trackinizer") {
    const traxCheck = context?.environment.trackinizer?.checks?.find(
      (check) => check.name === "Trax CLI",
    );
    return !traxCheck || traxCheck.status === "available";
  }
  return context?.environment[id]?.status === "available";
}
