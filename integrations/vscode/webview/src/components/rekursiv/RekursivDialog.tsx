import { useState, type ReactElement } from "react";

import type {
  SagentRekursivContext,
  SagentRekursivSkillCandidate,
} from "../../../../src/shared/sagentProtocol";
import { ModelDialog } from "../modelSettings/ModelDialog";
import { RekursivContextPanel } from "./RekursivContextPanel";

export function RekursivDialog({
  anchor,
  candidates,
  context,
  open,
  onDisable,
  onSetup,
  onClose,
}: {
  readonly anchor: HTMLElement | null;
  readonly candidates: SagentRekursivSkillCandidate[];
  readonly context?: SagentRekursivContext;
  readonly open: boolean;
  readonly onDisable: () => void;
  readonly onSetup: (skills: string[]) => void;
  readonly onClose: () => void;
}): ReactElement {
  const [dialogElement, setDialogElement] = useState<HTMLDialogElement | null>(null);

  return (
    <ModelDialog
      anchor={anchor}
      ariaLabel="Rekursiv capabilities"
      className="rekursiv-dialog"
      open={open}
      onDialogReady={setDialogElement}
      onClose={onClose}
    >
      <RekursivContextPanel
        candidates={candidates}
        context={context}
        popoverTarget={dialogElement ?? undefined}
        onDisable={onDisable}
        onSetup={onSetup}
      />
    </ModelDialog>
  );
}
