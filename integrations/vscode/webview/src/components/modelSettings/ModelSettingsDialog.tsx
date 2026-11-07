import type { ReactElement } from "react";

import type {
  ModelSettingsAction,
  ModelSettingsSnapshot,
} from "../../../../src/shared/modelSettings/modelSettingsContract";
import { ModelDialog } from "./ModelDialog";
import { ModelSettingsPanel } from "./ModelSettingsPanel";

export function ModelSettingsDialog({
  anchor,
  open,
  snapshot,
  onAction,
  onClose,
}: {
  readonly anchor: HTMLElement | null;
  readonly open: boolean;
  readonly snapshot: ModelSettingsSnapshot;
  readonly onAction: (action: ModelSettingsAction) => void;
  readonly onClose: () => void;
}): ReactElement {
  return (
    <ModelDialog
      anchor={anchor}
      ariaLabel="Sagent model capabilities"
      open={open}
      onClose={onClose}
    >
      <ModelSettingsPanel snapshot={snapshot} onAction={onAction} onClose={onClose} />
    </ModelDialog>
  );
}
