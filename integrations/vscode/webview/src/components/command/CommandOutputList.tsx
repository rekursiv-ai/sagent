import { useEffect, useState, type CSSProperties, type ReactElement } from "react";

import type { CommandOutputRecord } from "../../../../src/shared/conversationTypes";

export type CommandOutputListProps = {
  onDelete?: (id: string) => void;
  outputs: readonly CommandOutputRecord[];
};

export function CommandOutputEntry({
  onDelete,
  output,
}: {
  onDelete?: () => void;
  output: CommandOutputRecord;
}): ReactElement {
  const [contextMenu, setContextMenu] = useState<{ left: number; top: number }>();

  useEffect(() => {
    if (!contextMenu) {
      return;
    }
    const close = () => setContextMenu(undefined);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
      }
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [contextMenu]);

  const handleContextMenu = (event: React.MouseEvent<HTMLElement>) => {
    if (!onDelete) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const menuWidth = 190;
    const menuHeight = 42;
    setContextMenu({
      left: Math.min(event.clientX, Math.max(8, window.innerWidth - menuWidth - 8)),
      top: Math.min(event.clientY, Math.max(8, window.innerHeight - menuHeight - 8)),
    });
  };

  const menuStyle: CSSProperties | undefined = contextMenu
    ? { left: contextMenu.left, top: contextMenu.top }
    : undefined;

  return (
    <article
      className={`command-output command-output-${output.tone}`}
      onContextMenu={handleContextMenu}
    >
      {output.command && <code>{output.command}</code>}
      <pre>{output.text}</pre>
      {contextMenu && onDelete && (
        <div
          className="command-output-context-menu"
          role="menu"
          style={menuStyle}
          onPointerDown={(event) => event.stopPropagation()}
        >
          <button
            role="menuitem"
            type="button"
            onClick={() => {
              setContextMenu(undefined);
              onDelete();
            }}
          >
            Delete command output
          </button>
        </div>
      )}
    </article>
  );
}

/** Render native Sagent command feedback separately. */
export function CommandOutputList({ onDelete, outputs }: CommandOutputListProps): ReactElement | null {
  if (outputs.length === 0) {
    return null;
  }
  return (
    <section aria-label="Sagent command output" className="command-output-list">
      {outputs.map((output) => (
        <CommandOutputEntry
          key={output.id}
          onDelete={onDelete ? () => onDelete(output.id) : undefined}
          output={output}
        />
      ))}
    </section>
  );
}
