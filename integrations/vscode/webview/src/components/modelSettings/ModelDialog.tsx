import { useEffect, useRef, type ReactElement, type ReactNode } from "react";

export function ModelDialog({
  anchor,
  ariaLabel,
  children,
  open,
  onClose,
}: {
  readonly anchor: HTMLElement | null;
  readonly ariaLabel: string;
  readonly children: ReactNode;
  readonly open: boolean;
  readonly onClose: () => void;
}): ReactElement {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  }, [open]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const handleCancel = (event: Event): void => {
      event.preventDefault();
      onClose();
    };
    const handleClose = (): void => onClose();
    dialog.addEventListener("cancel", handleCancel);
    dialog.addEventListener("close", handleClose);
    return () => {
      dialog.removeEventListener("cancel", handleCancel);
      dialog.removeEventListener("close", handleClose);
    };
  }, [onClose]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog || !open || !anchor) return;

    const reposition = (): void => {
      window.requestAnimationFrame(() => {
        if (!dialog.open) return;
        const anchorRect = anchor.getBoundingClientRect();
        const dialogRect = dialog.getBoundingClientRect();
        const edge = 12;
        const gap = 8;
        const left = Math.min(
          Math.max(edge, anchorRect.right - dialogRect.width),
          Math.max(edge, window.innerWidth - dialogRect.width - edge),
        );
        const above = anchorRect.top - dialogRect.height - gap;
        const below = anchorRect.bottom + gap;
        const top = above >= edge
          ? above
          : Math.min(below, Math.max(edge, window.innerHeight - dialogRect.height - edge));
        dialog.style.left = `${left}px`;
        dialog.style.top = `${top}px`;
      });
    };

    reposition();
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    const resizeObserver = new ResizeObserver(reposition);
    resizeObserver.observe(dialog);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      resizeObserver.disconnect();
    };
  }, [anchor, open]);

  return (
    <dialog
      ref={dialogRef}
      aria-label={ariaLabel}
      className="model-settings-dialog"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="model-settings-dialog-content">
        {children}
      </div>
    </dialog>
  );
}
