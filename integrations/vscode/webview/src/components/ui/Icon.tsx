import type { ReactElement } from "react";

export type CodiconName =
  | "add"
  | "close"
  | "copy"
  | "debug-stop"
  | "loading"
  | "organization"
  | "refresh"
  | "send"
  | "settings"
  | "diff"
  | "file"
  | "symbol-event";

export function Icon({
  name,
  className = "",
  label,
}: {
  name: CodiconName;
  className?: string;
  label?: string;
}): ReactElement {
  return (
    <span
      aria-hidden={label === undefined}
      aria-label={label}
      className={["codicon", `codicon-${name}`, className].join(" ")}
      role={label ? "img" : undefined}
    />
  );
}
