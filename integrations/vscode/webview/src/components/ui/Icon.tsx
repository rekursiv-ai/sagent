import type { ReactElement } from "react";

export type CodiconName =
  | "add"
  | "close"
  | "check"
  | "chevron-down"
  | "copy"
  | "debug-stop"
  | "loading"
  | "organization"
  | "refresh"
  | "send"
  | "settings"
  | "diff"
  | "file"
  | "symbol-event"
  | "screen-full"
  | "screen-normal"
  | "zoom-in"
  | "zoom-out";

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
