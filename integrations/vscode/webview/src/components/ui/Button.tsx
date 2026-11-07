import type { ButtonHTMLAttributes, ReactElement } from "react";

import { joinClasses, sagentUiTokens } from "../../ui";

type ButtonVariant = "primary" | "secondary" | "ghost";

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-sagent-button text-sagent-button-foreground hover:bg-sagent-button-hover",
  secondary:
    "border border-sagent-border bg-sagent-raised-surface text-sagent-foreground hover:bg-sagent-hover",
  ghost:
    "text-sagent-muted-foreground hover:bg-sagent-hover hover:text-sagent-foreground",
};

/**
 * The extension owns this small visual primitive so feature components share
 * focus, sizing, and VS Code theme behavior without importing product logic.
 */
export function Button({
  className = "",
  type = "button",
  variant = "secondary",
  ...props
}: ButtonProps): ReactElement {
  return (
    <button
      {...props}
      className={joinClasses(
        "inline-flex min-h-7 items-center justify-center gap-1.5 rounded px-2.5 text-[var(--vscode-font-size)] font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
        sagentUiTokens.focus,
        variantClasses[variant],
        className,
      )}
      type={type}
    />
  );
}
