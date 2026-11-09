import { useEffect, useRef, useState, type ReactElement } from "react";

import { Button, type ButtonProps } from "./Button";
import { Icon } from "./Icon";

type CopyStatus = "idle" | "copied" | "failed";

export type CopyButtonProps = Omit<ButtonProps, "aria-label" | "children" | "onClick" | "title"> & {
  label: string;
  persistent?: boolean;
  text: string;
};

export function CopyButton({
  className = "",
  label,
  persistent = false,
  text,
  ...props
}: CopyButtonProps): ReactElement {
  const [status, setStatus] = useState<CopyStatus>("idle");
  const resetTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => {
    if (resetTimer.current !== undefined) {
      window.clearTimeout(resetTimer.current);
    }
  }, []);

  async function handleCopy(): Promise<void> {
    if (resetTimer.current !== undefined) {
      window.clearTimeout(resetTimer.current);
    }
    try {
      await copyText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    resetTimer.current = window.setTimeout(() => setStatus("idle"), 1_500);
  }

  const copied = status === "copied";
  const failed = status === "failed";
  return (
    <Button
      {...props}
      aria-label={copied ? `${label} copied` : failed ? `${label} failed` : label}
      className={[
        "copy-hover-action copy-button-feedback",
        persistent || status !== "idle" ? "copy-action-persistent" : "",
        copied ? "copy-button-copied" : "",
        failed ? "copy-button-failed" : "",
        className,
      ].filter(Boolean).join(" ")}
      onClick={() => void handleCopy()}
      title={copied ? "Copied" : failed ? "Copy failed" : label}
      variant="ghost"
    >
      <Icon name={copied ? "check" : "copy"} />
    </Button>
  );
}

async function copyText(text: string): Promise<void> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch {
    // Fall through to the legacy document command for VS Code webviews.
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Could not copy text");
  }
}
