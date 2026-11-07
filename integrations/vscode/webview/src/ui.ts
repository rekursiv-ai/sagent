/**
 * Shared Sagent presentation tokens mapped to VS Code's active theme.
 *
 * Components use these tokens instead of hard-coding light/dark colors. VS Code
 * owns the theme values; this layer only gives the Sagent UI a consistent
 * vocabulary for surfaces, text, borders, and interactive states.
 */
export const sagentUiTokens = {
  surface: "bg-sagent-surface",
  raisedSurface: "bg-sagent-raised-surface",
  foreground: "text-sagent-foreground",
  mutedForeground: "text-sagent-muted-foreground",
  border: "border-sagent-border",
  focus: "focus-visible:outline-sagent-focus",
  input: "bg-sagent-input text-sagent-input-foreground",
} as const;

export function joinClasses(...classNames: Array<string | false | undefined>): string {
  return classNames.filter(Boolean).join(" ");
}
