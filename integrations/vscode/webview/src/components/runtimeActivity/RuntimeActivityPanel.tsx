import type { ReactElement, ReactNode } from "react";

import type { RuntimeActivityState } from "../../../../src/shared/runtimeActivity/runtimeActivityModel";
import { Icon } from "../ui/Icon";

/**
 * Render the structured facts Sagent has already published about the live run.
 * This is intentionally read-only: runtime control and accounting remain in
 * Sagent, while VS Code contributes the native presentation surface.
 */
export function RuntimeActivityPanel({
  activity,
}: {
  readonly activity: RuntimeActivityState;
}): ReactElement | null {
  const hasActivity = activity.statusText !== undefined
    || activity.tools.length > 0
    || activity.retry.status !== "idle"
    || activity.compaction.status !== "idle"
    || activity.diagnostics.agents.length > 0;
  if (!hasActivity) {
    return null;
  }

  return (
    <section
      aria-label="Sagent runtime activity"
      className="flex flex-col gap-2 border-b border-sagent-border p-3"
    >
      <header className="flex items-center gap-2 text-sm font-semibold">
        <Icon name="symbol-event" />
        <span>Runtime activity</span>
        {activity.statusText && (
          <span className="ml-auto max-w-[60%] truncate text-xs font-normal text-sagent-muted-foreground">
            {activity.statusText}
          </span>
        )}
      </header>

      {activity.retry.status === "waiting" && (
        <ActivityNotice>
          Retry {activity.retry.attempt}: {activity.retry.message}
        </ActivityNotice>
      )}

      {activity.compaction.status === "running" && (
        <ActivityNotice>Compacting session…</ActivityNotice>
      )}
      {activity.compaction.status === "completed" && (
        <ActivityNotice>
          Compacted {activity.compaction.tokenBefore} → {activity.compaction.tokenAfter} tokens
        </ActivityNotice>
      )}
      {activity.compaction.status === "failed" && (
        <ActivityNotice>Compaction failed: {activity.compaction.message}</ActivityNotice>
      )}

      {activity.tools.length > 0 && (
        <details open>
          <summary className="cursor-pointer text-xs font-semibold text-sagent-muted-foreground">
            Tools ({activity.tools.length})
          </summary>
          <ul className="m-0 mt-2 flex list-none flex-col gap-1 p-0">
            {activity.tools.map((tool) => (
              <li key={tool.callId} className="flex min-w-0 items-start gap-2 text-xs">
                <Icon name={tool.status === "running" ? "loading" : "symbol-event"} />
                <span className="min-w-0 flex-1">
                  <span className="font-medium">{tool.name}</span>
                  <span className="ml-1 text-sagent-muted-foreground">{tool.summary}</span>
                  {tool.path && (
                    <span className="block truncate text-sagent-muted-foreground">{tool.path}</span>
                  )}
                </span>
                <span className="text-sagent-muted-foreground">{tool.status}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {activity.diagnostics.agents.length > 0 && (
        <details>
          <summary className="cursor-pointer text-xs font-semibold text-sagent-muted-foreground">
            Agent diagnostics ({activity.diagnostics.agents.length})
          </summary>
          <ul className="m-0 mt-2 flex list-none flex-col gap-1 p-0 text-xs">
            {activity.diagnostics.agents.map((agent) => (
              <li key={agent.label} className="flex min-w-0 items-center gap-2">
                <span className="min-w-0 flex-1 truncate">{agent.label}</span>
                <span className="text-sagent-muted-foreground">
                  {agent.model ?? "model unavailable"}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ActivityNotice({ children }: { readonly children: ReactNode }): ReactElement {
  return (
    <div className="text-xs text-sagent-muted-foreground" role="status">
      {children}
    </div>
  );
}
