import type { ReactElement } from "react";

import type { AgentActivityRecord } from "../../../../src/shared/conversationTypes";

export type AgentActivityRecordListProps = {
  records: AgentActivityRecord[];
};

/**
 * Render the structured records Sagent reports for a child agent.
 *
 * The record discriminant is the source of truth for the label. We do not
 * inspect or parse the text because the extension must remain compatible with
 * Sagent's typed activity payloads as their presentation changes.
 */
export function AgentActivityRecordList({
  records,
}: AgentActivityRecordListProps): ReactElement | null {
  if (records.length === 0) {
    return null;
  }

  return (
    <ul className="agent-records" aria-label="Agent activity records">
      {records.map((record, index) => (
        <li
          className="agent-detail"
          key={`${record.kind}:${record.callId ?? index}`}
        >
          <span className="agent-record-kind">{recordLabel(record)}</span>
          {record.text ? `: ${record.text}` : ""}
        </li>
      ))}
    </ul>
  );
}

function recordLabel(record: AgentActivityRecord): string {
  if (record.kind === "tool") {
    return record.name ?? "Tool";
  }
  return "Agent response";
}
