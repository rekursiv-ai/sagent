import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  createChangeReviewError,
  createChangeReviewModel,
  toChangeReviewPresentationTarget,
  type ChangeReviewPresentationTarget,
} from "./changeReviewModel";

function target(
  recordId: string,
  filePath: string,
  turnId: string,
  agentLabel: string,
): ChangeReviewPresentationTarget {
  return toChangeReviewPresentationTarget(
    {
      recordId,
      filePath,
      toolName: "Edit",
      turnId,
    },
    agentLabel,
  );
}

test("groups native changes by agent and turn while preserving record order", () => {
  const model = createChangeReviewModel([
    target("record-1", "/workspace/a.py", "turn-1", "Agent"),
    target("record-2", "/workspace/b.py", "turn-1", "Agent"),
    target("record-3", "/workspace/c.py", "turn-1", "reviewer"),
    target("record-4", "/workspace/d.py", "turn-2", "Agent"),
  ]);

  assert.equal(model.status, "review");
  if (model.status !== "review") {
    return;
  }

  assert.equal(model.fileCount, 4);
  assert.deepEqual(
    model.groups.map(({ agentLabel, turnId, files }) => ({
      agentLabel,
      turnId,
      files: files.map(({ id, path }) => ({ id, path })),
    })),
    [
      {
        agentLabel: "Agent",
        turnId: "turn-1",
        files: [
          { id: "record-1", path: "/workspace/a.py" },
          { id: "record-2", path: "/workspace/b.py" },
        ],
      },
      {
        agentLabel: "reviewer",
        turnId: "turn-1",
        files: [{ id: "record-3", path: "/workspace/c.py" }],
      },
      {
        agentLabel: "Agent",
        turnId: "turn-2",
        files: [{ id: "record-4", path: "/workspace/d.py" }],
      },
    ],
  );
});

test("returns an explicit empty state without inventing a change", () => {
  assert.deepEqual(createChangeReviewModel([]), {
    status: "empty",
    message: "No file changes are available to review.",
  });
});

test("returns an explicit error state and preserves retryability", () => {
  assert.deepEqual(createChangeReviewError("Git is unavailable.", true), {
    status: "error",
    message: "Git is unavailable.",
    retryable: true,
  });
});
