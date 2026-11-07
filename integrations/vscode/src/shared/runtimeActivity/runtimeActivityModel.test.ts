import { strict as assert } from "node:assert";
import { test } from "node:test";

import type { SagentEvent } from "../sagentProtocol";
import type { SagentTeamAgent } from "../sagentProtocol";
import {
  applyRuntimeActivityEvent,
  createInitialRuntimeActivityState,
  diagnosticsFromTeamAgent,
  isBackgroundToolActivity,
  setChangeReviewTargets,
  setTeamDiagnostics,
  toRuntimeActivityEvent,
  formatRuntimeElapsed,
} from "./runtimeActivityModel";

test("formats elapsed runtime using the same units as the Sagent TUI", () => {
  assert.equal(formatRuntimeElapsed(0), "0s");
  assert.equal(formatRuntimeElapsed(83), "1m 23s");
  assert.equal(formatRuntimeElapsed(3_721), "1h 2m 1s");
});

test("starts and ends a native turn activity clock without parsing text", () => {
  let state = createInitialRuntimeActivityState();
  state = applyRuntimeActivityEvent(
    state,
    { type: "turn.started", turnId: "turn-1" },
    10_000,
  );
  assert.deepEqual(state.activeTurn, { turnId: "turn-1", startedAtMs: 10_000 });
  assert.equal(state.phase, "thinking");

  state = applyRuntimeActivityEvent(state, {
    type: "status.changed",
    status: "working",
    text: "working",
    phase: "tools",
  });
  assert.equal(state.phase, "tools");

  state = applyRuntimeActivityEvent(state, {
    type: "turn.completed",
    turnId: "turn-1",
  });
  assert.equal(state.activeTurn, undefined);
  assert.equal(state.phase, "idle");
});

test("preserves structured tool lifecycle and detached background state", () => {
  let state = createInitialRuntimeActivityState();
  state = applyRuntimeActivityEvent(state, {
    type: "tool.started",
    callId: "call-1",
    name: "Bash",
    summary: "Run tests",
    arguments: { command: "pytest" },
    turnId: "turn-1",
  });
  state = applyRuntimeActivityEvent(state, {
    type: "tool.completed",
    callId: "call-1",
    resultCallId: "call-1:detached",
    status: "detached",
    text: "Still running",
    isError: false,
  });

  assert.deepEqual(state.tools, [
    {
      callId: "call-1",
      resultCallId: "call-1:detached",
      name: "Bash",
      summary: "Run tests",
      status: "detached",
      text: "Still running",
      isError: false,
      arguments: { command: "pytest" },
      turnId: "turn-1",
    },
  ]);
  assert.equal(state.tools[0]?.status, "detached");
  assert.equal(isBackgroundToolActivity(state.tools[0]!), true);
});

test("retains retry timing and compaction accounting without deriving it from text", () => {
  let state = createInitialRuntimeActivityState();
  state = applyRuntimeActivityEvent(state, {
    type: "retry.waiting",
    attempt: 2,
    until: 1_900_000_000,
    message: "Provider quota is temporarily unavailable",
  });
  assert.deepEqual(state.retry, {
    status: "waiting",
    attempt: 2,
    until: 1_900_000_000,
    message: "Provider quota is temporarily unavailable",
  });

  state = applyRuntimeActivityEvent(state, { type: "compaction.started" });
  assert.deepEqual(state.compaction, { status: "running" });
  state = applyRuntimeActivityEvent(state, {
    type: "compaction.completed",
    tokenBefore: 120_000,
    tokenAfter: 30_000,
    payloadEntries: 8,
    fallbackReason: "",
    preservedTailCount: 4,
  });
  assert.deepEqual(state.compaction, {
    status: "completed",
    tokenBefore: 120_000,
    tokenAfter: 30_000,
    payloadEntries: 8,
    fallbackReason: "",
    preservedTailCount: 4,
  });
});

test("maps native team diagnostics without estimating missing accounting", () => {
  const agent: SagentTeamAgent = {
    label: "researcher",
    children: [],
    persistent: false,
    running: false,
    tools: ["Read"],
    owedReplies: {},
    model: "gpt-5",
    provider: "OpenAI",
    account: "default",
    spendUsd: 0.42,
    maxSpendUsd: 2,
  };
  assert.deepEqual(diagnosticsFromTeamAgent(agent), {
    label: "researcher",
    model: "gpt-5",
    provider: "OpenAI",
    account: "default",
    spendUsd: 0.42,
    maxSpendUsd: 2,
  });

  const state = setTeamDiagnostics(createInitialRuntimeActivityState(), [agent]);
  assert.equal(state.diagnostics.agents[0]?.spendUsd, 0.42);
  assert.equal(state.diagnostics.agents[0]?.tokens, undefined);
});

test("retains typed change-review grouping inputs without importing the host API", () => {
  const state = setChangeReviewTargets(createInitialRuntimeActivityState(), [
    {
      id: "record-1",
      path: "/workspace/app.py",
      toolName: "Edit",
      turnId: "turn-1",
      agentLabel: "Agent",
    },
    {
      id: "record-2",
      path: "/workspace/other.py",
      toolName: "Write",
      turnId: "turn-1",
      agentLabel: "reviewer",
    },
  ]);

  assert.deepEqual(state.changeReviewTargets, [
    {
      id: "record-1",
      path: "/workspace/app.py",
      toolName: "Edit",
      turnId: "turn-1",
      agentLabel: "Agent",
    },
    {
      id: "record-2",
      path: "/workspace/other.py",
      toolName: "Write",
      turnId: "turn-1",
      agentLabel: "reviewer",
    },
  ]);
});

test("accepts only structured server fields and ignores unrelated notifications", () => {
  const toolEvent = toRuntimeActivityEvent({
    type: "tool.completed",
    callId: "call-2",
    resultCallId: "call-2:detached",
    status: "cancelled",
    text: "cancelled",
    isError: true,
    agentLabel: "Agent",
  } satisfies SagentEvent);
  assert.deepEqual(toolEvent, {
    type: "tool.completed",
    callId: "call-2",
    resultCallId: "call-2:detached",
    status: "cancelled",
    text: "cancelled",
    isError: true,
    agentLabel: "Agent",
  });

  assert.deepEqual(
    toRuntimeActivityEvent({
      type: "status.changed",
      status: "working",
      text: "working",
      phase: "thinking",
    }),
    {
      type: "status.changed",
      status: "working",
      text: "working",
      phase: "thinking",
    },
  );

  assert.equal(
    toRuntimeActivityEvent({ type: "assistant.delta", text: "not runtime state" }),
    undefined,
  );
  assert.equal(
    toRuntimeActivityEvent({
      type: "tool.completed",
      callId: "call-3",
      status: "unknown",
      text: "result",
    }),
    undefined,
  );
});
