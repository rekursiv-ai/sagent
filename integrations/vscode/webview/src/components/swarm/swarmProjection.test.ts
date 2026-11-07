import type { SagentTeamAgent } from "../../../../src/shared/sagentProtocol";
import { buildSwarmTree } from "./swarmProjection";

function agent(
  label: string,
  options: Partial<SagentTeamAgent> = {},
): SagentTeamAgent {
  return {
    label,
    children: [],
    persistent: false,
    running: false,
    tools: [],
    owedReplies: {},
    ...options,
  };
}

function testPreservesTeamChildrenOrderAndNativeHierarchy(): void {
  const tree = buildSwarmTree([
    agent("Agent", { children: ["reviewer", "researcher"] }),
    agent("researcher", { parent: "Agent" }),
    agent("reviewer", { parent: "Agent", children: ["reviewer_0"] }),
    agent("reviewer_0", { parent: "reviewer" }),
  ]);

  assertJsonEqual(
    tree.map((node) => [node.agent.label, node.children.map((child) => child.agent.label)]),
    [["Agent", ["reviewer", "researcher"]]],
  );
  assertEqual(tree[0]?.children[0]?.children[0]?.agent.label, "reviewer_0");
}

function testDoesNotDropAgentsFromAnIncompleteNativeProjection(): void {
  const tree = buildSwarmTree([
    agent("Agent", { children: ["missing"] }),
    agent("orphan", { parent: "missing" }),
  ]);

  assertJsonEqual(
    tree.map((node) => node.agent.label),
    ["Agent", "orphan"],
  );
}

function testDoesNotRecurseForeverWhenProjectionContainsCycle(): void {
  const tree = buildSwarmTree([
    agent("first", { parent: "second", children: ["second"] }),
    agent("second", { parent: "first", children: ["first"] }),
  ]);

  assertEqual(tree.length, 1);
  assertEqual(tree[0]?.children[0]?.agent.label, "second");
}

function assertEqual<T>(actual: T, expected: T): void {
  if (actual !== expected) {
    throw new Error(`Expected ${String(expected)}, received ${String(actual)}`);
  }
}

function assertJsonEqual(actual: unknown, expected: unknown): void {
  const actualJson = JSON.stringify(actual);
  const expectedJson = JSON.stringify(expected);
  if (actualJson !== expectedJson) {
    throw new Error(`Expected ${expectedJson}, received ${actualJson}`);
  }
}

// This file intentionally uses only browser-compatible TypeScript so the
// webview typecheck needs no Node test dependency. The focused check compiles
// these two modules to a temporary CommonJS directory and runs the result with
// Node.
testPreservesTeamChildrenOrderAndNativeHierarchy();
testDoesNotDropAgentsFromAnIncompleteNativeProjection();
testDoesNotRecurseForeverWhenProjectionContainsCycle();
