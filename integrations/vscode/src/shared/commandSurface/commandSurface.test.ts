import { strict as assert } from "node:assert";
import test from "node:test";

import {
  commandCompletionContext,
  commandCompletionItems,
  createCommandInvocation,
  type SagentCommandDescriptor,
} from "./commandSurface";

const descriptors: readonly SagentCommandDescriptor[] = [
  {
    name: "compact",
    kind: "action",
    scope: "session",
    helpShort: "compact history",
    usage: "[hints]",
    readable: false,
    operation: "run-action",
  },
  {
    name: "model.thinking.effort",
    kind: "setting",
    scope: "agent",
    helpShort: "reasoning effort",
    usage: "LEVEL",
    options: ["none", "low", "high"],
    readable: true,
    operation: "apply-setting",
  },
  {
    name: "tasks",
    kind: "action",
    scope: "runtime",
    helpShort: "list running tool calls",
    readable: false,
    operation: "run-action",
  },
];

test("completion context only accepts an incomplete slash command token", () => {
  assert.deepEqual(commandCompletionContext("/"), { query: "" });
  assert.deepEqual(commandCompletionContext("/model.thinking."), {
    query: "model.thinking.",
  });
  assert.equal(commandCompletionContext("/compact "), undefined);
  assert.equal(commandCompletionContext("fix the bug"), undefined);
});

test("completion items preserve native metadata without executing a command", () => {
  const items = commandCompletionItems(descriptors, { query: "model" });

  assert.equal(items.length, 1);
  assert.equal(items[0]?.label, "/model.thinking.effort");
  assert.equal(items[0]?.insertText, "/model.thinking.effort");
  assert.equal(items[0]?.detail, "reasoning effort · LEVEL");
  assert.deepEqual(items[0]?.descriptor.options, ["none", "low", "high"]);
});

test("a readable setting with no arguments creates a native read intent", () => {
  const invocation = createCommandInvocation(descriptors[1]!, "  ");

  assert.deepEqual(invocation.request, {
    commandName: "model.thinking.effort",
    argumentsText: "  ",
    operation: "read-setting",
  });
});

test("command arguments remain opaque and are never re-parsed by the surface", () => {
  const argumentsText = 'high --literal "quoted value"';
  const invocation = createCommandInvocation(descriptors[1]!, argumentsText);

  assert.equal(invocation.request.operation, "apply-setting");
  assert.equal(invocation.request.argumentsText, argumentsText);
});

