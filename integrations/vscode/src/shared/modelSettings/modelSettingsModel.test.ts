import assert from "node:assert/strict";
import test from "node:test";

import type { ModelSettingsSnapshot } from "./modelSettingsContract";
import { createModelSettingsViewModel } from "./modelSettingsModel";

function snapshot(): ModelSettingsSnapshot {
  return {
    status: "ready",
    agentLabel: "/root",
    availableScopes: ["current-agent"],
    selectedScope: "current-agent",
    provider: {
      providers: ["openai-sub", "anthropic-cli"],
      currentProvider: "openai-sub",
      accounts: ["default", "work"],
      currentAccount: "work",
    },
    model: {
      models: ["sol-6.1", "luna-6.0"],
      currentModel: "sol-6.1",
    },
    capability: {
      modelId: "sol-6.1",
      contextSizes: [
        {
          name: "fullcontext",
          tokens: 1_000_000,
          displayName: "1m",
          isDefault: true,
        },
        {
          name: "smallcontext",
          tokens: 272_000,
          displayName: "272k",
          isDefault: false,
        },
      ],
      thinkingEfforts: ["none", "low", "high"],
      thinkingBudgets: ["none", "auto"],
      thinkingOutputs: ["none", "text"],
      serviceTiers: ["auto", "priority"],
      cacheTtlSeconds: [0, 300],
      manageContextServerSide: [false, true],
      defaults: {
        context: {
          name: "fullcontext",
          tokens: 1_000_000,
          displayName: "1m",
          isDefault: true,
        },
        thinkingEffort: "none",
        thinkingBudget: "none",
        thinkingOutput: "none",
        serviceTier: "auto",
        cacheTtlSeconds: 0,
        manageContextServerSide: false,
      },
      limits: {
        maxRequestTokens: 1_000_000,
        maxResponseTokens: 128_000,
        maxContextTokens: 1_000_000,
      },
    },
    settings: {
      context: {
        name: "smallcontext",
        tokens: 272_000,
        displayName: "272k",
        isDefault: false,
      },
      thinkingEffort: "high",
      thinkingBudget: "auto",
      thinkingOutput: "text",
      serviceTier: "priority",
      cacheTtlSeconds: 300,
      manageContextServerSide: true,
    },
    writableSettings: [
      "thinking.effort",
      "thinking.budget",
      "thinking.output",
      "service_tier",
    ],
    diagnostics: {
      label: "/root",
      model: "work@openai-sub/sol-6.1+272k+highthink+priority",
      context: "272k",
      thinking: { budget: "auto", output: "text", effort: "high" },
      serviceTier: "priority",
      cacheTtlSeconds: 300,
      limits: {
        maxRequestTokens: 1_000_000,
        maxResponseTokens: 128_000,
        maxContextTokens: 1_000_000,
      },
      cwd: "/workspace",
      tools: ["Read", "Edit"],
    },
  };
}

test("projects native provider and model inventories without replacing them", () => {
  const view = createModelSettingsViewModel(snapshot());

  assert.deepEqual(view.providerOptions.map((option) => option.value), [
    "openai-sub",
    "anthropic-cli",
  ]);
  assert.deepEqual(view.accountOptions.map((option) => option.value), [
    "default",
    "work",
  ]);
  assert.deepEqual(view.modelOptions.map((option) => option.value), [
    "sol-6.1",
    "luna-6.0",
  ]);
  assert.equal(view.controls[1]?.value, "high");
  assert.equal(view.controls[0]?.value, "smallcontext");
  assert.deepEqual(
    view.controls[1]?.options.map((option) => option.value),
    ["none", "low", "high"],
  );
});

test("preserves Sagent diagnostics and explicit current-agent scope", () => {
  const view = createModelSettingsViewModel(snapshot());

  assert.equal(view.selectedScope, "current-agent");
  assert.deepEqual(view.availableScopes, ["current-agent"]);
  assert.equal(view.diagnostics?.model, "work@openai-sub/sol-6.1+272k+highthink+priority");
  assert.deepEqual(view.diagnostics?.tools, ["Read", "Edit"]);
});

test("does not present settings the current native host cannot mutate", () => {
  const view = createModelSettingsViewModel(snapshot());

  assert.deepEqual(view.controls.map((control) => control.name), [
    "thinking.effort",
    "thinking.budget",
    "thinking.output",
    "service_tier",
  ]);
});
