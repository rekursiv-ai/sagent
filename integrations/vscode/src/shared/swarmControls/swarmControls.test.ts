import { strict as assert } from "node:assert";
import { test } from "node:test";

import type { SagentTeamAgent, SagentTeamState } from "../sagentProtocol";
import {
  createSwarmControlsModel,
  currentSwarmControlCapabilities,
  type SwarmControlCapabilities,
} from "./swarmControls";

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

test("keeps the native Team hierarchy instead of inferring it from labels", () => {
  const root = agent("root", { children: ["reviewer"] });
  const child = agent("reviewer", { parent: "root" });
  const team: SagentTeamState = { open: true, agents: [root, child] };

  const model = createSwarmControlsModel(team, "reviewer");

  assert.equal(model.teamOpen, true);
  assert.deepEqual(model.agents.map(({ details }) => details.agent.label), [
    "root",
    "reviewer",
  ]);
  assert.deepEqual(model.agents[0]?.details.children.map(({ label }) => label), [
    "reviewer",
  ]);
  assert.equal(model.agents[1]?.details.parent?.label, "root");
  assert.equal(model.agents[1]?.isFocused, true);
});

test("retains native metadata in the agent detail contract", () => {
  const reviewer = agent("reviewer", {
    model: "sonnet",
    provider: "anthropic",
    persistent: true,
    running: true,
    tools: ["Read", "Edit"],
    owedReplies: { Agent: "review the patch" },
    spendUsd: 0.42,
  });

  const model = createSwarmControlsModel(
    { open: true, agents: [reviewer] },
    undefined,
  );
  const details = model.agents[0]!.details;

  assert.strictEqual(details.agent, reviewer);
  assert.equal(details.agent.model, "sonnet");
  assert.equal(details.agent.provider, "anthropic");
  assert.deepEqual(details.agent.tools, ["Read", "Edit"]);
  assert.deepEqual(details.agent.owedReplies, { Agent: "review the patch" });
});

test("only current host capabilities are enabled", () => {
  const model = createSwarmControlsModel(
    { open: true, agents: [agent("reviewer")] },
    undefined,
  );
  const controls = model.agents[0]!.affordances;

  assert.equal(
    controls.find(({ type }) => type === "focus-agent")?.enabled,
    currentSwarmControlCapabilities.focus,
  );
  assert.equal(
    controls.find(({ type }) => type === "open-agent-details")?.enabled,
    currentSwarmControlCapabilities.details,
  );
  for (const type of [
    "open-send-composer",
    "halt-agent",
    "kill-agent",
    "open-spawn-composer",
  ] as const) {
    const control = controls.find((candidate) => candidate.type === type);
    assert.equal(control?.enabled, false);
    assert.ok(control?.disabledReason);
  }
});

test("a future host can enable native operations without changing the model", () => {
  const futureCapabilities: SwarmControlCapabilities = {
    ...currentSwarmControlCapabilities,
    send: true,
    halt: true,
    kill: true,
    spawn: true,
  };
  const model = createSwarmControlsModel(
    { open: true, agents: [agent("reviewer")] },
    undefined,
    futureCapabilities,
  );

  assert.deepEqual(
    model.agents[0]!.affordances
      .filter(({ type }) => type !== "focus-agent" && type !== "open-agent-details")
      .map(({ type, enabled, disabledReason }) => ({ type, enabled, disabledReason })),
    [
      { type: "open-send-composer", enabled: true, disabledReason: undefined },
      { type: "halt-agent", enabled: true, disabledReason: undefined },
      { type: "kill-agent", enabled: true, disabledReason: undefined },
      { type: "open-spawn-composer", enabled: true, disabledReason: undefined },
    ],
  );
});

test("missing native child labels do not create synthetic agents", () => {
  const root = agent("root", { children: ["missing-child"] });
  const model = createSwarmControlsModel(
    { open: true, agents: [root] },
    undefined,
  );

  assert.deepEqual(model.agents[0]?.details.children, []);
});
