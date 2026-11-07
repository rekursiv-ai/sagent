import { strict as assert } from "node:assert";
import test from "node:test";

import {
  createInitialConversationState,
  type CommandOutputRecord,
} from "../shared/conversationTypes";
import { reduceConversation } from "./conversationStore";

function commandOutput(id: string, text: string): CommandOutputRecord {
  return {
    id,
    agentLabel: "Agent",
    text,
    tone: "note",
    command: "/tasks",
  };
}

test("keeps command output and normal turns in dispatch order", () => {
  let state = createInitialConversationState();

  state = reduceConversation(state, {
    type: "command.output",
    output: commandOutput("command-1", "first command"),
  });
  state = reduceConversation(state, {
    type: "user.submitted",
    text: "normal message",
  });
  state = reduceConversation(state, {
    type: "command.output",
    output: commandOutput("command-2", "second command"),
  });

  assert.deepEqual(
    state.timeline.map((item) => item.id),
    ["command-1", state.turns[0]!.id, "command-2"],
  );
});

test("updates the timeline when a pending turn receives Sagent's turn ID", () => {
  let state = createInitialConversationState();
  state = reduceConversation(state, {
    type: "user.submitted",
    text: "normal message",
  });
  const pendingId = state.turns[0]!.id;

  state = reduceConversation(state, {
    type: "turn.started",
    text: "normal message",
    turnId: "native-turn-1",
  });

  assert.equal(state.turns[0]!.id, "native-turn-1");
  assert.deepEqual(state.timeline, [{ type: "turn", id: "native-turn-1" }]);
  assert.notEqual(pendingId, state.timeline[0]!.id);
});

test("removes command output from the presentation timeline only", () => {
  let state = createInitialConversationState();
  state = reduceConversation(state, {
    type: "command.output",
    output: commandOutput("command-1", "remove me"),
  });
  state = reduceConversation(state, {
    type: "user.submitted",
    text: "keep me",
  });

  state = reduceConversation(state, {
    type: "command.output.removed",
    id: "command-1",
  });

  assert.deepEqual(state.commandOutputs, []);
  assert.deepEqual(state.timeline, [{ type: "turn", id: state.turns[0]!.id }]);
  assert.equal(state.turns[0]!.userText, "keep me");
});

test("keeps live assistant and tool activity in native event order", () => {
  let state = createInitialConversationState();
  state = reduceConversation(state, { type: "user.submitted", text: "work" });
  state = reduceConversation(state, { type: "assistant.delta", text: "before" });
  state = reduceConversation(state, {
    type: "tool.started",
    callId: "a",
    name: "Read",
    summary: "Read a",
  });
  state = reduceConversation(state, {
    type: "tool.started",
    callId: "b",
    name: "Read",
    summary: "Read b",
  });
  state = reduceConversation(state, { type: "assistant.delta", text: "after" });
  state = reduceConversation(state, {
    type: "tool.started",
    callId: "c",
    name: "Write",
    summary: "Write c",
  });
  state = reduceConversation(state, {
    type: "tool.completed",
    callId: "b",
    text: "b result",
    isError: false,
    state: "completed",
  });
  state = reduceConversation(state, {
    type: "tool.completed",
    callId: "a",
    text: "a result",
    isError: false,
    state: "completed",
  });

  const turn = state.turns[0]!;
  assert.deepEqual(
    turn.items.map((item) => item.type === "tool" ? item.activity.callId : item.text),
    ["before", "a", "b", "after", "c"],
  );
  assert.deepEqual(turn.tools.map((tool) => tool.callId), ["a", "b", "c"]);
  assert.equal(turn.tools[0]!.text, "a result");
  assert.equal(turn.tools[1]!.text, "b result");
});

test("hydrates the ordered activity stream from the native session snapshot", () => {
  let state = createInitialConversationState();
  state = reduceConversation(state, {
    type: "session.opened",
    snapshot: {
      version: 1,
      items: [
        { id: "u", type: "user", turnId: "native-1", text: "work" },
        { id: "a1", type: "assistant", turnId: "native-1", text: "before" },
        { id: "ta", type: "tool", turnId: "native-1", callId: "a", name: "Read", status: "completed", text: "a result" },
        { id: "tb", type: "tool", turnId: "native-1", callId: "b", name: "Read", status: "completed", text: "b result" },
        { id: "a2", type: "assistant", turnId: "native-1", text: "after" },
        { id: "tc", type: "tool", turnId: "native-1", callId: "c", name: "Write", status: "completed", text: "c result" },
      ],
    },
  });

  assert.deepEqual(
    state.turns[0]!.items.map((item) => item.type === "tool" ? item.activity.callId : item.text),
    ["before", "a", "b", "after", "c"],
  );
});

test("hydrates the effective model recorded at each session turn", () => {
  let state = createInitialConversationState();
  state = reduceConversation(state, {
    type: "session.opened",
    snapshot: {
      version: 1,
      items: [
        { id: "u1", type: "user", turnId: "turn-1", text: "first", model: "opus" },
        { id: "a1", type: "assistant", turnId: "turn-1", text: "one" },
        { id: "u2", type: "user", turnId: "turn-2", text: "second", model: "sonnet" },
        { id: "a2", type: "assistant", turnId: "turn-2", text: "two" },
      ],
    },
  });

  assert.deepEqual(
    state.turns.map((turn) => turn.modelIdentity?.model),
    ["opus", "sonnet"],
  );
});
